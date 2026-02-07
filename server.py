#!/usr/bin/env python3
"""
Piper TTS Backend Server
Fast local TTS using Piper with pre-loaded models
Grammar correction using Ollama LLM
"""

import io
import wave
import httpx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from piper import PiperVoice

app = FastAPI(title="Piper TTS Server")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2:0.5b"

# Model paths
MODELS_DIR = Path(__file__).parent / "models"
VOICES = {
    "en_US-hfc_female-medium": MODELS_DIR / "en_US-hfc_female-medium.onnx",
    "en_US-hfc_male-medium": MODELS_DIR / "en_US-hfc_male-medium.onnx",
}

# Pre-loaded voice instances for fast inference
loaded_voices: dict[str, PiperVoice] = {}


class TTSRequest(BaseModel):
    text: str
    voice: str = "en_US-hfc_female-medium"
    speaker_id: Optional[int] = None


class GrammarRequest(BaseModel):
    text: str


def get_voice(voice_id: str) -> PiperVoice:
    """Get or load a voice model"""
    if voice_id not in loaded_voices:
        if voice_id not in VOICES:
            raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_id}")
        
        model_path = VOICES[voice_id]
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        
        print(f"Loading voice: {voice_id}")
        loaded_voices[voice_id] = PiperVoice.load(str(model_path))
        print(f"Loaded: {voice_id}")
    
    return loaded_voices[voice_id]


@app.on_event("startup")
async def preload_models():
    """Pre-load all models on startup for minimal latency"""
    print("Pre-loading voice models...")
    for voice_id in VOICES:
        try:
            get_voice(voice_id)
        except Exception as e:
            print(f"Failed to load {voice_id}: {e}")
    print("All models loaded!")
    
    # Warm up the LLM
    print("Warming up LLM...")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": "Hi", "stream": False},
                timeout=60.0
            )
        print("LLM warmed up!")
    except Exception as e:
        print(f"LLM warm-up failed (non-fatal): {e}")


@app.get("/")
async def root():
    return {"status": "ok", "voices": list(VOICES.keys()), "llm": LLM_MODEL}


@app.get("/voices")
async def list_voices():
    return {"voices": list(VOICES.keys())}


@app.post("/correct")
async def correct_grammar(request: GrammarRequest):
    """Convert simplified/broken English to proper English"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    prompt = f"""Convert this simplified or broken English to proper, grammatically correct English. 
Only output the corrected sentence, nothing else. Keep it natural and conversational.

Input: "{request.text}"
Output:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 100
                    }
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM request failed")
            
            result = response.json()
            corrected = result.get("response", "").strip()
            
            # Clean up the response - remove quotes if present
            corrected = corrected.strip('"\'')
            
            return {"original": request.text, "corrected": corrected}
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """Synthesize speech from text"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    voice = get_voice(request.voice)
    
    # Synthesize - returns iterator of AudioChunk
    audio_chunks = list(voice.synthesize(request.text))
    
    if not audio_chunks:
        raise HTTPException(status_code=500, detail="No audio generated")
    
    # Get audio properties from first chunk
    first_chunk = audio_chunks[0]
    sample_rate = first_chunk.sample_rate
    sample_width = first_chunk.sample_width
    channels = first_chunk.sample_channels
    
    # Combine all audio bytes
    audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    
    wav_buffer.seek(0)
    
    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

