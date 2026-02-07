#!/usr/bin/env python3
"""
Piper TTS Backend Server
Fast local TTS using Piper with pre-loaded models
"""

import io
import wave
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


@app.get("/")
async def root():
    return {"status": "ok", "voices": list(VOICES.keys())}


@app.get("/voices")
async def list_voices():
    return {"voices": list(VOICES.keys())}


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
