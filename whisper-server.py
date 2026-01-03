#!/usr/bin/env python3
"""
Ultra-Fast Whisper API Server for Community Captioner

This server keeps the Whisper model warm in memory for instant transcription.
OpenAI-compatible API at /v1/audio/transcriptions

Usage:
    python3 whisper-server.py [--model tiny.en] [--port 8000]

Then configure Community Captioner to use "Local Whisper API" mode with:
    URL: http://localhost:8000/v1/audio/transcriptions
"""

import argparse
import io
import time
import wave
import numpy as np
from typing import Optional

# Check dependencies
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip3 install fastapi uvicorn python-multipart")
    exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster-whisper not installed. Install with:")
    print("   pip3 install faster-whisper")
    exit(1)

# Global model instance (kept warm)
model: Optional[WhisperModel] = None
model_name: str = ""
device: str = "cpu"
compute_type: str = "int8"

app = FastAPI(title="Whisper API Server", version="1.0.0")

# Performance stats
stats = {
    "requests": 0,
    "total_audio_seconds": 0,
    "total_transcription_ms": 0,
    "avg_rtf": 0  # Real-time factor
}


def load_model(name: str = "tiny.en"):
    """Load and warm up the Whisper model"""
    global model, model_name, device, compute_type

    # Detect best device
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print(f"🚀 NVIDIA GPU detected - using CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "cpu"  # MPS not supported by faster-whisper
            compute_type = "int8"
            print(f"🍎 Apple Silicon detected - using optimized CPU")
        else:
            print(f"💻 Using CPU with int8 quantization")
    except ImportError:
        print(f"💻 Using CPU with int8 quantization")

    import os
    cpu_threads = min(os.cpu_count() or 4, 8)

    print(f"🔄 Loading Whisper model '{name}'...")
    start = time.time()

    model = WhisperModel(
        name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1
    )
    model_name = name

    # Warm up with a dummy transcription
    print("🔥 Warming up model...")
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    list(model.transcribe(dummy_audio, beam_size=1, without_timestamps=True))

    load_time = time.time() - start
    print(f"✅ Model '{name}' ready in {load_time:.1f}s ({device}, {cpu_threads} threads)")


@app.get("/")
async def root():
    """Health check and status"""
    return {
        "status": "ok",
        "model": model_name,
        "device": device,
        "stats": stats
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatible)"""
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model", "owned_by": "faster-whisper"}
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default="en"),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-compatible transcription endpoint.
    Optimized for real-time captioning with ultra-low latency.
    """
    global stats

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start_time = time.time()

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Convert to numpy array
        # Support WAV format (most common for real-time audio)
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav:
                    sample_rate = wav.getframerate()
                    n_frames = wav.getnframes()
                    audio_data = wav.readframes(n_frames)

                    # Convert to float32
                    if wav.getsampwidth() == 2:
                        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif wav.getsampwidth() == 4:
                        audio = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        audio = np.frombuffer(audio_data, dtype=np.float32)

                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        # Simple resampling (good enough for speech)
                        ratio = 16000 / sample_rate
                        new_length = int(len(audio) * ratio)
                        audio = np.interp(
                            np.linspace(0, len(audio), new_length),
                            np.arange(len(audio)),
                            audio
                        ).astype(np.float32)
        except Exception as e:
            # Try raw float32 audio
            audio = np.frombuffer(audio_bytes, dtype=np.float32)

        audio_duration = len(audio) / 16000

        # Transcribe with fastest settings
        transcribe_start = time.time()
        segments, info = globals()['model'].transcribe(
            audio,
            beam_size=1,
            best_of=1,
            language=language if language else "en",
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=temperature,
        )

        # Collect text
        text = " ".join(seg.text for seg in segments).strip()

        transcribe_time = (time.time() - transcribe_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # Update stats
        stats["requests"] += 1
        stats["total_audio_seconds"] += audio_duration
        stats["total_transcription_ms"] += transcribe_time
        if stats["total_audio_seconds"] > 0:
            stats["avg_rtf"] = (stats["total_transcription_ms"] / 1000) / stats["total_audio_seconds"]

        # Return based on format
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return {
                "text": text,
                "language": language,
                "duration": audio_duration,
                "transcription_ms": round(transcribe_time, 1),
                "total_ms": round(total_time, 1)
            }
        else:  # json (default)
            return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    response_format: str = Form(default="json"),
):
    """Translate audio to English (OpenAI API compatible)"""
    # For now, just transcribe with English
    return await transcribe(file, model, "en", response_format, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Whisper API Server")
    parser.add_argument("--model", default="tiny.en", help="Whisper model (tiny.en, base.en, small.en)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║           🎤 Ultra-Fast Whisper API Server 🎤                    ║
╠══════════════════════════════════════════════════════════════════╣
║  OpenAI-compatible API for real-time transcription               ║
║  Model stays warm in memory for instant response                 ║
╚══════════════════════════════════════════════════════════════════╝
""")

    load_model(args.model)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  API Endpoint: http://localhost:{args.port}/v1/audio/transcriptions
║  Health Check: http://localhost:{args.port}/
║                                                                  ║
║  Configure Community Captioner "Local Whisper API" with:         ║
║    URL: http://localhost:{args.port}/v1/audio/transcriptions
╚══════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
