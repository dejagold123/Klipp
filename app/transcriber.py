# Speech-to-Text (Transcription) Module - Multi-Provider
# Supports: OpenAI Whisper (default), Vosk, wav2vec 2.0

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Optional

from app.models import TranscriptSegment

logger = logging.getLogger(__name__)

# Whisper API limit is 25 MB; keep some headroom for multipart overhead.
MAX_AUDIO_BYTES = 24 * 1024 * 1024

# Bitrate used when extracting audio. Used to estimate a safe per-chunk duration when splitting.
AUDIO_BITRATE_KBPS = 64


def extract_audio(video_path: str, output_dir: str, job_id: str) -> str:
    """Extract a compressed mono audio track suitable for transcription."""
    audio_path = os.path.join(output_dir, f"{job_id}.mp3")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", f"{AUDIO_BITRATE_KBPS}k",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Audio extracted: {audio_path}")
    return audio_path


def get_audio_duration(audio_path: str) -> float:
    """Return the duration of `audio_path` in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def split_audio(audio_path: str, output_dir: str, job_id: str) -> list[tuple[str, float]]:
    """
    Split `audio_path` into sequential chunks that each fit under MAX_AUDIO_BYTES.
    
    Returns a list of (chunk_path, offset_seconds) tuples, where offset_seconds 
    is the chunk's ACTUAL start time within the original audio.
    """
    if os.path.getsize(audio_path) <= MAX_AUDIO_BYTES:
        return [(audio_path, 0.0)]

    # Estimate a safe chunk duration with 10% safety margin
    bytes_per_second = (AUDIO_BITRATE_KBPS * 1000) / 8
    chunk_seconds = int((MAX_AUDIO_BYTES / bytes_per_second) * 0.9)
    chunk_seconds = max(chunk_seconds, 60)  # sanity floor

    pattern = os.path.join(output_dir, f"{job_id}_chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    chunk_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith(f"{job_id}_chunk_") and f.endswith(".mp3")
    )

    if not chunk_files:
        return [(audio_path, 0.0)]

    # Accumulate offsets from REAL chunk durations
    chunks: list[tuple[str, float]] = []
    offset = 0.0
    for fname in chunk_files:
        chunk_path = os.path.join(output_dir, fname)
        chunks.append((chunk_path, offset))
        offset += get_audio_duration(chunk_path)

    logger.info(f"Audio split into {len(chunks)} chunks")
    return chunks


# ============================================================================
# STT PROVIDER CLASSES
# ============================================================================

class STTProvider(ABC):
    """Abstract base class for speech-to-text providers"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe audio file to text with timestamps"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available/configured"""
        pass


class WhisperProvider(STTProvider):
    """OpenAI Whisper API transcription"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("Whisper provider: OPENAI_API_KEY not set")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe using OpenAI Whisper API with chunk support"""
        if not self.is_available():
            raise RuntimeError("Whisper provider not available: Missing OPENAI_API_KEY")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            output_dir = os.path.dirname(audio_path) or "."
            job_id = os.path.splitext(os.path.basename(audio_path))[0]

            chunks = split_audio(audio_path, output_dir, job_id)

            all_segments: list[TranscriptSegment] = []
            for chunk_path, offset in chunks:
                chunk_segments = self._transcribe_chunk(chunk_path, client)
                for seg in chunk_segments:
                    all_segments.append(TranscriptSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                    ))

                # Clean up split chunk files
                if chunk_path != audio_path:
                    os.remove(chunk_path)

            logger.info(f"Whisper transcription completed: {len(all_segments)} segments")
            return all_segments
        
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def _transcribe_chunk(self, audio_path: str, client) -> list[TranscriptSegment]:
        """Transcribe a single audio chunk"""
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in response.segments:
            start = seg["start"] if isinstance(seg, dict) else seg.start
            end = seg["end"] if isinstance(seg, dict) else seg.end
            text = seg["text"] if isinstance(seg, dict) else seg.text
            segments.append(TranscriptSegment(start=start, end=end, text=text.strip()))

        return segments


class VoskProvider(STTProvider):
    """Vosk offline speech-to-text (lightweight, local, no API key required)"""

    def __init__(self, model_name: str = "en"):
        # `model_name` here is actually a language code (e.g. "en"). Vosk's
        # constructor needs that passed as `lang=`, not as the positional
        # `model_path` arg - otherwise it tries to open a folder literally
        # named "en" and always fails. `lang=` triggers Vosk's built-in
        # auto-download/cache of an appropriate model for that language.
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Vosk model"""
        try:
            import vosk
            self.model = vosk.Model(lang=self.model_name)
            logger.info(f"Vosk model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Vosk model failed to load: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe using Vosk (local, offline)"""
        if not self.is_available():
            raise RuntimeError("Vosk provider not available: Model failed to load")

        try:
            import vosk
            import wave
            import json

            recognizer = vosk.KaldiRecognizer(self.model, 16000)
            wf = wave.open(audio_path, 'rb')
            
            segments: list[TranscriptSegment] = []
            current_time = 0.0
            chunk_duration = 4000 / 16000  # 4000 samples at 16kHz

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if "result" in result and result["result"]:
                        text = " ".join(item.get("conf", "") for item in result["result"])
                        if text.strip():
                            segments.append(TranscriptSegment(
                                start=current_time,
                                end=current_time + chunk_duration,
                                text=text.strip()
                            ))
                
                current_time += chunk_duration

            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            if "result" in final_result and final_result["result"]:
                text = " ".join(item.get("conf", "") for item in final_result["result"])
                if text.strip():
                    segments.append(TranscriptSegment(
                        start=current_time,
                        end=current_time + chunk_duration,
                        text=text.strip()
                    ))

            wf.close()
            logger.info(f"Vosk transcription completed: {len(segments)} segments")
            return segments
        
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            raise


class Wav2VecProvider(STTProvider):
    """Meta wav2vec 2.0 - state-of-the-art speech recognition"""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load wav2vec 2.0 model from HuggingFace"""
        try:
            import torch
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            logger.info(f"wav2vec 2.0 model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"wav2vec 2.0 model failed to load: {e}")

    def is_available(self) -> bool:
        return self.model is not None and self.processor is not None

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe using wav2vec 2.0"""
        if not self.is_available():
            raise RuntimeError("wav2vec 2.0 provider not available: Model failed to load")

        try:
            import torch
            import librosa

            # Load audio
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Process input
            inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Inference
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.model.device)).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0]
            
            # For simplicity, return one segment for the entire audio
            duration = librosa.get_duration(y=speech, sr=16000)
            segments = [TranscriptSegment(start=0.0, end=duration, text=transcript)]
            
            logger.info(f"wav2vec 2.0 transcription completed: {len(segments)} segment(s)")
            return segments
        
        except Exception as e:
            logger.error(f"wav2vec 2.0 transcription failed: {e}")
            raise


class STTFactory:
    """Factory for creating STT provider instances"""

    _providers: Dict[str, STTProvider] = {}

    @classmethod
    def get_provider(cls, provider_name: str = "openai") -> STTProvider:
        """Get or create a provider instance"""
        if provider_name in cls._providers:
            return cls._providers[provider_name]

        if provider_name == "openai":
            provider = WhisperProvider()
        elif provider_name == "vosk":
            provider = VoskProvider()
        elif provider_name == "wav2vec":
            provider = Wav2VecProvider()
        else:
            raise ValueError(f"Unknown STT provider: {provider_name}")

        cls._providers[provider_name] = provider
        return provider

    @classmethod
    def transcribe(cls, audio_path: str, provider_name: str = "openai") -> list[TranscriptSegment]:
        """Convenience method to transcribe with specified provider"""
        provider = cls.get_provider(provider_name)

        if not provider.is_available() and provider_name != "openai":
            logger.warning(f"Provider {provider_name} not available, falling back to OpenAI Whisper")
            provider = cls.get_provider("openai")

        return provider.transcribe(audio_path)


# ============================================================================
# LEGACY FUNCTION FOR COMPATIBILITY
# ============================================================================

def transcribe_audio(audio_path: str, provider: str = "openai") -> list[TranscriptSegment]:
    """
    Compatibility function that maintains the original API while supporting
    multiple providers. Defaults to OpenAI Whisper (the supported, working
    default). Vosk/wav2vec are available as opt-in, experimental alternatives
    for avoiding per-call API costs, but are not production-hardened yet.
    """
    return STTFactory.transcribe(audio_path, provider)
