# Large Language Model (LLM) Module
# Supports: OpenAI GPT-4o-mini (default), Mistral-7B, Llama-2, Phi-3, Ollama (local)

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.models import RankedMoment, TranscriptSegment

logger = logging.getLogger(__name__)


def chunk_segments(
    segments: list[TranscriptSegment],
    chunk_seconds: float = 30.0,
) -> list[dict]:
    """
    Group consecutive transcript segments into time-based chunks so the
    LLM has manageable, timestamp-anchored context windows.
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_start = segments[0].start
    current_texts: list[str] = []
    current_end = segments[0].start

    for seg in segments:
        if seg.start - current_start >= chunk_seconds and current_texts:
            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts).strip(),
            })
            current_start = seg.start
            current_texts = []

        current_texts.append(seg.text)
        current_end = seg.end

    if current_texts:
        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts).strip(),
        })

    return chunks


# ============================================================================
# LLM PROVIDER CLASSES
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank and extract moments from transcript chunks"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available/configured"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o-mini for moment ranking"""

    SYSTEM_PROMPT = """You are Klipp, an assistant that finds the most \
compelling short-form clip moments from a video transcript.

You will receive a transcript broken into chunks, each with a start and \
end time in seconds. Identify the moments most likely to make engaging \
short clips: strong hooks, emotional peaks, surprising statements, \
actionable advice, or punchy one-liners.

Respond ONLY with JSON in this exact shape, and nothing else:
{
  "moments": [
    {"start": <number>, "end": <number>, "reason": "<short reason>", "score": <0-10>}
  ]
}

Rules:
- "start" and "end" must be real timestamps taken from the transcript chunks provided.
- Each moment's duration (end - start) must be between min_clip_seconds and max_clip_seconds.
- Order moments by score, highest first.
- Return at most max_clips moments.
- Do not include any text outside the JSON object.
"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"
        if not self.api_key:
            logger.warning("OpenAI provider: OPENAI_API_KEY not set")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank moments using GPT-4o-mini"""
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available: Missing OPENAI_API_KEY")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            transcript_blob = "\n".join(
                f"[{c['start']:.2f} - {c['end']:.2f}] {c['text']}" for c in chunks
            )

            user_prompt = (
                f"max_clips: {max_clips}\n"
                f"min_clip_seconds: {min_clip_seconds}\n"
                f"max_clip_seconds: {max_clip_seconds}\n\n"
                f"Transcript chunks:\n{transcript_blob}"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            return self._parse_response(response.choices[0].message.content, max_clips, min_clip_seconds, max_clip_seconds)

        except Exception as e:
            logger.error(f"OpenAI ranking failed: {e}")
            raise

    def _parse_response(self, content: str, max_clips: int, min_clip_seconds: int, max_clip_seconds: int) -> list[RankedMoment]:
        """Parse and validate LLM response"""
        if not content:
            raise ValueError("LLM returned an empty response while ranking moments")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON while ranking moments: {exc}") from exc

        raw_moments = data.get("moments", [])
        if not isinstance(raw_moments, list):
            raise ValueError("LLM response did not contain a 'moments' list")

        moments: list[RankedMoment] = []
        for m in raw_moments:
            try:
                moment = RankedMoment(**m)
            except (TypeError, ValueError):
                continue

            duration = moment.end - moment.start
            if duration <= 0:
                continue
            if duration < min_clip_seconds or duration > max_clip_seconds:
                continue

            moments.append(moment)

        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:max_clips]


class MistralProvider(LLMProvider):
    """Mistral-7B for moment ranking (local or API)"""

    def __init__(self, api_key: Optional[str] = None, use_local: bool = False):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.use_local = use_local
        self.model = "mistral-7b-instruct-v0.1"

    def is_available(self) -> bool:
        if self.use_local:
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                return response.status_code == 200
            except:
                return False
        return bool(self.api_key)

    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank moments using Mistral"""
        if not self.is_available():
            raise RuntimeError("Mistral provider not available")

        try:
            transcript_blob = "\n".join(
                f"[{c['start']:.2f} - {c['end']:.2f}] {c['text']}" for c in chunks
            )

            prompt = f"""Find the most engaging moments from this transcript.

max_clips: {max_clips}
min_clip_seconds: {min_clip_seconds}
max_clip_seconds: {max_clip_seconds}

Respond ONLY with JSON:
{{
  "moments": [
    {{"start": <number>, "end": <number>, "reason": "<reason>", "score": <0-10>}}
  ]
}}

Transcript chunks:
{transcript_blob}"""

            if self.use_local:
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                result_text = response.json()["response"]
            else:
                from mistralai.client import MistralClient
                client = MistralClient(api_key=self.api_key)
                message = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = message.choices[0].message.content

            return self._parse_response(result_text, max_clips, min_clip_seconds, max_clip_seconds)

        except Exception as e:
            logger.error(f"Mistral ranking failed: {e}")
            raise

    def _parse_response(self, content: str, max_clips: int, min_clip_seconds: int, max_clip_seconds: int) -> list[RankedMoment]:
        """Parse and validate LLM response"""
        try:
            result = json.loads(content)
            raw_moments = result.get("moments", [])
        except json.JSONDecodeError:
            logger.warning("Mistral returned non-JSON response, attempting recovery")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_moments = result.get("moments", [])
            else:
                raise ValueError("Could not parse Mistral response")

        moments: list[RankedMoment] = []
        for m in raw_moments:
            try:
                moment = RankedMoment(**m)
            except (TypeError, ValueError):
                continue

            duration = moment.end - moment.start
            if duration <= 0:
                continue
            if duration < min_clip_seconds or duration > max_clip_seconds:
                continue

            moments.append(moment)

        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:max_clips]


class LlamaProvider(LLMProvider):
    """Llama-2 local inference for moment ranking"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH", "./models/llama-2-7b-chat.gguf")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Llama model"""
        try:
            from llama_cpp import Llama
            if os.path.exists(self.model_path):
                self.model = Llama(model_path=self.model_path)
                logger.info(f"Llama model loaded: {self.model_path}")
            else:
                logger.warning(f"Llama model not found at {self.model_path}")
        except Exception as e:
            logger.warning(f"Llama model failed to load: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank moments using Llama-2"""
        if not self.is_available():
            raise RuntimeError("Llama provider not available: Model failed to load")

        try:
            transcript_blob = "\n".join(
                f"[{c['start']:.2f} - {c['end']:.2f}] {c['text']}" for c in chunks
            )

            prompt = f"""[INST] Find the most engaging moments from this transcript.

max_clips: {max_clips}
min_clip_seconds: {min_clip_seconds}
max_clip_seconds: {max_clip_seconds}

Respond ONLY with JSON:
{{
  "moments": [
    {{"start": <number>, "end": <number>, "reason": "<reason>", "score": <0-10>}}
  ]
}}

Transcript chunks:
{transcript_blob} [/INST]"""

            output = self.model(
                prompt,
                max_tokens=2000,
                stop=["[INST]"],
            )

            result_text = output["choices"][0]["text"]
            return self._parse_response(result_text, max_clips, min_clip_seconds, max_clip_seconds)

        except Exception as e:
            logger.error(f"Llama ranking failed: {e}")
            raise

    def _parse_response(self, content: str, max_clips: int, min_clip_seconds: int, max_clip_seconds: int) -> list[RankedMoment]:
        """Parse and validate LLM response"""
        try:
            result = json.loads(content)
            raw_moments = result.get("moments", [])
        except json.JSONDecodeError:
            logger.warning("Llama returned non-JSON response, attempting recovery")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_moments = result.get("moments", [])
            else:
                raise ValueError("Could not parse Llama response")

        moments: list[RankedMoment] = []
        for m in raw_moments:
            try:
                moment = RankedMoment(**m)
            except (TypeError, ValueError):
                continue

            duration = moment.end - moment.start
            if duration <= 0:
                continue
            if duration < min_clip_seconds or duration > max_clip_seconds:
                continue

            moments.append(moment)

        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:max_clips]


class PhiProvider(LLMProvider):
    """Microsoft Phi-3 for moment ranking (efficient small model)"""

    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Phi-3 model"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

            logger.info(f"Phi-3 model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Phi-3 model failed to load: {e}")

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank moments using Phi-3"""
        if not self.is_available():
            raise RuntimeError("Phi-3 provider not available: Model failed to load")

        try:
            import torch

            transcript_blob = "\n".join(
                f"[{c['start']:.2f} - {c['end']:.2f}] {c['text']}" for c in chunks
            )

            prompt = f"""Find the most engaging moments from this transcript.

max_clips: {max_clips}
min_clip_seconds: {min_clip_seconds}
max_clip_seconds: {max_clip_seconds}

Respond ONLY with JSON:
{{
  "moments": [
    {{"start": <number>, "end": <number>, "reason": "<reason>", "score": <0-10>}}
  ]
}}

Transcript chunks:
{transcript_blob}"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    temperature=0.7,
                    do_sample=True,
                )

            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_response(result_text, max_clips, min_clip_seconds, max_clip_seconds)

        except Exception as e:
            logger.error(f"Phi-3 ranking failed: {e}")
            raise

    def _parse_response(self, content: str, max_clips: int, min_clip_seconds: int, max_clip_seconds: int) -> list[RankedMoment]:
        """Parse and validate LLM response"""
        try:
            result = json.loads(content)
            raw_moments = result.get("moments", [])
        except json.JSONDecodeError:
            logger.warning("Phi-3 returned non-JSON response, attempting recovery")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_moments = result.get("moments", [])
            else:
                raise ValueError("Could not parse Phi-3 response")

        moments: list[RankedMoment] = []
        for m in raw_moments:
            try:
                moment = RankedMoment(**m)
            except (TypeError, ValueError):
                continue

            duration = moment.end - moment.start
            if duration <= 0:
                continue
            if duration < min_clip_seconds or duration > max_clip_seconds:
                continue

            moments.append(moment)

        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:max_clips]


class OllamaProvider(LLMProvider):
    """Ollama local LLM serving (any model)"""

    def __init__(self, model_name: str = "mistral", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def rank_moments(
        self,
        chunks: list[dict],
        max_clips: int,
        min_clip_seconds: int,
        max_clip_seconds: int,
    ) -> list[RankedMoment]:
        """Rank moments using Ollama"""
        if not self.is_available():
            raise RuntimeError(f"Ollama provider not available: Not running at {self.endpoint}")

        try:
            import requests

            transcript_blob = "\n".join(
                f"[{c['start']:.2f} - {c['end']:.2f}] {c['text']}" for c in chunks
            )

            prompt = f"""Find the most engaging moments from this transcript.

max_clips: {max_clips}
min_clip_seconds: {min_clip_seconds}
max_clip_seconds: {max_clip_seconds}

Respond ONLY with JSON:
{{
  "moments": [
    {{"start": <number>, "end": <number>, "reason": "<reason>", "score": <0-10>}}
  ]
}}

Transcript chunks:
{transcript_blob}"""

            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300,
            )

            result_text = response.json()["response"]
            return self._parse_response(result_text, max_clips, min_clip_seconds, max_clip_seconds)

        except Exception as e:
            logger.error(f"Ollama ranking failed: {e}")
            raise

    def _parse_response(self, content: str, max_clips: int, min_clip_seconds: int, max_clip_seconds: int) -> list[RankedMoment]:
        """Parse and validate LLM response"""
        try:
            result = json.loads(content)
            raw_moments = result.get("moments", [])
        except json.JSONDecodeError:
            logger.warning("Ollama returned non-JSON response, attempting recovery")
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_moments = result.get("moments", [])
            else:
                raise ValueError("Could not parse Ollama response")

        moments: list[RankedMoment] = []
        for m in raw_moments:
            try:
                moment = RankedMoment(**m)
            except (TypeError, ValueError):
                continue

            duration = moment.end - moment.start
            if duration <= 0:
                continue
            if duration < min_clip_seconds or duration > max_clip_seconds:
                continue

            moments.append(moment)

        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:max_clips]


# ============================================================================
# FACTORY & CONVENIENCE FUNCTIONS
# ============================================================================

class LLMFactory:
    """Factory for creating LLM provider instances"""

    _providers: Dict[str, LLMProvider] = {}

    @classmethod
    def get_provider(cls, provider_name: str = "openai") -> LLMProvider:
        """Get or create a provider instance"""
        if provider_name in cls._providers:
            return cls._providers[provider_name]

        if provider_name == "openai":
            provider = OpenAIProvider()
        elif provider_name == "mistral":
            provider = MistralProvider()
        elif provider_name == "llama":
            provider = LlamaProvider()
        elif provider_name == "phi":
            provider = PhiProvider()
        elif provider_name == "local":
            provider = OllamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        cls._providers[provider_name] = provider
        return provider

    @classmethod
    def rank_moments(
        cls,
        chunks: list[dict],
        provider_name: str = "openai",
        max_clips: int = 3,
        min_clip_seconds: int = 15,
        max_clip_seconds: int = 60,
    ) -> list[RankedMoment]:
        """Convenience method to rank moments with specified provider"""
        provider = cls.get_provider(provider_name)

        if not provider.is_available() and provider_name != "openai":
            logger.warning(f"Provider {provider_name} not available, falling back to OpenAI")
            provider = cls.get_provider("openai")

        return provider.rank_moments(chunks, max_clips, min_clip_seconds, max_clip_seconds)


def rank_moments(
    chunks: list[dict],
    max_clips: int,
    min_clip_seconds: int,
    max_clip_seconds: int,
    provider: str = "openai",
) -> list[RankedMoment]:
    """
    Compatibility function that maintains the original API while supporting
    multiple providers. Defaults to OpenAI GPT-4o-mini (the supported,
    working default). Mistral/Llama/Phi/Ollama are available as opt-in,
    experimental alternatives for avoiding per-call API costs, but are not
    production-hardened yet.
    """
    return LLMFactory.rank_moments(chunks, provider, max_clips, min_clip_seconds, max_clip_seconds)
