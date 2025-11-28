"""
LLM Provider Abstraction Layer

Supports multiple LLM backends:
- OpenAI (GPT-4o-mini, GPT-4o, etc.)
- Anthropic (Claude Haiku, Sonnet, Opus)
- Google (Gemini Flash, Pro)
- LM Studio (local models via OpenAI-compatible API)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from functools import lru_cache

import requests


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.default_model
        self._session = requests.Session()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display"""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model to use (should be cheap/fast)"""
        pass

    @property
    @abstractmethod
    def available_models(self) -> list:
        """List of available models for this provider"""
        pass

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            system_prompt: The system/instruction prompt
            user_prompt: The user's message

        Returns:
            LLMResponse with the model's response
        """
        pass

    def is_configured(self) -> bool:
        """Check if the provider has necessary configuration (API key, etc.)"""
        return self.api_key is not None and len(self.api_key) > 0


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT models)"""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def available_models(self) -> list:
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        response = self._session.post(self.BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude models)"""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    @property
    def name(self) -> str:
        return "Anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-5-haiku-latest"

    @property
    def available_models(self) -> list:
        return ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        response = self._session.post(self.BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data
        )


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash-lite"

    @property
    def available_models(self) -> list:
        return ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-pro"]

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }

        response = self._session.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        content = data["candidates"][0]["content"]["parts"][0]["text"]

        return LLMResponse(
            content=content,
            model=self.model,
            usage=data.get("usageMetadata"),
            raw_response=data
        )


class LMStudioProvider(LLMProvider):
    """LM Studio local inference (OpenAI-compatible API)"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: str = "http://localhost:1234/v1"):
        # Normalize base_url - strip any path beyond /v1
        # Users might paste http://host:port/v1/chat/completions by mistake
        if "/v1/" in base_url:
            base_url = base_url.split("/v1/")[0] + "/v1"
        elif base_url.endswith("/v1"):
            pass  # Already correct
        elif "/v1" not in base_url:
            # Add /v1 if missing entirely
            base_url = base_url.rstrip("/") + "/v1"

        self.base_url = base_url
        super().__init__(api_key or "lm-studio", model)  # LM Studio doesn't need a real key

    @property
    def name(self) -> str:
        return "LM Studio (Local)"

    @property
    def default_model(self) -> str:
        return "local-model"  # LM Studio uses whatever model is loaded

    @property
    def available_models(self) -> list:
        # LM Studio serves whatever model is currently loaded
        return ["local-model"]

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        response = self._session.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", "local-model"),
            usage=data.get("usage"),
            raw_response=data
        )

    def is_configured(self) -> bool:
        """LM Studio just needs the server to be running"""
        try:
            response = self._session.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# ==========================================
# Provider Registry & Factory
# ==========================================

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "lmstudio": LMStudioProvider
}


def get_provider(provider_name: str, api_key: Optional[str] = None,
                 model: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to get a configured LLM provider.

    Args:
        provider_name: One of 'openai', 'anthropic', 'gemini', 'lmstudio'
        api_key: API key for the provider (not needed for lmstudio)
        model: Specific model to use (defaults to provider's cheap/fast model)
        **kwargs: Additional provider-specific arguments (e.g., base_url for lmstudio)

    Returns:
        Configured LLMProvider instance
    """
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[provider_name](api_key=api_key, model=model, **kwargs)


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get info about all available providers and their models.

    Returns:
        Dict with provider info including available models
    """
    result = {}
    for name, provider_class in PROVIDERS.items():
        # Create a temporary instance to get model info
        temp = provider_class.__new__(provider_class)
        temp.api_key = None
        temp.model = None
        result[name] = {
            "display_name": temp.name,
            "default_model": temp.default_model,
            "models": temp.available_models
        }
    return result
