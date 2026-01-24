"""Unified LLM provider abstraction for memex.

This module provides a consistent interface for making LLM calls using
either Anthropic's direct API or OpenRouter (OpenAI-compatible gateway).

Provider detection:
1. If llm.provider explicitly set in .kbconfig -> use that
2. If only ANTHROPIC_API_KEY set -> use anthropic
3. If only OPENROUTER_API_KEY set -> use openrouter
4. If both keys set and no explicit config -> error
5. If no keys set -> error

Usage:
    # Sync
    client, provider = get_sync_client()

    # Async (for llm.py)
    client, provider = get_async_client()

    # Model resolution
    model = resolve_model("claude-3.5-haiku", provider)
"""

from __future__ import annotations

import os
from typing import Any

from .config import LLMConfig, get_llm_config


class LLMProviderError(Exception):
    """Raised when LLM provider is misconfigured or unavailable."""

    pass


# Backwards compatibility alias
LLMConfigurationError = LLMProviderError


# =============================================================================
# Model Name Translation
# =============================================================================

# Canonical model names mapped to (anthropic_name, openrouter_name)
# Users can specify any of these and they'll be translated appropriately
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Haiku models
    "claude-3-haiku": ("claude-3-haiku-20240307", "anthropic/claude-3-haiku"),
    "claude-3.5-haiku": ("claude-3-5-haiku-20241022", "anthropic/claude-3-5-haiku"),
    "claude-haiku-4.5": ("claude-haiku-4-5-20250414", "anthropic/claude-haiku-4.5"),
    # Sonnet models
    "claude-3.5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic/claude-3.5-sonnet"),
    "claude-sonnet-4": ("claude-sonnet-4-20250514", "anthropic/claude-sonnet-4"),
    # Opus models
    "claude-3-opus": ("claude-3-opus-20240229", "anthropic/claude-3-opus"),
    "claude-opus-4": ("claude-opus-4-20250514", "anthropic/claude-opus-4"),
}


def resolve_model(model: str, provider: str) -> str:
    """Resolve a model name to provider-specific format.

    Handles:
    - Canonical names (e.g., "claude-3.5-haiku") -> translated per provider
    - Already provider-specific names -> passed through unchanged
    - OpenRouter format for Anthropic -> stripped of "anthropic/" prefix

    Args:
        model: Model name (canonical or provider-specific).
        provider: Target provider ('anthropic' or 'openrouter').

    Returns:
        Provider-appropriate model name.

    Examples:
        >>> resolve_model("claude-3.5-haiku", "anthropic")
        "claude-3-5-haiku-20241022"
        >>> resolve_model("claude-3.5-haiku", "openrouter")
        "anthropic/claude-3-5-haiku"
        >>> resolve_model("anthropic/claude-3-5-haiku", "anthropic")
        "claude-3-5-haiku"  # strips prefix
    """
    # Check for canonical alias
    if model in MODEL_ALIASES:
        anthropic_name, openrouter_name = MODEL_ALIASES[model]
        return anthropic_name if provider == "anthropic" else openrouter_name

    # Handle OpenRouter format being used with Anthropic provider
    if provider == "anthropic" and model.startswith("anthropic/"):
        return model.removeprefix("anthropic/")

    # Handle Anthropic format being used with OpenRouter provider
    if provider == "openrouter" and not model.startswith("anthropic/"):
        # Check if it looks like an Anthropic model name
        if model.startswith("claude-"):
            return f"anthropic/{model}"

    # Pass through as-is
    return model


# =============================================================================
# Provider Detection
# =============================================================================

# Error messages for helpful user guidance
_ERROR_BOTH_KEYS = """\
Both ANTHROPIC_API_KEY and OPENROUTER_API_KEY are set.
Please specify which provider to use in .kbconfig:

llm:
  provider: anthropic  # or 'openrouter'
"""

_ERROR_NO_KEY = """\
No LLM API key configured.

Set one of:
  ANTHROPIC_API_KEY - for direct Anthropic API access
  OPENROUTER_API_KEY - for OpenRouter (multi-model gateway)

Get keys at:
  https://console.anthropic.com/
  https://openrouter.ai/keys
"""


def detect_provider(config: LLMConfig | None = None) -> str:
    """Detect which LLM provider to use.

    Priority:
    1. Explicit llm.provider in config
    2. If only ANTHROPIC_API_KEY set -> anthropic
    3. If only OPENROUTER_API_KEY set -> openrouter
    4. If both keys set and no explicit config -> error
    5. If no keys set -> error

    Args:
        config: Optional LLMConfig. If None, loads from .kbconfig.

    Returns:
        Provider name: 'anthropic' or 'openrouter'.

    Raises:
        LLMProviderError: If provider cannot be determined.
    """
    if config is None:
        config = get_llm_config()

    # 1. Explicit config takes precedence
    if config.provider:
        provider = config.provider.lower()
        if provider not in ("anthropic", "openrouter"):
            raise LLMProviderError(
                f"Invalid llm.provider '{config.provider}'. "
                "Must be 'anthropic' or 'openrouter'."
            )
        return provider

    # 2. Auto-detect from environment
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

    if has_anthropic and has_openrouter:
        raise LLMProviderError(_ERROR_BOTH_KEYS)

    if has_anthropic:
        return "anthropic"

    if has_openrouter:
        return "openrouter"

    raise LLMProviderError(_ERROR_NO_KEY)


# =============================================================================
# Sync Clients
# =============================================================================


def get_sync_client(provider: str | None = None) -> tuple[Any, str]:
    """Get a synchronous LLM client.

    Args:
        provider: Explicit provider name, or None to auto-detect.

    Returns:
        Tuple of (client, provider_name) where:
        - client is anthropic.Anthropic or openai.OpenAI
        - provider_name is 'anthropic' or 'openrouter'

    Raises:
        LLMProviderError: If provider cannot be determined or SDK not installed.
    """
    if provider is None:
        provider = detect_provider()

    if provider == "anthropic":
        return _get_anthropic_sync_client(), provider
    else:
        return _get_openrouter_sync_client(), provider


def _get_anthropic_sync_client():
    """Get a synchronous Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get an API key at https://console.anthropic.com/"
        )

    try:
        import anthropic
    except ImportError:
        raise LLMProviderError(
            "anthropic package is required for Anthropic provider. "
            "Install with: uv add anthropic"
        )

    return anthropic.Anthropic(api_key=api_key)


def _get_openrouter_sync_client():
    """Get a synchronous OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Get an API key at https://openrouter.ai/keys"
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise LLMProviderError(
            "openai package is required for OpenRouter provider. "
            "Install with: uv add openai"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# =============================================================================
# Async Clients (for llm.py)
# =============================================================================


def get_async_client(provider: str | None = None) -> tuple[Any, str]:
    """Get an asynchronous LLM client.

    Args:
        provider: Explicit provider name, or None to auto-detect.

    Returns:
        Tuple of (client, provider_name) where:
        - client is anthropic.AsyncAnthropic or openai.AsyncOpenAI
        - provider_name is 'anthropic' or 'openrouter'

    Raises:
        LLMProviderError: If provider cannot be determined or SDK not installed.
    """
    if provider is None:
        provider = detect_provider()

    if provider == "anthropic":
        return _get_anthropic_async_client(), provider
    else:
        return _get_openrouter_async_client(), provider


def _get_anthropic_async_client():
    """Get an asynchronous Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get an API key at https://console.anthropic.com/"
        )

    try:
        import anthropic
    except ImportError:
        raise LLMProviderError(
            "anthropic package is required for Anthropic provider. "
            "Install with: uv add anthropic"
        )

    return anthropic.AsyncAnthropic(api_key=api_key)


def _get_openrouter_async_client():
    """Get an asynchronous OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Get an API key at https://openrouter.ai/keys"
        )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise LLMProviderError(
            "openai package is required for OpenRouter provider. "
            "Install with: uv add openai"
        )

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# =============================================================================
# Completion Helpers
# =============================================================================


def make_completion_sync(
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Make a synchronous completion request.

    Abstracts the API differences between Anthropic and OpenAI/OpenRouter.

    Args:
        client: LLM client from get_sync_client().
        provider: Provider name ('anthropic' or 'openrouter').
        model: Model name (will be resolved for provider).
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        The text content of the response.
    """
    resolved_model = resolve_model(model, provider)

    if provider == "anthropic":
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


async def make_completion_async(
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Make an asynchronous completion request.

    Abstracts the API differences between Anthropic and OpenAI/OpenRouter.

    Args:
        client: Async LLM client from get_async_client().
        provider: Provider name ('anthropic' or 'openrouter').
        model: Model name (will be resolved for provider).
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        json_mode: Request JSON output format (OpenRouter only, Anthropic ignores).

    Returns:
        The text content of the response.
    """
    resolved_model = resolve_model(model, provider)

    if provider == "anthropic":
        response = await client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.content[0].text
    else:
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
