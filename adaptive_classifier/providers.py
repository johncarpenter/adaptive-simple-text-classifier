"""Pluggable LLM providers for fallback classification.

Provides a Protocol-based abstraction so any LLM backend can be used.
Includes implementations for Anthropic (direct), Vertex AI, Bedrock,
and a generic callable wrapper.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# --- Prompt templates ---

SYSTEM_PROMPT = """You are a classification engine. You classify items into categories from a provided taxonomy.

TAXONOMY:
{taxonomy}

RULES:
1. Each item MUST be classified into exactly one leaf category from the taxonomy.
2. Return the FULL PATH using " > " as separator (e.g., "Food > Burgers > Cheeseburger").
3. If an item is ambiguous, pick the BEST match. Never return "Unknown" or categories outside the taxonomy.
4. Return ONLY valid JSON. No markdown, no explanation."""

USER_PROMPT = """Classify these items. Return a JSON array of objects with "input" and "category" keys.

Items:
{items}

Return format:
[{{"input": "item text", "category": "Full > Path > Here"}}]"""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM classification providers."""

    def classify_batch(
        self,
        items: list[str],
        taxonomy_prompt: str,
        batch_size: int = 50,
    ) -> list[dict[str, str]]:
        """Classify a batch of items against a taxonomy.

        Args:
            items: List of text items to classify.
            taxonomy_prompt: Rendered taxonomy string for the system prompt.
            batch_size: Max items per LLM call.

        Returns:
            List of dicts with "input" and "category" keys.
        """
        ...


def _parse_llm_response(text: str) -> list[dict[str, str]]:
    """Parse LLM JSON response, handling common formatting issues."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "classifications" in result:
            return result["classifications"]
        return [result]
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse LLM response as JSON: {text[:200]}")


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class AnthropicProvider:
    """Direct Anthropic API provider.

    Uses prompt caching on the system prompt (taxonomy) for efficiency
    on repeated batch calls.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 4096,
        **client_kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required: pip install adaptive-classifier[anthropic]")

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, **client_kwargs)

    def classify_batch(
        self,
        items: list[str],
        taxonomy_prompt: str,
        batch_size: int = 50,
    ) -> list[dict[str, str]]:
        results = []
        chunks = _chunk(items, batch_size)

        for i, chunk in enumerate(chunks):
            logger.info(f"LLM batch {i+1}/{len(chunks)} ({len(chunk)} items)")

            items_text = "\n".join(f"- {item}" for item in chunk)
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(taxonomy=taxonomy_prompt),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": USER_PROMPT.format(items=items_text),
                }],
            )

            text = response.content[0].text
            parsed = _parse_llm_response(text)
            results.extend(parsed)

        return results


class VertexProvider:
    """Google Cloud Vertex AI provider (Claude via Vertex)."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        project_id: str | None = None,
        region: str = "us-east5",
        **client_kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required: pip install adaptive-classifier[vertex]")

        self.model = model
        self._client = anthropic.AnthropicVertex(
            project_id=project_id,
            region=region,
            **client_kwargs,
        )
        self.max_tokens = 4096

    def classify_batch(
        self,
        items: list[str],
        taxonomy_prompt: str,
        batch_size: int = 50,
    ) -> list[dict[str, str]]:
        results = []
        chunks = _chunk(items, batch_size)

        for i, chunk in enumerate(chunks):
            logger.info(f"Vertex batch {i+1}/{len(chunks)} ({len(chunk)} items)")

            items_text = "\n".join(f"- {item}" for item in chunk)
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT.format(taxonomy=taxonomy_prompt),
                messages=[{
                    "role": "user",
                    "content": USER_PROMPT.format(items=items_text),
                }],
            )

            text = response.content[0].text
            results.extend(_parse_llm_response(text))

        return results


class BedrockProvider:
    """AWS Bedrock provider (Claude via Bedrock)."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name: str = "us-east-1",
        **client_kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required: pip install adaptive-classifier[bedrock]")

        self.model = model
        self._client = anthropic.AnthropicBedrock(
            aws_region=region_name,
            **client_kwargs,
        )
        self.max_tokens = 4096

    def classify_batch(
        self,
        items: list[str],
        taxonomy_prompt: str,
        batch_size: int = 50,
    ) -> list[dict[str, str]]:
        results = []
        chunks = _chunk(items, batch_size)

        for i, chunk in enumerate(chunks):
            logger.info(f"Bedrock batch {i+1}/{len(chunks)} ({len(chunk)} items)")

            items_text = "\n".join(f"- {item}" for item in chunk)
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT.format(taxonomy=taxonomy_prompt),
                messages=[{
                    "role": "user",
                    "content": USER_PROMPT.format(items=items_text),
                }],
            )

            text = response.content[0].text
            results.extend(_parse_llm_response(text))

        return results


class CallableLLMProvider:
    """Wrap any function as an LLM provider.

    The callable receives (items: list[str], system_prompt: str, user_prompt: str)
    and returns raw text (JSON string).

    Usage:
        def my_llm(items, system_prompt, user_prompt):
            # Call your LLM here
            return '[{"input": "x", "category": "Y > Z"}]'

        provider = CallableLLMProvider(fn=my_llm)
    """

    def __init__(self, fn: Callable[..., str], batch_size: int = 50):
        self._fn = fn
        self._batch_size = batch_size

    def classify_batch(
        self,
        items: list[str],
        taxonomy_prompt: str,
        batch_size: int = 50,
    ) -> list[dict[str, str]]:
        results = []
        actual_batch_size = batch_size or self._batch_size
        chunks = _chunk(items, actual_batch_size)

        for chunk in chunks:
            items_text = "\n".join(f"- {item}" for item in chunk)
            system = SYSTEM_PROMPT.format(taxonomy=taxonomy_prompt)
            user = USER_PROMPT.format(items=items_text)

            text = self._fn(chunk, system, user)
            results.extend(_parse_llm_response(text))

        return results


def resolve_provider(
    provider: str | LLMProvider | Callable | None,
    **kwargs,
) -> LLMProvider:
    """Resolve a provider specification to an LLMProvider instance.

    Args:
        provider: One of:
            - "anthropic" / "claude": AnthropicProvider
            - "vertex": VertexProvider
            - "bedrock": BedrockProvider
            - An LLMProvider instance: used directly
            - A callable: wrapped in CallableLLMProvider
            - None: AnthropicProvider with defaults
        **kwargs: Passed to the provider constructor.
    """
    if provider is None:
        return AnthropicProvider(**kwargs)

    if isinstance(provider, str):
        name = provider.lower().strip()
        if name in ("anthropic", "claude"):
            return AnthropicProvider(**kwargs)
        elif name == "vertex":
            return VertexProvider(**kwargs)
        elif name == "bedrock":
            return BedrockProvider(**kwargs)
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                "Use 'anthropic', 'vertex', 'bedrock', or pass an LLMProvider instance."
            )

    if isinstance(provider, LLMProvider):
        return provider

    if callable(provider):
        return CallableLLMProvider(fn=provider, **kwargs)

    raise TypeError(f"Cannot resolve provider of type {type(provider)}")
