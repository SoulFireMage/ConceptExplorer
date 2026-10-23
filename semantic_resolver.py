"""
Semantic Resolver - The "Cognitive Lens" Layer

Takes raw mechanically-generated concept names and resolves them to:
- Real-world equivalents (e.g., "Anti-Structure" -> "Entropy")
- Valid theoretical concepts (e.g., "Time-Loop-Recursive" -> "Closed Timelike Curve")
- Creative metaphors (e.g., "Social-Toxin-Network" -> "Viral Misinformation")
- Or marks them as DISCARD if they're nonsense

Uses caching to avoid repeated API calls for the same concept.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from functools import lru_cache

from llm_provider import LLMProvider


# The system prompt for single concept resolution
RESOLVER_SYSTEM_PROMPT = """
You are the "Semantic Collapser" for a concept discovery engine.
Your goal is to take a raw, mechanically generated concept name (created by combining words) and determine if it points to a real, theoretical, or novel idea, or if it is just linguistic garbage.

INPUT: A raw concept string (e.g., "Anti-Structure", "Global-Biology-Hybrid").
CONTEXT: The domain seeds involved (e.g., "Physics", "Sociology").

INSTRUCTIONS:
1. ANALYZE: Look at the raw string. Does it describe a known phenomenon, a sci-fi trope, or a valid philosophical construct?
2. RESOLVE:
   - If it matches a specific real-world term, return that term (e.g., "Anti-Structure" -> "Entropy").
   - If it is a valid but abstract compound, refine the name to sound standard (e.g., "Time-Loop-Recursive" -> "Closed Timelike Curve").
   - If it is a novel/creative metaphor that makes sense, give it a cool name (e.g., "Social-Toxin-Network" -> "Viral Misinformation").
   - If it is redundant, contradictory, or meaningless (e.g., "Global Global Global Chair", "Water-Fire-Hybrid"), mark it as DISCARD.
   - A repeated prefix is not rubbish by itself: keep it when the repetition means something (e.g., "Anti-Anti-Viral" -> "Antiviral Resistance").

OUTPUT FORMAT:
Return ONLY a valid JSON object with no markdown formatting:
{
  "status": "KEEP" or "DISCARD",
  "original": "The input string",
  "resolved_name": "The best short name for this concept",
  "definition": "A 1-sentence explanation of what this concept is.",
  "confidence": A score 0.0 to 1.0
}
"""

# The system prompt for BATCH resolution (multiple concepts at once)
BATCH_RESOLVER_SYSTEM_PROMPT = """
You are the "Semantic Collapser" for a concept discovery engine.
Your goal is to take raw, mechanically generated concept names and determine if they point to real, theoretical, or novel ideas, or if they are linguistic garbage.

INSTRUCTIONS:
1. ANALYZE each concept. Does it describe a known phenomenon, a sci-fi trope, or a valid philosophical construct?
2. RESOLVE each:
   - If it matches a real-world term, return that term (e.g., "Anti-Structure" -> "Entropy")
   - If it's a valid abstract compound, refine to standard terminology (e.g., "Time-Loop-Recursive" -> "Closed Timelike Curve")
   - If it's a creative metaphor that makes sense, give it a cool name (e.g., "Social-Toxin-Network" -> "Viral Misinformation")
   - If redundant, contradictory, or meaningless (e.g., "Global Global Global Chair"), mark as DISCARD
   - A repeated prefix is not rubbish by itself: keep it when the repetition means something (e.g., "Anti-Anti-Viral" -> "Antiviral Resistance")

OUTPUT FORMAT:
Return ONLY a valid JSON array with no markdown formatting. Each element must have:
[
  {
    "status": "KEEP" or "DISCARD",
    "original": "The input string exactly as given",
    "resolved_name": "The best short name",
    "definition": "A 1-sentence explanation",
    "confidence": 0.0 to 1.0
  },
  ...
]

IMPORTANT: Return results in the SAME ORDER as the input concepts.
"""


@dataclass
class ResolvedConcept:
    """Result of resolving a concept through the LLM"""
    original: str
    resolved_name: str
    definition: str
    status: str  # "KEEP" or "DISCARD"
    confidence: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "resolved_name": self.resolved_name,
            "definition": self.definition,
            "status": self.status,
            "confidence": self.confidence,
            "error": self.error
        }


def _strip_fences(text: str) -> str:
    """Remove a markdown code fence around a JSON reply, if there is one"""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r'^```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        text = match.group(1) if match else re.sub(r'^```(?:json)?\s*', '', text)
    return text


def _normalize_status(value) -> str:
    """Anything the model calls DISCARD or RUBBISH is a discard; the rest is kept"""
    return "DISCARD" if str(value or "").strip().upper() in ("DISCARD", "RUBBISH") else "KEEP"


def _to_result(original: str, data) -> ResolvedConcept:
    """Build a ResolvedConcept from one parsed JSON answer"""
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return ResolvedConcept(
        original=original,
        resolved_name=data.get("resolved_name") or original,
        definition=data.get("definition") or "",
        status=_normalize_status(data.get("status")),
        confidence=float(data.get("confidence", 0.5))
    )


def _error_result(original: str, error: str, definition: str = "") -> ResolvedConcept:
    """An unresolved concept: kept under its own name, with the reason"""
    return ResolvedConcept(original=original, resolved_name=original, definition=definition,
                           status="KEEP", confidence=0.0, error=error)


class SemanticResolver:
    """
    Resolves raw concept names to meaningful terms using an LLM.

    Features:
    - LRU cache to avoid repeated API calls
    - Batch processing support
    - Graceful error handling
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._cache: Dict[str, ResolvedConcept] = {}

    def _parse_response(self, response_text: str, original: str) -> ResolvedConcept:
        """Parse the LLM response into a ResolvedConcept"""
        try:
            return _to_result(original, json.loads(_strip_fences(response_text)))
        except (ValueError, TypeError) as e:
            # If parsing fails, keep the original concept
            return _error_result(original, str(e), "Failed to parse LLM response")

    def resolve_single(self, concept_name: str, domain: str = "General") -> ResolvedConcept:
        """
        Resolve a single concept name.

        Args:
            concept_name: The raw concept name to resolve
            domain: The domain context for resolution

        Returns:
            ResolvedConcept with the resolution result
        """
        # Check cache first
        cache_key = f"{concept_name}|{domain}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            user_prompt = f'Resolve this concept: "{concept_name}" in the domain of "{domain}"'

            response = self.provider.complete(
                system_prompt=RESOLVER_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            result = self._parse_response(response.content, concept_name)

            # Cache the result (not failures, so a retry asks again)
            if result.error is None:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            # On error, return the original concept as KEEP
            return ResolvedConcept(
                original=concept_name,
                resolved_name=concept_name,
                definition="",
                status="KEEP",
                confidence=0.0,
                error=str(e)
            )

    def _parse_batch_response(self, response_text: str, concepts: List[Dict[str, str]]) -> List[ResolvedConcept]:
        """
        Parse a batch LLM response into one ResolvedConcept per input concept.

        Answers are matched by their "original" field, falling back to position,
        so a reordered reply still lands on the right concept. Each answer is
        parsed on its own: one malformed entry doesn't sink the batch.
        """
        try:
            data_list = json.loads(_strip_fences(response_text))
            if not isinstance(data_list, list):
                raise ValueError("Expected JSON array")
        except (ValueError, TypeError) as e:
            # If batch parsing fails, return all as unresolved
            return [_error_result(c.get("name", ""), str(e), "Batch parse failed") for c in concepts]

        names = [c.get("name", "") for c in concepts]
        # Match by name only where the name is unique in this batch: the same
        # name in two domains has to go by position
        unique = {n for n in names if names.count(n) == 1}
        by_original = {
            item["original"]: item for item in data_list
            if isinstance(item, dict) and item.get("original") in unique
        }

        def names_someone_else(item, original):
            # An answer that names another input in this batch isn't ours,
            # even if it sits in our position
            return isinstance(item, dict) and item.get("original") in names and item.get("original") != original

        results = []
        for i, concept in enumerate(concepts):
            original = concept.get("name", "")
            if original in by_original:
                data = by_original[original]
            elif i < len(data_list) and not names_someone_else(data_list[i], original):
                data = data_list[i]
            else:
                results.append(_error_result(original, "Missing from batch", "Missing from batch response"))
                continue
            try:
                results.append(_to_result(original, data))
            except (ValueError, TypeError) as e:
                results.append(_error_result(original, str(e), "Failed to parse LLM response"))
        return results

    def resolve_batch_llm(self, concepts: List[Dict[str, str]], batch_size: int = 5) -> List[ResolvedConcept]:
        """
        Resolve multiple concepts using batched LLM calls (more efficient).

        Args:
            concepts: List of dicts with 'name' and 'domain' keys
            batch_size: Number of concepts to resolve in a single LLM call

        Returns:
            List of ResolvedConcept results, in the same order as `concepts`.
            Failures carry their error; they aren't cached, so a retry asks again.
        """
        def key(c):
            return f"{c.get('name', '')}|{c.get('domain', 'General')}"

        results: Dict[str, ResolvedConcept] = {}
        uncached = [c for c in concepts if key(c) not in self._cache]

        # Process uncached concepts in batches
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]

            # Build the prompt for this batch
            concept_list = "\n".join([
                f'{j+1}. "{c.get("name", "")}" (domain: {c.get("domain", "General")})'
                for j, c in enumerate(batch)
            ])

            user_prompt = f"Resolve these {len(batch)} concepts:\n{concept_list}"

            try:
                response = self.provider.complete(
                    system_prompt=BATCH_RESOLVER_SYSTEM_PROMPT,
                    user_prompt=user_prompt
                )
                batch_results = self._parse_batch_response(response.content, batch)
            except Exception as e:
                batch_results = [_error_result(c.get("name", ""), str(e)) for c in batch]

            for c, result in zip(batch, batch_results):
                results[key(c)] = result
                if result.error is None:
                    self._cache[key(c)] = result

        return [results.get(key(c)) or self._cache[key(c)] for c in concepts]

    def resolve_batch(self, concepts: List[Dict[str, str]], use_batching: bool = True, batch_size: int = 5) -> List[ResolvedConcept]:
        """
        Resolve multiple concepts.

        Args:
            concepts: List of dicts with 'name' and 'domain' keys
            use_batching: If True, send multiple concepts per LLM call (faster)
            batch_size: Number of concepts per batch when batching

        Returns:
            List of ResolvedConcept results
        """
        if use_batching and len(concepts) > 1:
            return self.resolve_batch_llm(concepts, batch_size=batch_size)

        # Fall back to single resolution
        results = []
        for concept in concepts:
            name = concept.get("name", "")
            domain = concept.get("domain", "General")
            result = self.resolve_single(name, domain)
            results.append(result)
        return results

    def clear_cache(self):
        """Clear the resolution cache"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_concepts": len(self._cache),
            "kept": sum(1 for r in self._cache.values() if r.status == "KEEP"),
            "discarded": sum(1 for r in self._cache.values() if r.status == "DISCARD")
        }
