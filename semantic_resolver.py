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
   - If it is redundant, contradictory, or meaningless (e.g., "Anti-Anti-Time", "Water-Fire-Hybrid"), mark it as RUBBISH.

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
   - If redundant, contradictory, or meaningless (e.g., "Anti-Anti-Time"), mark as DISCARD

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
            # Try to extract JSON from the response
            # Handle cases where the model wraps it in markdown code blocks
            text = response_text.strip()

            # Remove markdown code blocks if present
            if text.startswith("```"):
                # Find the JSON content between code blocks
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if match:
                    text = match.group(1)
                else:
                    # Try removing just the backticks
                    text = re.sub(r'^```(?:json)?\s*', '', text)
                    text = re.sub(r'\s*```$', '', text)

            data = json.loads(text)

            return ResolvedConcept(
                original=original,
                resolved_name=data.get("resolved_name", original),
                definition=data.get("definition", ""),
                status=data.get("status", "KEEP").upper(),
                confidence=float(data.get("confidence", 0.5))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, keep the original concept
            return ResolvedConcept(
                original=original,
                resolved_name=original,
                definition="Failed to parse LLM response",
                status="KEEP",
                confidence=0.0,
                error=str(e)
            )

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

            # Cache the result
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
        """Parse a batch LLM response into multiple ResolvedConcepts"""
        results = []

        try:
            text = response_text.strip()

            # Remove markdown code blocks if present
            if text.startswith("```"):
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
                if match:
                    text = match.group(1)
                else:
                    text = re.sub(r'^```(?:json)?\s*', '', text)
                    text = re.sub(r'\s*```$', '', text)

            data_list = json.loads(text)

            if not isinstance(data_list, list):
                raise ValueError("Expected JSON array")

            # Match results to original concepts by order
            for i, concept in enumerate(concepts):
                original = concept.get("name", "")

                if i < len(data_list):
                    data = data_list[i]
                    result = ResolvedConcept(
                        original=original,
                        resolved_name=data.get("resolved_name", original),
                        definition=data.get("definition", ""),
                        status=data.get("status", "KEEP").upper(),
                        confidence=float(data.get("confidence", 0.5))
                    )
                else:
                    # Not enough results returned
                    result = ResolvedConcept(
                        original=original,
                        resolved_name=original,
                        definition="Missing from batch response",
                        status="KEEP",
                        confidence=0.0,
                        error="Missing from batch"
                    )
                results.append(result)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If batch parsing fails, return all as unresolved
            for concept in concepts:
                original = concept.get("name", "")
                results.append(ResolvedConcept(
                    original=original,
                    resolved_name=original,
                    definition="Batch parse failed",
                    status="KEEP",
                    confidence=0.0,
                    error=str(e)
                ))

        return results

    def resolve_batch_llm(self, concepts: List[Dict[str, str]], batch_size: int = 5) -> List[ResolvedConcept]:
        """
        Resolve multiple concepts using batched LLM calls (more efficient).

        Args:
            concepts: List of dicts with 'name' and 'domain' keys
            batch_size: Number of concepts to resolve in a single LLM call

        Returns:
            List of ResolvedConcept results
        """
        all_results = []

        # Filter out already cached concepts
        uncached = []
        cached_results = {}

        for concept in concepts:
            name = concept.get("name", "")
            domain = concept.get("domain", "General")
            cache_key = f"{name}|{domain}"

            if cache_key in self._cache:
                cached_results[name] = self._cache[cache_key]
            else:
                uncached.append(concept)

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

                # Cache the results
                for j, result in enumerate(batch_results):
                    name = batch[j].get("name", "")
                    domain = batch[j].get("domain", "General")
                    cache_key = f"{name}|{domain}"
                    self._cache[cache_key] = result

            except Exception as e:
                # On error, create error results for this batch
                batch_results = [
                    ResolvedConcept(
                        original=c.get("name", ""),
                        resolved_name=c.get("name", ""),
                        definition="",
                        status="KEEP",
                        confidence=0.0,
                        error=str(e)
                    )
                    for c in batch
                ]

        # Rebuild results in original order
        for concept in concepts:
            name = concept.get("name", "")
            domain = concept.get("domain", "General")
            cache_key = f"{name}|{domain}"

            if cache_key in self._cache:
                all_results.append(self._cache[cache_key])
            elif name in cached_results:
                all_results.append(cached_results[name])
            else:
                # Shouldn't happen, but fallback
                all_results.append(ResolvedConcept(
                    original=name,
                    resolved_name=name,
                    definition="",
                    status="KEEP",
                    confidence=0.0,
                    error="Not found in results"
                ))

        return all_results

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
