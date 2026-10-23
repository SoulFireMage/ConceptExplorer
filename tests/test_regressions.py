"""
Regression tests for bugs found in review. No network: the LLM and the
Datamuse API are replaced with fakes.

Run from the repo root:  python -m unittest discover tests
"""

import json
import os
import random
import sys
import types
import unittest

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import concept_explorer as ce
import semantic_resolver as sr
from llm_provider import GeminiProvider, LLMResponse


class FakeProvider:
    def __init__(self, text=None, exc=None):
        self.text, self.exc = text, exc

    def complete(self, system_prompt, user_prompt):
        if self.exc:
            raise self.exc
        return LLMResponse(self.text, "fake")

    def is_configured(self):
        return True


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return types.SimpleNamespace(json=lambda: [])


TWO = [{"name": "A", "domain": "X"}, {"name": "B", "domain": "X"}]


class ResolverTests(unittest.TestCase):

    def test_rubbish_counts_as_discard(self):
        reply = '{"status": "RUBBISH", "resolved_name": "x", "confidence": 0.9}'
        r = sr.SemanticResolver(FakeProvider(reply)).resolve_single("Anti-Anti-Time")
        self.assertEqual(r.status, "DISCARD")

    def test_provider_error_reaches_the_result(self):
        res = sr.SemanticResolver(FakeProvider(exc=RuntimeError("quota exceeded"))).resolve_batch(TWO)
        self.assertEqual([r.error for r in res], ["quota exceeded"] * 2)

    def test_failures_are_not_cached(self):
        resolver = sr.SemanticResolver(FakeProvider(exc=RuntimeError("down")))
        resolver.resolve_batch(TWO)
        resolver.provider = FakeProvider(json.dumps([
            {"original": "A", "status": "KEEP", "resolved_name": "Ay"},
            {"original": "B", "status": "KEEP", "resolved_name": "Bee"}]))
        self.assertEqual([r.resolved_name for r in resolver.resolve_batch(TWO)], ["Ay", "Bee"])

    def test_one_malformed_item_spares_the_rest(self):
        reply = json.dumps([{"status": "KEEP", "resolved_name": "Good"}, "oops"])
        res = sr.SemanticResolver(FakeProvider(reply)).resolve_batch(TWO)
        self.assertEqual((res[0].resolved_name, res[0].error), ("Good", None))
        self.assertEqual(res[1].original, "B")
        self.assertIsNotNone(res[1].error)

    def test_bad_confidence_stays_on_its_own_concept(self):
        reply = json.dumps([{"status": "KEEP", "resolved_name": "Good", "confidence": 0.9},
                            {"status": "KEEP", "resolved_name": "Also", "confidence": "high"}])
        res = sr.SemanticResolver(FakeProvider(reply)).resolve_batch(TWO)
        self.assertEqual([r.original for r in res], ["A", "B"])
        self.assertIsNone(res[0].error)
        self.assertIn("high", res[1].error)

    def test_reordered_answers_match_by_name(self):
        reply = json.dumps([{"original": "B", "status": "KEEP", "resolved_name": "Bee"},
                            {"original": "A", "status": "KEEP", "resolved_name": "Ay"}])
        res = sr.SemanticResolver(FakeProvider(reply)).resolve_batch(TWO)
        self.assertEqual([(r.original, r.resolved_name) for r in res], [("A", "Ay"), ("B", "Bee")])

    def test_fenced_json_with_trailing_text(self):
        reply = '```json\n{"status": "KEEP", "resolved_name": "Entropy"}\n```\nHope that helps!'
        r = sr.SemanticResolver(FakeProvider(reply)).resolve_single("Anti-Structure")
        self.assertEqual(r.resolved_name, "Entropy")


class ProviderTests(unittest.TestCase):

    def test_gemini_key_not_in_url(self):
        seen = {}

        class Resp:
            def raise_for_status(self):
                raise requests.HTTPError(f"429 Client Error: Too Many Requests for url: {seen['url']}")

        def post(url, **kw):
            seen["url"], seen["headers"] = url, kw.get("headers", {})
            return Resp()

        g = GeminiProvider(api_key="AIzaSECRET123")
        g._session.post = post
        with self.assertRaises(requests.HTTPError) as ctx:
            g.complete("s", "u")
        self.assertNotIn("AIzaSECRET123", str(ctx.exception))
        self.assertEqual(seen["headers"].get("x-goog-api-key"), "AIzaSECRET123")


class ExplorerTests(unittest.TestCase):

    def setUp(self):
        self.session = FakeSession()
        ce.AutoSeeder._session = self.session
        ce.seeds.clear()
        self.client = ce.app.test_client()
        # Don't overwrite the sample graph in static/
        self._save_viz = ce.VisualGEE.save_viz
        ce.VisualGEE.save_viz = lambda viz, filename="": filename

    def tearDown(self):
        ce.VisualGEE.save_viz = self._save_viz

    def test_datamuse_name_is_encoded(self):
        ce.AutoSeeder.fetch_data("R&D")
        url, params = self.session.calls[0]
        self.assertEqual(params["rel_jjb"], "R&D")
        self.assertNotIn("R&D", url)

    def test_domain_guess_uses_whole_words(self):
        for word in ["excellent", "flawed", "sometimes", "settle"]:
            self.assertEqual(ce.AutoSeeder.guess_domain({word}), "General", word)
        self.assertEqual(ce.AutoSeeder.guess_domain({"genetic code"}), "Bio")
        self.assertEqual(ce.AutoSeeder.guess_domain({"robots"}), "Tech")

    def test_generate_never_rotates_into_auto_detect(self):
        random.seed(0)
        self.client.post("/add_seed", json={"name": "Democracy", "domain": "Social", "properties": "vote"})
        self.assertEqual(self.client.post("/generate", json={"steps": 3}).status_code, 200)
        domains = {c.domain for c in ce.last_generated_viz.concepts.values()}
        self.assertFalse(any("Auto-Detect" in d for d in domains), domains)

    def test_generate_accepts_numeric_strings_and_rejects_junk(self):
        self.client.post("/add_seed", json={"name": "Democracy", "domain": "Social", "properties": "vote"})
        self.assertEqual(self.client.post("/generate", json={"steps": "2"}).status_code, 200)
        self.assertEqual(self.client.post("/generate", json={"steps": "lots"}).status_code, 400)

    def test_debug_is_off_by_default(self):
        with open(os.path.join(ROOT, "concept_explorer.py")) as f:
            src = f.read()
        self.assertNotIn("debug=True", src)


if __name__ == "__main__":
    unittest.main()
