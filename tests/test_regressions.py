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


DOMAINS = ["Social", "Tech", "Bio", "Physics", "Phil", "Abstract", "General"]


class OperatorTests(unittest.TestCase):

    def setUp(self):
        self.rules = dict(ce.RULES)
        self.cache = dict(ce.Antonyms._cache)

    def tearDown(self):
        ce.RULES.clear(); ce.RULES.update(self.rules)
        ce.Antonyms._cache.clear(); ce.Antonyms._cache.update(self.cache)

    def test_invert_swaps_properties_for_antonyms(self):
        ce.Antonyms._cache.update({"liberal": "conservative", "conservative": "liberal"})
        seed = ce.Concept("Democracy", "Social", {"liberal", "parliamentary"})
        anti = ce.op_invert(seed)
        self.assertEqual(anti.properties, {"conservative", "parliamentary", "inverted"})
        self.assertIn("liberal", ce.op_invert(anti).properties)

    def test_antonym_lookup_prefers_the_commonest_word(self):
        class Session:
            def get(self, url, params=None, timeout=None):
                return types.SimpleNamespace(json=lambda: [
                    {"word": "conservativist", "tags": ["f:0.0"]},
                    {"word": "conservative", "tags": ["f:24.9"]}])
        ce.AutoSeeder._session = Session()
        self.assertEqual(ce.Antonyms._fetch("liberal"), (True, "conservative"))

    def test_rotate_favours_domains_the_properties_point_at(self):
        random.seed(1)
        c = ce.Concept("Bitcoin", "Social", {"digital code", "network data"})
        hits = sum(ce.op_rotate(c, DOMAINS).domain == "Tech" for _ in range(600))
        # Uniform over 6 domains would give ~100; weight 1 + 3*2 = 7 of 12 gives ~350
        self.assertGreater(hits, 250)

    def test_rotate_replaces_the_domain_and_never_goes_home(self):
        c = ce.Concept("Viral", "Bio", {"x"})
        once = ce.op_rotate(c, DOMAINS)
        for _ in range(50):
            twice = ce.op_rotate(once, DOMAINS)
            self.assertEqual(twice.name, f"Viral-in-{twice.domain}")
            self.assertNotIn(twice.domain, ("Bio", once.domain))
        ce.RULES["rotate_replaces_domain"] = False
        self.assertTrue(ce.op_rotate(once, DOMAINS).name.startswith(f"{once.name}-in-"))

    def test_blend_needs_different_seeds_and_keeps_the_last_prefix(self):
        dem, btc = ce.Concept("Democracy", "Social", set()), ce.Concept("Bitcoin", "Tech", set())
        self.assertIsNone(ce.op_blend(ce.op_invert(dem), ce.op_lift(dem)))
        blend = ce.op_blend(ce.op_invert(ce.op_lift(dem)), ce.op_rotate(btc, DOMAINS))
        self.assertEqual(blend.name, "Anti-Democracy-Bitcoin Hybrid")
        self.assertEqual(blend.roots, {"Democracy", "Bitcoin"})

    def test_seed_names_with_prefix_words_survive_blending(self):
        warming, btc = ce.Concept("Global Warming", "Physics", set()), ce.Concept("Bitcoin", "Tech", set())
        self.assertEqual(ce.op_blend(warming, btc).name, "Global Warming-Bitcoin Hybrid")

    def test_skip_rules(self):
        seeds = [ce.Concept("Democracy", "Social", {"x"}), ce.Concept("Bitcoin", "Tech", {"y"})]
        random.seed(0)
        v = ce.VisualGEE(); v.run(seeds, DOMAINS, steps=3)
        self.assertTrue(any("Anti-Anti-" in n for n in v.concepts))   # kept by default
        ce.RULES.update(skip_double_invert=True, skip_repeat_scale=True, skip_repeat_recurse=True)
        random.seed(0)
        v = ce.VisualGEE(); v.run(seeds, DOMAINS, steps=3)
        for repeat in ("Anti-Anti-", "Global Global ", "Self-Self-"):
            self.assertFalse(any(repeat in n for n in v.concepts), repeat)

    def test_every_step_adds_nodes(self):
        sizes = []
        for steps in (1, 2, 3):
            random.seed(0)
            seeds = [ce.Concept(n, d, {"p"}) for n, d in
                     [("Democracy", "Social"), ("Photosynthesis", "Bio"), ("Entropy", "Physics"), ("Bitcoin", "Tech")]]
            v = ce.VisualGEE(); v.run(seeds, DOMAINS, steps=steps)
            sizes.append(len(v.concepts))
        self.assertLess(sizes[0], sizes[1])
        self.assertLess(sizes[1], sizes[2])

    def test_lineage_survives_save_and_load(self):
        c = ce.op_blend(ce.op_invert(ce.Concept("Democracy", "Social", set())), ce.Concept("Bitcoin", "Tech", set()))
        back = ce.Concept.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual((back.roots, back.base, back.unrotated), (c.roots, c.base, c.unrotated))
        old = ce.Concept.from_dict({"name": "Anti-X", "domain": "Bio", "properties": []})
        self.assertEqual(old.roots, {"Anti-X"})


if __name__ == "__main__":
    unittest.main()
