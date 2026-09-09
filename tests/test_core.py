import unittest

from app.server import app
from scripts.crawler.crawler import parse_list
from src.generate.answer import build_sources, merge_retrieval_results
from src.embed.chunking import split_document
from src.retrieval.kb import _filter_doc_ids, _select_diverse_chunks, extract_metadata_filters
from src.security.compliance import check_content


class RetrievalTests(unittest.TestCase):
    def test_merge_keeps_multiple_chunks_from_same_document(self):
        primary = [
            {"chunk_id": "a-1", "doc_id": "a", "score": 0.9},
            {"chunk_id": "a-2", "doc_id": "a", "score": 0.8},
        ]
        secondary = [{"chunk_id": "a-1", "doc_id": "a", "score": 0.7}]
        merged = merge_retrieval_results(primary, secondary, 3)
        self.assertEqual([item["chunk_id"] for item in merged], ["a-1", "a-2"])

    def test_sources_are_unique_but_count_evidence_chunks(self):
        results = [
            {"doc_id": "a", "score": 0.8},
            {"doc_id": "a", "score": 0.9},
            {"doc_id": "b", "score": 0.7},
        ]
        sources = build_sources(results)
        self.assertEqual([item["doc_id"] for item in sources], ["a", "b"])
        self.assertEqual(sources[0]["evidence_chunks"], 2)
        self.assertEqual(sources[0]["score"], 0.9)

    def test_extracts_year_and_most_specific_region(self):
        self.assertEqual(
            extract_metadata_filters("深圳市南山区2023年产业情况"),
            {"year": "2023", "region": "南山区"},
        )

    def test_metadata_filter(self):
        index = {"doc_metadata": {
            "a": {"year": "2023", "region": "南山区"},
            "b": {"year": "2024", "region": "南山区"},
        }}
        self.assertEqual(_filter_doc_ids({"a", "b"}, index, {"year": "2023", "region": "南山区"}), {"a"})

    def test_short_paragraph_merging_respects_max_chars(self):
        text = "各位代表\n" + "\n".join("短段落" * 9 for _ in range(30))
        chunks = split_document("demo.txt", text, {}, max_chars=64, min_chars=40)
        self.assertTrue(chunks)
        self.assertLessEqual(max(len(item["text"]) for item in chunks), 64)

    def test_chunk_selection_limits_single_document(self):
        chunks = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]
        selected = _select_diverse_chunks([(0, .9), (1, .8), (2, .7), (3, .6)], chunks, 3)
        self.assertEqual([i for i, _ in selected], [0, 1, 3])


class CrawlerTests(unittest.TestCase):
    def test_parse_list_resolves_relative_url_and_skips_missing_year(self):
        html = '<a href="/zfgb/2025/report.html">2025年政府工作报告</a><a href="/zfgb/no-year">未知</a>'
        cfg = {"link_pattern": "/zfgb/"}
        items = parse_list(html, cfg, "https://example.gov/list")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.gov/zfgb/2025/report.html")


class SecurityAndApiTests(unittest.TestCase):
    def test_sensitive_word(self):
        self.assertIn("制毒", check_content("提供制毒方法"))

    def test_api_rejects_non_object_json(self):
        with app.test_client() as client:
            self.assertEqual(client.post("/api/ask", json=[]).status_code, 400)

    def test_api_rejects_non_string_question(self):
        with app.test_client() as client:
            self.assertEqual(client.post("/api/ask", json={"question": 123}).status_code, 400)

    def test_api_rejects_unknown_mode(self):
        with app.test_client() as client:
            response = client.post("/api/ask", json={"question": "测试", "mode": "unknown"})
            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
