"""
Comprehensive Test Suite for Hanoi University AI Assistant.
Validates:
1. Data Ingestion & HTML table parsing.
2. Content hashing (SHA-256) & incremental sync (skip unchanged).
3. ChromaDB vector indexing and retrieval.
4. Temporal-Aware logic (validity filtering & recency ranking).
5. Question Answering grounding on sample queries from the midterm report.
"""

import sys
import unittest
from pathlib import Path
from datetime import date

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.ingestion import (
    clean_html_content,
    compute_content_hash,
    chunk_text,
    process_announcement
)
from backend.services.vector_db import VectorDBService
from backend.services.temporal_ranker import rank_and_filter_chunks, parse_date
from backend.services.rag_engine import generate_answer
from backend.mock_portal.portal_data import fetch_portal_announcements


class TestUniversityAIAssistant(unittest.TestCase):

    def setUp(self):
        import uuid
        self.test_id = str(uuid.uuid4())[:8]
        self.test_dir = str(Path(__file__).resolve().parent / f"temp_test_chroma_{self.test_id}")
        self.vdb = VectorDBService(persist_dir=self.test_dir)

    def tearDown(self):
        import gc
        import shutil
        del self.vdb
        gc.collect()
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_html_parsing_and_hashing(self):
        """Test HTML table preservation and SHA-256 hashing."""
        sample_html = "<div><h2>Quy định</h2><table><tr><th>Môn</th><th>Tín chỉ</th></tr><tr><td>ATI</td><td>3</td></tr></table></div>"
        cleaned = clean_html_content(sample_html)
        self.assertIn("Môn | Tín chỉ", cleaned)
        self.assertIn("ATI | 3", cleaned)

        hash1 = compute_content_hash(cleaned)
        hash2 = compute_content_hash(cleaned)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex length

    def test_02_text_chunking(self):
        """Test that long documents are chunked with overlap."""
        long_text = "Câu văn số 1. " * 80
        chunks = chunk_text(long_text, chunk_size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) > 0 for c in chunks))

    def test_03_incremental_sync(self):
        """Test incremental synchronization: initial insert, then skip unchanged."""
        raw_docs = fetch_portal_announcements()
        
        # 1st Sync: All docs should be newly inserted
        stats1 = self.vdb.sync_knowledge_base(raw_docs)
        self.assertEqual(stats1["new"], len(raw_docs))
        self.assertEqual(stats1["skipped"], 0)
        self.assertGreater(self.vdb.collection.count(), 0)

        # 2nd Sync: With the exact same data, all docs should be skipped (content hash match)
        stats2 = self.vdb.sync_knowledge_base(raw_docs)
        self.assertEqual(stats2["new"], 0)
        self.assertEqual(stats2["skipped"], len(raw_docs))

    def test_04_temporal_validity_filtering(self):
        """Test that expired documents are filtered out as of October 2026."""
        # Sync knowledge base first
        self.vdb.sync_knowledge_base(fetch_portal_announcements())

        # Query about tuition fee
        chunks = self.vdb.query_chunks("Học phí học kỳ 1", n_results=10)
        self.assertGreater(len(chunks), 0)

        # Reference date: October 22, 2026
        ref_date = date(2026, 10, 22)
        selected, expired, meets_threshold = rank_and_filter_chunks(
            chunks=chunks,
            current_date=ref_date,
            top_n=5,
            threshold=0.55
        )

        self.assertTrue(meets_threshold)
        self.assertGreater(len(selected), 0)

        # Verify that expired 2025 announcement ('hanu-ann-2025-099') was filtered out!
        selected_ids = [c["metadata"]["doc_id"] for c in selected]
        self.assertNotIn("hanu-ann-2025-099", selected_ids)

        expired_ids = [c["metadata"]["doc_id"] for c in expired]
        self.assertIn("hanu-ann-2025-099", expired_ids)

    def test_05_relevance_threshold_fallback(self):
        """Test that completely irrelevant questions do not meet the similarity threshold."""
        self.vdb.sync_knowledge_base(fetch_portal_announcements())

        irrelevant_query = "Thời tiết ngày mai ở sao Hỏa và giá vàng bao nhiêu?"
        chunks = self.vdb.query_chunks(irrelevant_query, n_results=5)

        selected, expired, meets_threshold = rank_and_filter_chunks(
            chunks=chunks,
            current_date=date(2026, 10, 22),
            top_n=5,
            threshold=0.58
        )
        self.assertFalse(meets_threshold)

    def test_06_rag_answer_grounding(self):
        """Test answer generation with source citation extraction."""
        self.vdb.sync_knowledge_base(fetch_portal_announcements())

        query = "Điều kiện tốt nghiệp ngành Công nghệ thông tin"
        chunks = self.vdb.query_chunks(query, n_results=5)
        selected, _, _ = rank_and_filter_chunks(chunks, current_date=date(2026, 10, 22), top_n=3)

        result = generate_answer(query, selected)
        self.assertIn("answer", result)
        self.assertGreater(len(result["sources"]), 0)
        # Check source metadata
        top_source = result["sources"][0]
        self.assertIn("Quy định Chuẩn đầu ra", top_source["title"])


if __name__ == "__main__":
    unittest.main()
