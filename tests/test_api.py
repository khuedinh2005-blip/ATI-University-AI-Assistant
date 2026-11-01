"""
API Integration Tests for FastAPI backend endpoints:
- GET /health
- GET /status
- POST /sync (with and without admin key)
- POST /chat (question answering)
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient
from backend.main import app
from backend.config import settings

client = TestClient(app)


class TestAPIEndpoints(unittest.TestCase):

    def test_01_health_check(self):
        """Verify GET /health responds with 200 and 'ok' status."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")

    def test_02_status_endpoint(self):
        """Verify GET /status returns knowledge base information."""
        response = client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("knowledge_base", data)

    def test_03_sync_authentication(self):
        """Verify POST /sync rejects unauthorized calls and accepts valid admin keys."""
        # Unauthorized call without admin key
        unauth_resp = client.post("/sync")
        self.assertEqual(unauth_resp.status_code, 401)

        # Authorized call with valid admin key in header
        auth_resp = client.post("/sync", headers={"X-Admin-Key": settings.ADMIN_API_KEY})
        self.assertEqual(auth_resp.status_code, 200)
        data = auth_resp.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("summary", data)

    def test_04_chat_endpoint(self):
        """Verify POST /chat accepts student questions and returns grounded answers."""
        payload = {
            "message": "Học phí Học kỳ 1 năm học 2026-2027 là bao nhiêu?",
            "current_date": "2026-10-22"
        }
        response = client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertTrue(data.get("meets_threshold"))
        self.assertGreater(len(data.get("sources", [])), 0)

    def test_05_upload_pdf_authentication(self):
        """Verify POST /upload-pdf rejects unauthorized calls."""
        files = {"file": ("unauthorized.pdf", b"pdf content", "application/pdf")}
        unauth_resp = client.post("/upload-pdf", files=files)
        self.assertEqual(unauth_resp.status_code, 401)

    def test_06_upload_pdf_processing(self):
        """Verify POST /upload-pdf accepts and ingests valid PDF with admin key."""
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 72), "Trường Đại học Hà Nội\nQuy định điều kiện tốt nghiệp năm 2026.")
            pdf_bytes = doc.tobytes()
            doc.close()

            files = {"file": ("thong_bao_tot_nghiep_test.pdf", pdf_bytes, "application/pdf")}
            form_data = {"title": "Quy định tốt nghiệp 2026", "category": "Graduation"}
            headers = {"X-Admin-Key": settings.ADMIN_API_KEY}

            response = client.post("/upload-pdf", files=files, data=form_data, headers=headers)
            self.assertEqual(response.status_code, 200)
            res_data = response.json()
            self.assertEqual(res_data.get("status"), "success")
            self.assertIn("summary", res_data)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
