"""
PDF Parser Service for Hanoi University AI Assistant.
Extracts structured text from PDF documents page-by-page using PyMuPDF (fitz),
converts the extracted content into an HTML-like representation consistent with
the university ingestion pipeline, and formats document records for vector indexing.
"""

import hashlib
import logging
import re
from datetime import date
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Attempt importing PyMuPDF gracefully
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) is not installed. PDF parsing will not be functional until installed.")


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
}


def generate_pdf_doc_id(identifier: str) -> str:
    """
    Generates deterministic document ID:
    'hanu-pdf-' + sha256(url or title)[:12]
    """
    clean_id = identifier.strip().lower()
    hash_prefix = hashlib.sha256(clean_id.encode("utf-8")).hexdigest()[:12]
    return f"hanu-pdf-{hash_prefix}"


def detect_pdf_category(title: str, text: str = "") -> str:
    """Classifies category based on document title and text."""
    combined = f"{title} {text}".lower()
    if any(k in combined for k in ["học phí", "hoc phi", "lệ phí", "miễn giảm", "kinh phí", "tiền học"]):
        return "Tuition"
    if any(k in combined for k in ["tốt nghiệp", "xét tốt nghiệp", "chuẩn đầu ra", "bằng tốt nghiệp", "đồ án", "khóa luận"]):
        return "Graduation"
    if any(k in combined for k in ["quy chế", "quy định", "quy trình", "hướng dẫn", "nội quy", "điều lệ"]):
        return "Regulations"
    if any(k in combined for k in ["chương trình", "đào tạo", "môn học", "học phần", "thời khóa biểu", "lịch học", "lịch thi", "hoãn thi", "thi lại"]):
        return "Curriculum"
    if any(k in combined for k in ["học bổng", "khen thưởng", "tài trợ"]):
        return "Scholarship"
    if any(k in combined for k in ["tuyển sinh", "xét tuyển", "nhập học"]):
        return "Admissions"
    if any(k in combined for k in ["cơ sở vật chất", "phòng máy", "thư viện", "ký túc xá"]):
        return "Facility"
    return "Announcement"


def parse_pdf_from_bytes(
    pdf_bytes: bytes,
    metadata: Optional[Dict[str, Any]] = None,
    per_page: bool = False
) -> List[Dict[str, Any]]:
    """
    Parses PDF bytes using PyMuPDF (fitz), extracts text page by page,
    formats it into HTML-like structure, and returns a list of document
    dictionaries with fields:
    id, title, category, publication_date, expiration_date, source_url, content_html

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        metadata: Optional dictionary with user-supplied title, category, dates, source_url.
        per_page: If True, returns one dict per page; if False (default), returns
                  a unified document containing all pages.

    Returns:
        List of standardized document dictionaries.
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        logger.error("PyMuPDF (fitz) is not installed. Unable to parse PDF bytes.")
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing but is not installed.")

    if not pdf_bytes:
        logger.warning("Empty PDF bytes provided to parse_pdf_from_bytes.")
        return []

    meta = dict(metadata or {})

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF stream with fitz: {e}")
        return []

    num_pages = len(doc)
    if num_pages == 0:
        logger.warning("PDF document contains 0 pages.")
        doc.close()
        return []

    # Determine document title
    doc_title = meta.get("title")
    if not doc_title and doc.metadata and doc.metadata.get("title"):
        doc_title = doc.metadata.get("title").strip()

    # Extract text from pages
    page_data: List[Dict[str, Any]] = []
    first_page_text = ""

    for page_idx in range(num_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        text = page.get_text("text").strip()

        if page_idx == 0:
            first_page_text = text

        if not text:
            continue

        # Convert double-newline blocks to HTML paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        p_html = "\n".join(f"<p>{p.replace('\n', ' ')}</p>" for p in paragraphs)

        page_html = (
            f'<div class="pdf-page" data-page="{page_num}">\n'
            f'<h3>Trang {page_num}</h3>\n'
            f'{p_html}\n'
            f'</div>'
        )

        page_data.append({
            "page_num": page_num,
            "text": text,
            "html": page_html
        })

    doc.close()

    if not page_data:
        logger.warning("No readable text found across any pages of the PDF.")
        return []

    # If title still not set, infer from first line of text
    if not doc_title:
        lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
        if lines:
            doc_title = lines[0][:120]
        else:
            doc_title = "Tài liệu PDF Trường Đại học Hà Nội"

    source_url = meta.get("source_url", "")
    id_seed = source_url if source_url else doc_title
    base_id = generate_pdf_doc_id(id_seed)

    pub_date = meta.get("publication_date") or date.today().isoformat()
    exp_date = meta.get("expiration_date", "") or ""
    category = meta.get("category") or detect_pdf_category(doc_title, first_page_text)

    if per_page and len(page_data) > 1:
        # Return individual page records
        results = []
        for p in page_data:
            p_num = p["page_num"]
            results.append({
                "id": f"{base_id}-p{p_num}",
                "title": f"{doc_title} (Trang {p_num})",
                "category": category,
                "publication_date": pub_date,
                "expiration_date": exp_date,
                "source_url": source_url or f"pdf://{base_id}#page={p_num}",
                "content_html": p["html"]
            })
        return results

    # Unified document containing all structured pages
    combined_body = "\n".join(p["html"] for p in page_data)
    combined_html = (
        f'<div class="pdf-document">\n'
        f'<h2>{doc_title}</h2>\n'
        f'<p><strong>Nguồn tài liệu:</strong> {source_url or "Tập tin PDF tải lên"}</p>\n'
        f'<p><strong>Tổng số trang:</strong> {num_pages}</p>\n'
        f'{combined_body}\n'
        f'</div>'
    )

    return [{
        "id": base_id,
        "title": doc_title,
        "category": category,
        "publication_date": pub_date,
        "expiration_date": exp_date,
        "source_url": source_url or f"pdf://{base_id}",
        "content_html": combined_html
    }]


def parse_pdf_from_url(
    url: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Downloads a PDF document from a web URL and parses it using PyMuPDF.

    Args:
        url: Direct web URL to the PDF file.
        metadata: Optional dictionary with user-supplied metadata fields.

    Returns:
        List of standardized document dictionaries.
    """
    meta = dict(metadata or {})
    if "source_url" not in meta:
        meta["source_url"] = url

    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=15,
            allow_redirects=True
        )
        response.raise_for_status()
        pdf_bytes = response.content
        return parse_pdf_from_bytes(pdf_bytes=pdf_bytes, metadata=meta)
    except requests.RequestException as e:
        logger.warning(f"Network error while downloading PDF from {url}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Error parsing PDF from {url}: {e}")
        return []
