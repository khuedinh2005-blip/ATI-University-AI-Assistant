"""
Ingestion Service for University AI Assistant.
Handles:
- Fetching announcements from the university API (or mock portal).
- Parsing HTML and converting to clean structured text.
- Computing SHA-256 hash for change detection.
- Chunking documents into semantic passages.
- Attaching rich metadata (publication_date, expiration_date, source_url, etc.).
"""

import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup

from backend.mock_portal.portal_data import fetch_portal_announcements


def clean_html_content(html: str) -> str:
    """
    Parses HTML content using BeautifulSoup, formatting tables and paragraphs
    into clean plain text while preserving structure.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Format tables into readable text rows
    for table in soup.find_all("table"):
        table_lines = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
            if cells:
                table_lines.append(" | ".join(cells))
        table_str = "\n" + "\n".join(table_lines) + "\n"
        table.replace_with(table_str)

    # Convert list items
    for li in soup.find_all("li"):
        li.insert_before("- ")
        li.insert_after("\n")

    text = soup.get_text(separator="\n")
    # Clean multiple spaces and blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def compute_content_hash(text: str) -> str:
    """Computes a SHA-256 checksum of the normalized text content."""
    normalized = text.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    Splits text into chunks of roughly `chunk_size` characters with `overlap`
    attempting to split at line breaks or sentences.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Attempt to split at paragraph or newline
        newline_pos = text.rfind("\n", start, end)
        if newline_pos > start + (chunk_size // 2):
            split_pos = newline_pos
        else:
            # Fall back to sentence boundary
            sentence_pos = text.rfind(". ", start, end)
            if sentence_pos > start + (chunk_size // 2):
                split_pos = sentence_pos + 1
            else:
                # Fall back to space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    split_pos = space_pos
                else:
                    split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start by split_pos minus overlap
        start = max(start + 1, split_pos - overlap)

    return chunks


def process_announcement(raw_doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Processes a single raw announcement into chunk dictionaries with metadata.
    Returns (chunks, content_hash).
    """
    doc_id = raw_doc["id"]
    title = raw_doc.get("title", "")
    category = raw_doc.get("category", "General")
    pub_date = raw_doc.get("publication_date", "")
    exp_date = raw_doc.get("expiration_date", "") or ""
    source_url = raw_doc.get("source_url", "")
    raw_html = raw_doc.get("content_html", "")

    clean_body = clean_html_content(raw_html)
    full_text = f"Title: {title}\nCategory: {category}\nPublication Date: {pub_date}\n\n{clean_body}"
    content_hash = compute_content_hash(full_text)

    text_chunks = chunk_text(full_text, chunk_size=600, overlap=80)

    chunk_objects = []
    for idx, chunk_text_content in enumerate(text_chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        chunk_objects.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": chunk_text_content,
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "publication_date": pub_date,
                "expiration_date": exp_date,
                "source_url": source_url,
                "content_hash": content_hash,
                "chunk_index": idx,
                "total_chunks": len(text_chunks)
            }
        })

    return chunk_objects, content_hash
