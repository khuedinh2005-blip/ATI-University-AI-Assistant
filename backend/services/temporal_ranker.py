"""
Temporal-Aware Retrieval & Re-ranking Service.
Implements:
1. Version Filtering: Discards older chunks only when a newer notice covers
   the same announcement topic.
2. Recency Re-ranking: Boosts chunks based on publication freshness.
3. Relevance Threshold Check: Fallback if similarity is too low.
"""

from datetime import datetime, date
from typing import List, Dict, Any, Tuple
import re
import unicodedata
from backend.config import settings


def parse_date(date_str: str) -> date:
    """Safely parses YYYY-MM-DD string into a date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def calculate_recency_boost(pub_date: date, ref_date: date) -> float:
    """
    Computes a recency boost score between 0.0 and 0.15 based on age in days.
    Newer announcements receive higher weight.
    """
    if not pub_date:
        return 0.0

    days_old = (ref_date - pub_date).days
    if days_old < 0:
        # Future date or today
        return 0.15
    elif days_old <= 30:
        return 0.15
    elif days_old <= 90:
        return 0.10
    elif days_old <= 180:
        return 0.05
    elif days_old <= 365:
        return 0.02
    else:
        return 0.0


def _content_key(chunk: Dict[str, Any]) -> str:
        """Build a stable topic key so newer editions supersede older editions."""
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        normalized = unicodedata.normalize("NFD", title)
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        ).lower()
        normalized = re.sub(r"\([^)]*(?:het han|expired|hết hạn)[^)]*\)", "", normalized)
        normalized = re.sub(r"\b20\d{2}(?:\s*-\s*20\d{2})?\b", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        return f"{metadata.get('category', 'General').lower()}:{normalized}"


def rank_and_filter_chunks(
    chunks: List[Dict[str, Any]],
    current_date: date = None,
    top_n: int = None,
    threshold: float = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Applies the 3-step Temporal-Aware ranking algorithm:
    1. Marks an old chunk as superseded only when a newer chunk covers the
       same announcement topic. A unique announcement remains usable even if
       its optional expiration metadata is in the past.
    2. Calculates recency boost and combines with semantic similarity:
       combined_score = (0.85 * semantic_similarity) + recency_boost
    3. Selects top_n chunks and verifies if highest similarity meets threshold.

    Returns:
    - selected_chunks: top_n valid, re-ranked chunks
    - expired_chunks: list of chunks superseded by a newer announcement
    - meets_threshold: boolean flag indicating if relevance threshold was reached
    """
    if current_date is None:
        current_date = date.today()
    if top_n is None:
        top_n = settings.TOP_N_RERANKED
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD

    valid_chunks = []
    expired_chunks = []

    # Phase 1: Supersede only older versions of the same topic. Expiration
    # metadata alone is not enough to discard a unique official notice.
    latest_by_key = {}
    for chunk in chunks:
        key = _content_key(chunk)
        publication_date = parse_date(chunk.get("metadata", {}).get("publication_date", ""))
        previous = latest_by_key.get(key)
        previous_date = parse_date(
            previous.get("metadata", {}).get("publication_date", "")
        ) if previous else None
        if previous is None or (publication_date and (not previous_date or publication_date > previous_date)):
            latest_by_key[key] = chunk

    for chunk in chunks:
        key = _content_key(chunk)
        latest = latest_by_key[key]
        publication_date = parse_date(
            chunk.get("metadata", {}).get("publication_date", "")
        )
        latest_date = parse_date(
            latest.get("metadata", {}).get("publication_date", "")
        )

        if latest is not chunk and publication_date and latest_date and publication_date < latest_date:
            expired_chunk = dict(chunk)
            expired_chunk["expired_reason"] = (
                f"Superseded by the newer announcement published on {latest_date}"
            )
            expired_chunks.append(expired_chunk)
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        return [], expired_chunks, False

    # Phase 2: Recency Re-ranking
    ranked_chunks = []
    for chunk in valid_chunks:
        metadata = chunk.get("metadata", {})
        pub_date = parse_date(metadata.get("publication_date", ""))
        semantic_sim = chunk.get("similarity", 0.0)

        boost = calculate_recency_boost(pub_date, current_date)
        # Combined score formula: 85% semantic + 15% recency max
        combined_score = round((0.85 * semantic_sim) + boost, 4)

        ranked_chunk = dict(chunk)
        ranked_chunk["recency_boost"] = boost
        ranked_chunk["combined_score"] = combined_score
        ranked_chunks.append(ranked_chunk)

    # Sort descending by combined_score
    ranked_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
    selected_chunks = ranked_chunks[:top_n]

    # Phase 3: Relevance Threshold Checking
    # Check if the highest semantic similarity reaches threshold
    highest_sim = max([c.get("similarity", 0.0) for c in selected_chunks]) if selected_chunks else 0.0
    meets_threshold = highest_sim >= threshold

    return selected_chunks, expired_chunks, meets_threshold
