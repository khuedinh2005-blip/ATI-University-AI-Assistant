"""
Streamlit Web Application for ATI University AI Assistant.
Course: Advanced Topics in Information Technology (ATI) - Fall 2026.

Redesigned modern UI matching official university assistant mockup:
- Left Sidebar: University shield logo, "TRY ASKING" clickable prompt rows,
  Multi-session history, Navigation modes, Live sync status pill, and privacy disclaimer.
- Top Bar: "Ask about university information" heading & decorative "Official sources only" chip.
- Center Canvas: Soft lavender background, right-aligned indigo user bubbles (#4F46E5),
  left-aligned AI cards with blue "AI" circular avatar, and 2-column SOURCES cards grid.
- Bottom Bar: Full-width chat input with search placeholder and disclaimer footer note.
- Admin Panel: Automated portal sync + New interactive PDF uploader & chunk indexing.
- Temporal RAG Lab: Step-by-step pipeline inspection with recency boosting & validity filtering.
"""

import sys
import uuid
import html
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from backend.config import settings
from backend.services.vector_db import vector_db_service
from backend.services.temporal_ranker import (
    rank_and_filter_chunks,
    parse_date,
    calculate_recency_boost
)
from backend.services.rag_engine import generate_answer
from backend.mock_portal.portal_data import fetch_portal_announcements
from backend.services.ingestion import chunk_text, compute_content_hash

# Try importing PDF parser from backend services with graceful fallback
try:
    from backend.services.pdf_parser import parse_pdf_from_bytes, generate_pdf_doc_id
    HAS_PDF_PARSER = True
except ImportError:
    HAS_PDF_PARSER = False


# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="University AI Assistant - Hanoi University · FIT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================================
# HIGH-END MODERN CSS (INDIGO THEME & MOCKUP SPECIFICATION)
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }

    /* Constrain main conversation to centered readable column */
    [data-testid="stMainBlockContainer"] {
        max-width: 880px !important;
        padding-top: 4rem !important;
        padding-bottom: 6.5rem !important;
        margin: 0 auto !important;
    }
    #stMain [data-testid="stWidgetLabel"],
    #stMain [data-testid="stWidgetLabel"] p,
    #stMain label {
        color: #374151 !important;
    }
    #stMain input,
    #stMain textarea,
    #stMain [data-baseweb="select"],
    #stMain [data-baseweb="input"],
    #stMain [data-testid="stDateInputField"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border-color: #D1D5DB !important;
    }
    #stMain input::placeholder,
    #stMain textarea::placeholder {
        color: #9CA3AF !important;
    }
    #stMain [data-testid="stTextInput"] button {
        background-color: #F9FAFB !important;
        border-color: #D1D5DB !important;
        color: #4B5563 !important;
    }
    #stMain [data-testid="stAlert"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
    }
    #stMain [data-testid^="stAlertContent"],
    #stMain [data-testid^="stAlertContent"] * {
        color: #374151 !important;
    }

    /* Clean Streamlit Header & Toolbar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar Background & Border */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E5E7EB !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar Brand Header */
    .brand-header-wrapper {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .brand-shield-logo {
        width: 44px;
        height: 44px;
        min-width: 44px;
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
    }
    .brand-title-box {
        display: flex;
        flex-direction: column;
    }
    .brand-title-main {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.25;
        letter-spacing: -0.02em;
    }
    .brand-title-sub {
        font-size: 0.8rem;
        font-weight: 500;
        color: #6B7280;
        line-height: 1.3;
    }

    /* Sidebar Section Labels */
    .sidebar-meta-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #6B7280;
        text-transform: uppercase;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* TRY ASKING prompt items: Clickable rows (not chunky buttons) */
    .prompt-row-wrapper button {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        color: #1F2937 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 9px 12px !important;
        font-size: 0.82rem !important;
        line-height: 1.38 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.15s ease-in-out !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 44px !important;
        margin-bottom: 6px !important;
        width: 100% !important;
    }
    .prompt-row-wrapper button:hover {
        background-color: #EEF2FF !important;
        border-color: #C7D2FE !important;
        color: #4F46E5 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.08) !important;
    }
    .prompt-row-wrapper button p {
        text-align: left !important;
        font-size: 0.82rem !important;
        line-height: 1.35 !important;
        margin: 0 !important;
        color: inherit !important;
    }

    /* Primary & Action Buttons */
    div[data-testid="stSidebar"] button[kind="primary"],
    .main button[kind="primary"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 1px 3px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover,
    .main button[kind="primary"]:hover {
        background-color: #4338CA !important;
        border-color: #4338CA !important;
        box-shadow: 0 3px 8px rgba(79, 70, 229, 0.35) !important;
    }

    /* Sidebar Footer Info */
    .sidebar-footer-box {
        margin-top: 24px;
        padding-top: 14px;
        border-top: 1px solid #E5E7EB;
    }
    .sync-status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .sync-live-dot {
        width: 8px;
        height: 8px;
        min-width: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    .sync-status-text {
        font-size: 0.78rem;
        font-weight: 600;
        color: #374151;
    }
    .privacy-disclaimer-text {
        font-size: 0.72rem;
        color: #6B7280;
        line-height: 1.45;
    }

    /* Top Bar Header */
    .top-bar-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0 16px 0;
        border-bottom: 1px solid #F3F4F6;
        margin-bottom: 22px;
    }
    .top-bar-title-text {
        font-size: 0.98rem;
        font-weight: 500;
        color: #6B7280;
        letter-spacing: -0.01em;
    }
    .official-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4F46E5;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: 0 1px 2px rgba(79, 70, 229, 0.05);
    }

    /* Welcome / Fresh Chat Canvas */
    .canvas-welcome-banner {
        text-align: center;
        padding: 24px 16px 12px 16px;
        margin-bottom: 18px;
    }
    .canvas-welcome-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .canvas-welcome-sub {
        font-size: 0.92rem;
        color: #6B7280;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Canvas Starter Prompt Cards */
    .canvas-starter-grid button {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #1F2937 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.15s ease-in-out !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 60px !important;
        width: 100% !important;
    }
    .canvas-starter-grid button:hover {
        border-color: #C7D2FE !important;
        background-color: #F8FAFC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.08) !important;
        color: #4F46E5 !important;
    }
    .canvas-starter-grid button p {
        text-align: left !important;
        margin: 0 !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }

    /* User Message: Indigo bubble (#4F46E5), white text, aligned RIGHT */
    .user-bubble-row {
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin-top: 8px;
        margin-bottom: 20px;
    }
    .user-bubble-box {
        background-color: #4F46E5;
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 82%;
        font-size: 0.93rem;
        line-height: 1.55;
        box-shadow: 0 2px 5px rgba(79, 70, 229, 0.16);
        word-break: break-word;
    }

    /* AI Message: Left aligned, avatar "AI" (blue circle), white card with border #E5E7EB */
    .ai-msg-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        width: 100%;
        margin-top: 6px;
        margin-bottom: 24px;
    }
    .ai-avatar-circle {
        width: 36px;
        height: 36px;
        min-width: 36px;
        background-color: #4F46E5;
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
        margin-top: 2px;
    }
    .ai-card-wrapper {
        flex: 1;
        min-width: 0;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        color: #111827;
        font-size: 0.93rem;
        line-height: 1.62;
    }

    /* Sources Section & 2-Column Grid */
    .sources-container {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid #F3F4F6;
    }
    .sources-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #6B7280;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .sources-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    @media (max-width: 680px) {
        .sources-grid {
            grid-template-columns: 1fr;
        }
    }
    .source-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.15s ease-in-out;
    }
    .source-card:hover {
        border-color: #C7D2FE;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.08);
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
    }
    .source-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.35;
        flex: 1;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .source-card-meta {
        font-size: 0.75rem;
        color: #6B7280;
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .source-card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
    }
    .badge-valid {
        background-color: #ECFDF5;
        color: #10B981;
        border: 1px solid #A7F3D0;
        padding: 2px 7px;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .badge-expired {
        background-color: #FEF2F2;
        color: #EF4444;
        border: 1px solid #FECACA;
        padding: 2px 7px;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .source-card-link {
        font-size: 0.78rem;
        font-weight: 600;
        color: #4F46E5;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
    }
    .source-card-link:hover {
        color: #3730A3;
        text-decoration: underline;
    }

    /* Telemetry Info Pill */
    .telemetry-pill {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 10px;
    }

    /* Bottom Input & Footer */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    div[data-testid="stChatInput"] button {
        color: #4F46E5 !important;
    }
    .chat-footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    /* Keep Streamlit's app chrome and fixed chat composer in the same light theme. */
    :root {
        color-scheme: light;
        --primary-color: #4F46E5;
        --background-color: #F5F4FA;
        --secondary-background-color: #FFFFFF;
        --text-color: #1F2937;
    }
    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppScrollToBottomContainer"],
    .stApp,
    .main {
        color-scheme: light !important;
        color: #1F2937 !important;
        background-color: #F5F4FA !important;
    }
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(245, 244, 250, 0.92) !important;
    }

    /* A white navigation rail against the softly tinted conversation canvas. */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        border-right-color: #E8E6F0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #374151;
    }

    /* Override Streamlit theme defaults so primary and secondary actions match. */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] > button {
        background: #4F46E5 !important;
        border: 1px solid #4F46E5 !important;
        color: #FFFFFF !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"] > button:hover {
        background: #4338CA !important;
        border-color: #4338CA !important;
    }
    button[kind="primary"] p,
    button[data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] > button p {
        color: #FFFFFF !important;
    }
    #stMain button[kind="secondary"],
    #stMain button[data-testid="stBaseButton-secondary"],
    #stMain [data-testid="stBaseButton-secondary"] > button {
        background: #FFFFFF !important;
        border: 1px solid #E5E3ED !important;
        color: #1F2937 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(31, 41, 55, 0.04) !important;
    }
    #stMain button[kind="secondary"] p,
    #stMain button[data-testid="stBaseButton-secondary"] p,
    #stMain [data-testid="stBaseButton-secondary"] > button p {
        color: #1F2937 !important;
    }
    #stMain button[kind="secondary"]:hover,
    #stMain button[data-testid="stBaseButton-secondary"]:hover,
    #stMain [data-testid="stBaseButton-secondary"] > button:hover {
        background: #FFFFFF !important;
        border-color: #A5B4FC !important;
        color: #4338CA !important;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.08) !important;
    }
    #stMain button[kind="secondary"]:hover p,
    #stMain button[data-testid="stBaseButton-secondary"]:hover p {
        color: #4338CA !important;
    }
    #stMain button[data-testid="stBaseButton-secondary"] {
        height: auto !important;
        min-height: 60px !important;
        white-space: normal !important;
    }
    #stMain button[data-testid="stBaseButton-secondary"] .stMarkdownEllipsis,
    #stMain button[data-testid="stBaseButton-secondary"] .stMarkdownEllipsis p {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: pre-line !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] > button {
        background: #FFFFFF !important;
        border-color: #E5E7EB !important;
        color: #374151 !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] > button:hover {
        background: #F5F4FA !important;
        border-color: #C7D2FE !important;
        color: #4338CA !important;
    }

    /* The fixed composer was inheriting a dark bar from Streamlit's active theme. */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: linear-gradient(to bottom, rgba(245, 244, 250, 0), #F5F4FA 24px) !important;
    }
    div[data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 14px !important;
        box-shadow: 0 5px 18px rgba(31, 41, 55, 0.08) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #A5B4FC !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1), 0 5px 18px rgba(31, 41, 55, 0.08) !important;
    }
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] [data-baseweb="textarea"] {
        background: #FFFFFF !important;
        color: #1F2937 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important;
    }
    div[data-testid="stChatInput"] button {
        background: #4F46E5 !important;
        border: 0 !important;
        border-radius: 9px !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background: #4338CA !important;
    }

    .top-bar-container {
        border-bottom-color: #E8E6F0;
    }
    .ai-card-wrapper {
        border-color: #EAE8F1;
        border-radius: 14px;
        line-height: 1.7;
    }
    .user-bubble-box {
        background: #4338A8;
        box-shadow: 0 3px 10px rgba(67, 56, 168, 0.18);
    }
    .canvas-starter-grid button {
        border-color: #E5E3ED !important;
        border-radius: 12px !important;
    }
    .canvas-starter-grid button:hover {
        background: #FFFFFF !important;
        border-color: #A5B4FC !important;
    }
    .source-card {
        border-color: #EAE8F1;
        border-radius: 12px;
    }
    .telemetry-pill {
        background: #FFFFFF;
        border-color: #E8E6F0;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 4rem 1rem 6.5rem !important;
        }
        .canvas-welcome-banner {
            padding: 18px 8px 10px;
        }
        .canvas-welcome-title {
            font-size: 1.3rem;
        }
        .user-bubble-box {
            max-width: 92%;
        }
        .ai-card-wrapper {
            padding: 14px 16px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

try:
    import markdown_it
    _md_parser = markdown_it.MarkdownIt()
    def render_markdown_to_html(text: str) -> str:
        return _md_parser.render(text)
except Exception:
    def render_markdown_to_html(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

def render_sources_section(sources: List[Dict[str, Any]], ref_date: date) -> str:
    """Renders the 2-column grid of source cards matching the mockup."""
    if not sources:
        return ""

    cards_html = []
    for src in sources:
        title = src.get("title", "Official Announcement")
        pub_date = src.get("publication_date", "N/A")
        exp_date = src.get("expiration_date") or "Indefinite"
        source_url = src.get("source_url") or "#"

        # Check validity against reference date
        is_expired = False
        if src.get("expiration_date"):
            try:
                exp_d = parse_date(src["expiration_date"])
                if exp_d and exp_d < ref_date:
                    is_expired = True
            except Exception:
                pass

        badge_class = "badge-expired" if is_expired else "badge-valid"
        badge_text = "Expired" if is_expired else "Currently valid"

        card_html = f"""
        <div class="source-card">
            <div class="source-card-header">
                <div class="source-card-title" title="{html.escape(title)}">{html.escape(title)}</div>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            <div class="source-card-meta">
                Published {html.escape(pub_date)} · Valid until {html.escape(exp_date)}
            </div>
            <div class="source-card-footer">
                <a href="{html.escape(source_url)}" target="_blank" class="source-card-link">View source →</a>
            </div>
        </div>
        """
        cards_html.append(card_html)

    return f"""
    <div class="sources-container">
        <div class="sources-label">SOURCES</div>
        <div class="sources-grid">
            {"".join(cards_html)}
        </div>
    </div>
    """


def ingest_pdf_file(
    file_bytes: bytes,
    filename: str,
    title: str,
    category: str,
    pub_date: date,
    exp_date: Optional[date]
) -> Tuple[bool, str, int]:
    """
    Ingests a PDF file into ChromaDB vector database.
    Tries backend.services.pdf_parser if available; falls back to PyMuPDF.
    Returns (success, message, chunk_count).
    """
    # 1. Try dedicated PDF parser if available
    if HAS_PDF_PARSER:
        try:
            metadata = {
                "title": title,
                "category": category,
                "publication_date": pub_date.strftime("%Y-%m-%d"),
                "expiration_date": exp_date.strftime("%Y-%m-%d") if exp_date else "",
                "source_url": f"pdf://{filename}"
            }
            raw_docs = parse_pdf_from_bytes(file_bytes, metadata)
            if raw_docs:
                sync_result = vector_db_service.sync_knowledge_base(announcements=raw_docs)
                total_chunks = sum(d.get("chunks_count", 0) for d in sync_result.get("details", []))
                return True, f"Successfully parsed via pdf_parser ({sync_result.get('new', 0)} docs, {total_chunks} chunks).", total_chunks
        except Exception:
            pass  # Fall back to PyMuPDF below


    # 2. PyMuPDF (fitz) Text Extraction Fallback
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text() + "\n"

        clean_text = extracted_text.strip()
        if not clean_text:
            return False, "Could not extract text from PDF. The document may be an image scan or empty.", 0

        doc_id = f"pdf_{uuid.uuid4().hex[:8]}"
        pub_str = pub_date.strftime("%Y-%m-%d")
        exp_str = exp_date.strftime("%Y-%m-%d") if exp_date else ""

        full_text = f"Title: {title}\nCategory: {category}\nPublication Date: {pub_str}\n\n{clean_text}"
        content_hash = compute_content_hash(full_text)
        chunks = chunk_text(full_text, chunk_size=600, overlap=80)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metas = [{
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "publication_date": pub_str,
            "expiration_date": exp_str,
            "source_url": f"pdf://{filename}",
            "content_hash": content_hash,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        vector_db_service.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metas
        )
        return True, f"Successfully indexed '{title}' into ChromaDB ({len(chunks)} chunks created).", len(chunks)
    except Exception as ex:
        return False, f"Failed to parse and index PDF document: {str(ex)}", 0


# =========================================================================
# SESSION STATE INITIALIZATION
# =========================================================================
if "conversations" not in st.session_state:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state.conversations = {
        initial_id: {
            "title": "New Chat",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "👋 **Hello! I am the University AI Assistant for Hanoi University (HANU · FIT).**\n\n"
                        "I can answer questions regarding **tuition fees, graduation criteria, retake registration, "
                        "and academic regulations** grounded directly in official announcements.\n\n"
                        "Feel free to select one of the suggested topics on the left or type your inquiry below."
                    ),
                    "sources": [],
                    "telemetry": None
                }
            ]
        }
    }
    st.session_state.current_conv_id = initial_id

if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"  # "chat", "lab", "admin"

if "api_key_override" not in st.session_state:
    st.session_state.api_key_override = settings.GEMINI_API_KEY

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if st.session_state.current_conv_id not in st.session_state.conversations:
    first_id = list(st.session_state.conversations.keys())[0]
    st.session_state.current_conv_id = first_id


# =========================================================================
# SIDEBAR: BRANDING, TRY ASKING, HISTORY & SETTINGS
# =========================================================================
with st.sidebar:
    # 1. University Shield Logo & Assistant Title
    st.markdown("""
    <div class="brand-header-wrapper">
        <div class="brand-shield-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
        </div>
        <div class="brand-title-box">
            <span class="brand-title-main">University AI Assistant</span>
            <span class="brand-title-sub">Hanoi University · FIT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. "+ New Chat" Action Button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_conv_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[new_conv_id] = {
            "title": "New Chat",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello! What official university regulation or information would you like to explore today?",
                    "sources": [],
                    "telemetry": None
                }
            ]
        }
        st.session_state.current_conv_id = new_conv_id
        st.session_state.active_view = "chat"
        st.rerun()

    # 3. View Navigation (Chat / Lab / Admin)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.active_view == "chat" else "secondary"):
            st.session_state.active_view = "chat"
            st.rerun()
    with col_nav2:
        if st.button("🔬 Lab", use_container_width=True, type="primary" if st.session_state.active_view == "lab" else "secondary"):
            st.session_state.active_view = "lab"
            st.rerun()

    if st.button("🛡️ Admin Portal", use_container_width=True, type="primary" if st.session_state.active_view == "admin" else "secondary"):
        st.session_state.active_view = "admin"
        st.rerun()

    st.divider()

    # 4. "TRY ASKING" Section (Mockup Specification)
    st.markdown('<div class="sidebar-meta-label">TRY ASKING</div>', unsafe_allow_html=True)
    try_prompts = [
        "What is the tuition fee for Semester 1?",
        "What are the graduation requirements for Computer Science?",
        "How can I register for a retake course?",
        "Where can I find the curriculum for my major?"
    ]

    for idx, prompt_text in enumerate(try_prompts):
        st.markdown('<div class="prompt-row-wrapper">', unsafe_allow_html=True)
        if st.button(prompt_text, key=f"side_prompt_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.session_state.active_view = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # 5. Chat History List
    st.markdown('<div class="sidebar-meta-label">CHAT HISTORY</div>', unsafe_allow_html=True)
    for conv_id, conv_data in list(st.session_state.conversations.items()):
        is_active = (conv_id == st.session_state.current_conv_id and st.session_state.active_view == "chat")
        prefix = "👉 " if is_active else "💬 "
        btn_label = f"{prefix}{conv_data['title']}"

        col_c1, col_c2 = st.columns([5, 1])
        with col_c1:
            if st.button(btn_label, key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.current_conv_id = conv_id
                st.session_state.active_view = "chat"
                st.rerun()
        with col_c2:
            if len(st.session_state.conversations) > 1:
                if st.button("✕", key=f"del_{conv_id}", help="Delete chat"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conv_id == conv_id:
                        st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
                    st.rerun()

    st.divider()

    # 6. Settings Drawer (Simulated Date & Model display name: 'Gemini 2.5 Flash')
    with st.expander("⚙️ Settings & RAG Parameters", expanded=False):
        st.markdown("**LLM Model:** `Gemini 2.5 Flash`")
        user_gemini_key = st.text_input(
            "Gemini API Key:",
            value=st.session_state.api_key_override,
            type="password",
            help="Google Gemini API Key (https://aistudio.google.com/)"
        )
        if user_gemini_key != st.session_state.api_key_override:
            st.session_state.api_key_override = user_gemini_key

        simulated_date = st.date_input(
            "Reference Date (Simulated):",
            value=date.today(),
            help="Documents with expiration_date < this date are filtered out."
        )
        custom_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.10,
            max_value=0.80,
            value=settings.SIMILARITY_THRESHOLD,
            step=0.02
        )

    # 7. Sidebar Footer: Live sync indicator + Privacy Disclaimer (Mockup Specification)
    stats = vector_db_service.get_stats()
    last_sync = stats.get("last_sync")
    if last_sync and last_sync.get("timestamp"):
        try:
            sync_time_str = datetime.fromisoformat(last_sync["timestamp"]).strftime("%H:%M")
        except Exception:
            sync_time_str = last_sync["timestamp"][11:16]
    else:
        sync_time_str = datetime.now().strftime("%H:%M")

    st.markdown(f"""
    <div class="sidebar-footer-box">
        <div class="sync-status-indicator">
            <span class="sync-live-dot"></span>
            <span class="sync-status-text">Knowledge base synced {sync_time_str}</span>
        </div>
        <div class="privacy-disclaimer-text">
            Answers use public announcements and regulations only. No personal student data is accessed.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# MAIN AREA
# =========================================================================

# -------------------------------------------------------------------------
# VIEW 1: CHAT CANVAS (DEFAULT)
# -------------------------------------------------------------------------
if st.session_state.active_view == "chat":
    # 1. Top Bar of Main Area (Mockup Specification)
    st.markdown("""
    <div class="top-bar-container">
        <div class="top-bar-title-text">Ask about university information</div>
        <div class="official-badge-pill">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
            <span>Official sources only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    current_conv = st.session_state.conversations[st.session_state.current_conv_id]
    messages = current_conv["messages"]

    # 2. Fresh Conversation Welcome & Starter Cards
    prompt_from_card = None
    if len(messages) <= 1:
        st.markdown("""
        <div class="canvas-welcome-banner">
            <div class="canvas-welcome-title">🎓 University AI Assistant</div>
            <div class="canvas-welcome-sub">
                Official regulations, tuition schedules, retake procedures, and graduation guidelines powered by Temporal RAG.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="canvas-starter-grid">', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💰 Tuition Fee for Semester 1\nCredit rates, deadlines and payment methods", key="card_tuition", use_container_width=True):
                prompt_from_card = "What is the tuition fee for Semester 1?"
            if st.button("📝 Retake Course Registration\nStep-by-step procedure and exam conditions", key="card_retake", use_container_width=True):
                prompt_from_card = "How can I register for a retake course?"
        with col_s2:
            if st.button("🎓 Graduation Requirements (CS)\nCredits, cumulative GPA, and English exit standards", key="card_grad", use_container_width=True):
                prompt_from_card = "What are the graduation requirements for Computer Science?"
            if st.button("📚 Curriculum for Major\nCore foundation and specialized elective blocks", key="card_curr", use_container_width=True):
                prompt_from_card = "Where can I find the curriculum for my major?"
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Message Thread Rendering
    for msg in messages:
        if msg["role"] == "user":
            # Right-aligned indigo user bubble
            escaped_text = html.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(f"""
            <div class="user-bubble-row">
                <div class="user-bubble-box">{escaped_text}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Left-aligned AI message card with avatar "AI" (blue circle) and 2-column SOURCES grid
            with st.container():
                col_av, col_body = st.columns([1, 19], gap="small")
                with col_av:
                    st.markdown('<div class="ai-avatar-circle">AI</div>', unsafe_allow_html=True)
                with col_body:
                    rendered_content = render_markdown_to_html(msg["content"])
                    st.markdown(f'<div class="ai-card-wrapper">{rendered_content}</div>', unsafe_allow_html=True)

                    # 2-Column Sources Grid (Mockup Specification)
                    if msg.get("sources"):
                        sources_html = render_sources_section(msg["sources"], simulated_date)
                        st.markdown(sources_html, unsafe_allow_html=True)

                    # Expired Notices Expander
                    if msg.get("expired_sources"):
                        with st.expander(f"↻ Replaced by newer notices ({len(msg['expired_sources'])})", expanded=False):
                            for exp in msg["expired_sources"]:
                                meta = exp.get("metadata", {})
                                st.markdown(f"- **{meta.get('title')}** (Published: `{meta.get('publication_date')}`; an updated notice is being used)")

    # 4. Bottom Input Bar & Disclaimer Footer
    chat_input_val = st.chat_input("Ask about tuition, regulations, retakes, curriculum...")
    st.markdown("""
    <div class="chat-footer-note">
        Answers are generated from retrieved documents. Confirm important details in the linked source.
    </div>
    """, unsafe_allow_html=True)

    # 5. Handle Query Trigger (from sidebar TRY ASKING, starter cards, or chat input)
    prompt_to_send = None
    if st.session_state.get("pending_prompt"):
        prompt_to_send = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif prompt_from_card:
        prompt_to_send = prompt_from_card
    elif chat_input_val:
        prompt_to_send = chat_input_val

    if prompt_to_send:
        # Auto-update conversation title for first prompt
        if len(messages) <= 1 or current_conv["title"] == "New Chat":
            clean_title = prompt_to_send.strip()
            short_title = clean_title[:28] + ("..." if len(clean_title) > 28 else "")
            current_conv["title"] = short_title

        # Append User Message
        current_conv["messages"].append({"role": "user", "content": prompt_to_send})

        # Process Assistant Response with Temporal RAG Engine
        with st.spinner("Retrieving official university notices and synthesizing response..."):
            raw_chunks = vector_db_service.query_chunks(prompt_to_send, n_results=settings.TOP_K_CHUNKS)

            if not raw_chunks:
                reply_text = (
                    "⚠️ The knowledge base is currently empty. Please navigate to the "
                    "**Admin Portal** and click 'Sync Now' or upload official PDF documents."
                )
                sources = []
                expired_list = []
                telemetry_info = None
            else:
                selected_chunks, expired_chunks, meets_threshold = rank_and_filter_chunks(
                    chunks=raw_chunks,
                    current_date=simulated_date,
                    top_n=settings.TOP_N_RERANKED,
                    threshold=custom_threshold
                )

                telemetry_info = {
                    "scanned_chunks": len(raw_chunks),
                    "selected_chunks": len(selected_chunks),
                    "threshold": custom_threshold,
                    "expired_count": len(expired_chunks)
                }

                if not meets_threshold or not selected_chunks:
                    reply_text = (
                        "❌ **No matching official documents found.**\n\n"
                        "The system could not locate verified announcements corresponding to your inquiry "
                        "in the current knowledge base.\n\n"
                        "💡 **Recommendations:**\n"
                        "- Rephrase your question or use standard terminology (e.g., 'học phí', 'điều kiện tốt nghiệp').\n"
                        "- Contact the **FIT Academic Office (Room 201, Building C)** or visit the official training portal (https://qldt.hanu.edu.vn)."
                    )
                    sources = []
                    expired_list = expired_chunks
                else:
                    gen_result = generate_answer(
                        question=prompt_to_send,
                        chunks=selected_chunks,
                        api_key=st.session_state.api_key_override
                    )
                    reply_text = gen_result["answer"]
                    sources = gen_result["sources"]
                    expired_list = expired_chunks

            # Store Assistant Response in conversation history
            current_conv["messages"].append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
                "expired_sources": expired_list,
                "telemetry": telemetry_info
            })
            st.rerun()


# -------------------------------------------------------------------------
# VIEW 2: TEMPORAL RAG LAB
# -------------------------------------------------------------------------
elif st.session_state.active_view == "lab":
    st.markdown("### 🔬 **Temporal RAG Lab & Pipeline Inspector**")
    st.caption("Verify multi-stage temporal filtering, cosine scoring, and recency decay algorithms.")

    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        test_q = st.text_input("Test Query:", value="What is the tuition fee for Semester 1?")
    with col_l2:
        test_ref = st.date_input("Simulate Reference Date:", value=simulated_date)

    if st.button("🧪 Run Pipeline Step-by-Step", type="primary"):
        chunks = vector_db_service.query_chunks(test_q, n_results=settings.TOP_K_CHUNKS)
        if not chunks:
            st.warning("Knowledge base vector store is currently empty.")
        else:
            table_rows = []
            for c in chunks:
                meta = c.get("metadata", {})
                doc_id = meta.get("doc_id", "")
                title = meta.get("title", "")
                pub_str = meta.get("publication_date", "")
                exp_str = meta.get("expiration_date", "")
                pub_d = parse_date(pub_str)
                exp_d = parse_date(exp_str)

                is_exp = exp_d and (exp_d < test_ref)
                val_status = "❌ Expired" if is_exp else "✅ Valid"
                rec_boost = calculate_recency_boost(pub_d, test_ref) if not is_exp else 0.0
                sem = c.get("similarity", 0.0)
                final_score = round((0.85 * sem) + rec_boost, 4) if not is_exp else 0.0

                table_rows.append({
                    "Doc ID": doc_id,
                    "Title": title[:38] + ("..." if len(title) > 38 else ""),
                    "Published": pub_str,
                    "Expires": exp_str or "Indefinite",
                    "Cosine Sim": sem,
                    "Status": val_status,
                    "Recency Boost": f"+{rec_boost:.2f}",
                    "Final Score": final_score,
                    "Outcome": "Selected" if (not is_exp and sem >= custom_threshold) else "Excluded"
                })

            df_lab = pd.DataFrame(table_rows).sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_lab, use_container_width=True)


# -------------------------------------------------------------------------
# VIEW 3: ADMIN & INGESTION PORTAL (WITH NEW PDF UPLOADER)
# -------------------------------------------------------------------------
elif st.session_state.active_view == "admin":
    st.markdown("### 🛡️ **Admin Portal & Document Ingestion Engine**")
    admin_key_in = st.text_input("Enter Admin Key:", type="password", help="Default key: hanu_admin_secret_2026")

    if admin_key_in != settings.ADMIN_API_KEY:
        st.warning("Please provide a valid Admin Key to manage knowledge base synchronization and upload documents.")
    else:
        st.success("Admin authorization verified successfully.")

        # Quick Sync Actions & Metrics
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            if st.button("🔄 Sync Knowledge Base Now", type="primary", use_container_width=True):
                with st.spinner("Fetching portal announcements, calculating SHA-256 hashes, and updating ChromaDB..."):
                    sync_res = vector_db_service.sync_knowledge_base()
                    st.success(f"Sync complete at {sync_res['timestamp'][:19]}!")
                    st.json(sync_res)
                    st.rerun()

        stats = vector_db_service.get_stats()
        raw_docs = fetch_portal_announcements()
        m1, m2, m3 = st.columns(3)
        m1.metric("Indexed Chunks", stats["total_chunks"])
        m2.metric("Portal Notices", len(raw_docs))
        m3.metric("Total Syncs", stats["total_syncs"])

        st.divider()

        # =====================================================================
        # NEW SECTION: 📄 Upload PDF Document (Mockup Specification)
        # =====================================================================
        st.markdown("#### 📄 Upload PDF Document")
        st.caption("Upload official university documents, regulations, or circulars for automated parsing and vector chunk indexing.")

        uploaded_pdf = st.file_uploader("Select PDF file:", type=["pdf"], key="admin_pdf_uploader")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            default_title = Path(uploaded_pdf.name).stem if uploaded_pdf else ""
            pdf_title = st.text_input("Document Title:", value=default_title, placeholder="e.g., Quy chế Đào tạo Đại học Chính quy")
            pdf_category = st.selectbox(
                "Category:",
                ["Academic Regulations", "Tuition & Fees", "Curriculum", "Admissions", "Student Affairs", "General"]
            )
        with col_f2:
            pdf_pub_date = st.date_input("Publication Date:", value=date.today(), key="pdf_pub")
            has_exp = st.checkbox("Set Expiration Date (Optional)", value=False)
            pdf_exp_date = st.date_input("Expiration Date:", value=date.today(), key="pdf_exp") if has_exp else None

        if st.button("📤 Upload & Ingest PDF", type="primary", use_container_width=True):
            if not uploaded_pdf:
                st.error("Please choose a PDF file to upload.")
            elif not pdf_title.strip():
                st.error("Please enter a document title.")
            else:
                with st.spinner("Extracting text, computing checksums, and adding vector embeddings..."):
                    file_bytes = uploaded_pdf.getvalue()
                    success, msg, chunk_count = ingest_pdf_file(
                        file_bytes=file_bytes,
                        filename=uploaded_pdf.name,
                        title=pdf_title.strip(),
                        category=pdf_category,
                        pub_date=pdf_pub_date,
                        exp_date=pdf_exp_date
                    )
                    if success:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        st.divider()

        # Existing Document Catalog Table (SHA-256 Hashes)
        st.markdown("#### 📑 Document Catalog (SHA-256 Hashes)")
        catalog_rows = []
        for d in raw_docs:
            h = vector_db_service.get_document_stored_hash(d["id"]) or ""
            catalog_rows.append({
                "ID": d["id"],
                "Title": d["title"],
                "Category": d.get("category", "General"),
                "Published": d.get("publication_date", ""),
                "Expires": d.get("expiration_date") or "Indefinite",
                "SHA-256 Hash": h[:16] + "..." if len(h) > 16 else (h or "Not Synced")
            })
        st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True)
