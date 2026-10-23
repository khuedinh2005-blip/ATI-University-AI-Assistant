"""
Streamlit Web Application for Hanoi University AI Assistant (HANU).
Course: Advanced Topics in Information Technology (ATI) - Fall 2026.

ChatGPT-Style User Interface:
- Left Sidebar: New Chat Button (+), Multi-Session Chat History, Settings & Admin Portal.
- Center Canvas: Centered conversation thread, ChatGPT-style prompt cards, and bottom chat input.
- Temporal RAG Engine with transparent source citations and telemetry.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime, date

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

# --- Page Configuration ---
st.set_page_config(
    page_title="HANU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ChatGPT Style Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Constrain main conversation to ChatGPT-style center column */
    .main .block-container {
        max-width: 840px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 6rem !important;
        margin: 0 auto !important;
    }

    /* Top Centered Header */
    .chatgpt-header {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .chatgpt-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .chatgpt-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 12px;
    }

    /* System Status Badges */
    .status-pill-group {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        background: #f1f5f9;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #475569;
        font-weight: 500;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
    }

    /* ChatGPT Welcome Starter Cards */
    .starter-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .starter-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .starter-card:hover {
        border-color: #94a3b8;
        background: #f8fafc;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .starter-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #7a0c10;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .starter-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .starter-desc {
        font-size: 0.78rem;
        color: #64748b;
    }

    /* Citation Card */
    .citation-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #7a0c10;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .citation-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #0f172a;
    }
    .citation-meta {
        font-size: 0.76rem;
        color: #64748b;
        margin-bottom: 4px;
    }

    /* Badges */
    .badge-active {
        background-color: #ecfdf5;
        color: #059669;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-expired {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* Telemetry Pill in Chat */
    .chat-telemetry {
        display: inline-flex;
        gap: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 6px;
        margin-bottom: 8px;
    }

    /* Sidebar Conversation Item */
    .conv-item-btn {
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #334155;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .conv-item-btn:hover {
        background: #f1f5f9;
    }
    .conv-item-active {
        background: #e2e8f0;
        font-weight: 600;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# --- Multi-Session State Initialization ---
if "conversations" not in st.session_state:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state.conversations = {
        initial_id: {
            "title": "Hội thoại mới",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "👋 **Xin chào! Tôi là Trợ lý AI Trường Đại học Hà Nội (HANU Assistant).**\n\n"
                        "Tôi có thể giải đáp chính xác các thắc mắc về **quy chế đào tạo, mức thu học phí, "
                        "điều kiện tốt nghiệp, quy trình đăng ký học lại/thi lại** dựa trên các thông báo chính thức.\n\n"
                        "Bạn có thể chọn một chủ đề gợi ý bên dưới hoặc gõ trực tiếp câu hỏi vào thanh chat."
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


# Ensure current_conv_id is valid
if st.session_state.current_conv_id not in st.session_state.conversations:
    first_id = list(st.session_state.conversations.keys())[0]
    st.session_state.current_conv_id = first_id


# =========================================================================
# SIDEBAR: CHATGPT-STYLE HISTORY & CONTROLS
# =========================================================================
with st.sidebar:
    # App Logo & Branding
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Hanoi_University_logo.svg/330px-Hanoi_University_logo.svg.png",
        width=75
    )
    st.markdown("### 🎓 **HANU AI Assistant**")
    st.caption("Khoa Công nghệ Thông tin (FIT - HANU)")
    st.caption("Chuyên đề: *Advanced Topics in IT (ATI)*")

    # "+ New Chat" Button (like ChatGPT)
    if st.button("➕ Đoạn chat mới (New Chat)", type="primary", use_container_width=True):
        new_conv_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[new_conv_id] = {
            "title": "Hội thoại mới",
            "created_at": datetime.now().strftime("%H:%M"),
            "messages": [
                {
                    "role": "assistant",
                    "content": "Xin chào! Bạn muốn tra cứu thông tin gì về trường hôm nay?",
                    "sources": [],
                    "telemetry": None
                }
            ]
        }
        st.session_state.current_conv_id = new_conv_id
        st.session_state.active_view = "chat"
        st.rerun()

    st.divider()

    # Chat History List (ChatGPT-style list)
    st.markdown("##### 💬 **Lịch sử trò chuyện**")
    
    # Render each conversation item
    for conv_id, conv_data in list(st.session_state.conversations.items()):
        is_active = (conv_id == st.session_state.current_conv_id and st.session_state.active_view == "chat")
        prefix = "👉 💬 " if is_active else "💬 "
        button_label = f"{prefix}{conv_data['title']} ({conv_data['created_at']})"
        
        col_c1, col_c2 = st.columns([5, 1])
        with col_c1:
            if st.button(button_label, key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.current_conv_id = conv_id
                st.session_state.active_view = "chat"
                st.rerun()
        with col_c2:
            # Allow deleting conversation if more than 1 conversation exists
            if len(st.session_state.conversations) > 1:
                if st.button("✕", key=f"del_{conv_id}", help="Xóa đoạn chat này"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conv_id == conv_id:
                        st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]
                    st.rerun()

    st.divider()

    # View Mode Selectors (Chat vs. Lab vs. Admin)
    st.markdown("##### 🧭 **Chế độ làm việc**")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("💬 Chat", use_container_width=True, type="secondary" if st.session_state.active_view != "chat" else "primary"):
            st.session_state.active_view = "chat"
            st.rerun()
    with col_nav2:
        if st.button("🔬 Lab", use_container_width=True, type="secondary" if st.session_state.active_view != "lab" else "primary"):
            st.session_state.active_view = "lab"
            st.rerun()

    if st.button("🛡️ Quản trị viên (Admin Portal)", use_container_width=True, type="secondary" if st.session_state.active_view != "admin" else "primary"):
        st.session_state.active_view = "admin"
        st.rerun()

    st.divider()

    # Settings Drawer
    with st.expander("⚙️ Cài đặt & Tham số RAG", expanded=False):
        st.markdown("**Mô hình LLM:** `gemini-3.5-flash-lite`")
        user_gemini_key = st.text_input(
            "Gemini API Key:",
            value=st.session_state.api_key_override,
            type="password",
            help="Khóa Google Gemini API (lấy tại https://aistudio.google.com/)"
        )
        if user_gemini_key != st.session_state.api_key_override:
            st.session_state.api_key_override = user_gemini_key

        simulated_date = st.date_input(
            "Ngày tham chiếu (Reference Date):",
            value=date.today(),
            help="Văn bản có expiration_date < ngày này sẽ bị loại trừ."
        )
        custom_threshold = st.slider(
            "Ngưỡng tương đồng (Threshold):",
            min_value=0.30,
            max_value=0.80,
            value=settings.SIMILARITY_THRESHOLD,
            step=0.02
        )


# =========================================================================
# MAIN AREA: CENTERED CHATGPT-STYLE CANVAS
# =========================================================================

# --- VIEW 1: CHAT CANVAS (DEFAULT) ---
if st.session_state.active_view == "chat":
    # Centered Header
    db_count = vector_db_service.collection.count()
    st.markdown(f"""
    <div class="chatgpt-header">
        <div class="chatgpt-title">🎓 Trợ lý AI Đại học Hà Nội</div>
        <div class="chatgpt-subtitle">Hệ thống Tra cứu Thông tin & Quy chế Đào tạo (Temporal-Aware RAG)</div>
        <div class="status-pill-group">
            <span class="status-dot"></span>
            <span>ChromaDB: <strong>{db_count} chunks</strong></span>
            <span>•</span>
            <span>Mô hình: <strong>Gemini 3.5 Flash</strong></span>
            <span>•</span>
            <span>Ngày lọc: <strong>{simulated_date.strftime('%d/%m/%Y')}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    current_conv = st.session_state.conversations[st.session_state.current_conv_id]
    messages = current_conv["messages"]

    # If this is a fresh conversation with only initial greeting, show Starter Prompt Cards
    prompt_from_card = None
    if len(messages) <= 1:
        st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 15px;'>Gợi ý chủ đề thường gặp:</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💰 Học phí Học kỳ 1 năm học 2026-2027\nMức thu tín chỉ chính quy và thời hạn đóng", key="card1", use_container_width=True):
                prompt_from_card = "Học phí Học kỳ 1 năm học 2026-2027 là bao nhiêu tiền một tín chỉ và thời hạn nộp là khi nào?"
            if st.button("📝 Đăng ký học lại và thi lại học phần\nQuy trình 4 bước và điều kiện dự thi", key="card3", use_container_width=True):
                prompt_from_card = "Các bước cụ thể để đăng ký học lại môn học và điều kiện tham gia thi lại là gì?"
        with c2:
            if st.button("🎓 Điều kiện tốt nghiệp ngành CNTT\n135 tín chỉ, GPA và chuẩn ngoại ngữ", key="card2", use_container_width=True):
                prompt_from_card = "Chuẩn đầu ra và điều kiện để xét tốt nghiệp Cử nhân ngành Công nghệ Thông tin gồm những tiêu chuẩn gì?"
            if st.button("📚 Khung chương trình đào tạo ngành CNTT\nCác khối kiến thức cơ sở và chuyên ngành", key="card4", use_container_width=True):
                prompt_from_card = "Khung chương trình đào tạo ngành Công nghệ Thông tin gồm bao nhiêu tín chỉ và những môn chuyên ngành nào?"

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Render Active Conversation Messages
    for msg in messages:
        with st.chat_message(msg["role"], avatar="👨‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

            # Telemetry Pill
            if msg.get("telemetry"):
                t = msg["telemetry"]
                st.markdown(f"""
                <div class="chat-telemetry">
                    <span>⚡ Quét: <strong>{t['scanned_chunks']} chunks</strong></span>
                    <span>•</span>
                    <span>🎯 Chọn lọc: <strong>{t['selected_chunks']} chunks</strong> (Ngưỡng: {t['threshold']})</span>
                    <span>•</span>
                    <span>⏳ Lọc hết hạn: <strong>{t['expired_count']} thông báo cũ</strong></span>
                </div>
                """, unsafe_allow_html=True)

            # Source Citations
            if msg.get("sources"):
                with st.expander(f"📚 Văn bản nguồn trích dẫn ({len(msg['sources'])} tài liệu xác thực)", expanded=True):
                    for idx, src in enumerate(msg["sources"], 1):
                        sim_pct = int(src.get("similarity_score", 0.0) * 100)
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-title">[{idx}] {src['title']}</span>
                                <span class="badge-active">Đang có hiệu lực</span>
                            </div>
                            <div class="citation-meta">
                                📅 Ngày ban hành: <strong>{src['publication_date']}</strong> | Hạn: <strong>{src.get('expiration_date') or 'Vô hạn'}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                <span style="font-size: 0.76rem; color: #047857; font-weight: 600;">Độ tương đồng: {sim_pct}% (Score: {src.get('combined_score', src['similarity_score'])})</span>
                                <a href="{src['source_url']}" target="_blank" style="font-size: 0.78rem; color: #7a0c10; font-weight: 600; text-decoration: none;">Xem văn bản gốc ↗</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Expired Filter Transparency
            if msg.get("expired_sources"):
                with st.expander(f"⏳ Lọc bỏ {len(msg['expired_sources'])} thông báo cũ hết hạn", expanded=False):
                    for exp in msg["expired_sources"]:
                        meta = exp.get("metadata", {})
                        st.markdown(f"- ❌ **{meta.get('title')}** (Hết hạn: `{meta.get('expiration_date')}` | Ban hành: `{meta.get('publication_date')}`)")

    # Bottom Chat Input (Sticky center bar)
    chat_input_val = st.chat_input("Hỏi HANU AI Assistant về quy chế, học phí, lịch học...")
    prompt_to_send = prompt_from_card or chat_input_val

    if prompt_to_send:
        # Auto-update conversation title if it's the first user message
        if len(messages) <= 1 or current_conv["title"] == "Hội thoại mới":
            clean_title = prompt_to_send.strip()
            short_title = clean_title[:24] + ("..." if len(clean_title) > 24 else "")
            current_conv["title"] = short_title

        # Append User Message
        current_conv["messages"].append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(prompt_to_send)

        # Assistant Synthesis
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Đang tìm kiếm thông báo chính thức và tổng hợp câu trả lời..."):
                raw_chunks = vector_db_service.query_chunks(prompt_to_send, n_results=settings.TOP_K_CHUNKS)

                if not raw_chunks:
                    reply_text = "⚠️ Cơ sở dữ liệu thông báo đang trống. Vui lòng chuyển sang tab Quản trị và thực hiện 'Đồng bộ ngay'."
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
                            "❌ **Hiện tại hệ thống không tìm thấy văn bản chính thức phù hợp** "
                            "với câu hỏi của bạn trong cơ sở dữ liệu các thông báo của nhà trường.\n\n"
                            "💡 **Khuyến nghị:**\n"
                            "- Bạn vui lòng kiểm tra lại câu hỏi hoặc dùng từ khóa liên quan hơn.\n"
                            "- Liên hệ trực tiếp **Văn phòng Khoa CNTT (Phòng 201 Nhà C)** hoặc Cổng Quản lý đào tạo (https://qldt.hanu.edu.vn) để được hỗ trợ cụ thể."
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

                # Display reply
                st.markdown(reply_text)

                if telemetry_info:
                    st.markdown(f"""
                    <div class="chat-telemetry">
                        <span>⚡ Quét: <strong>{telemetry_info['scanned_chunks']} chunks</strong></span>
                        <span>•</span>
                        <span>🎯 Chọn lọc: <strong>{telemetry_info['selected_chunks']} chunks</strong> (Ngưỡng: {telemetry_info['threshold']})</span>
                        <span>•</span>
                        <span>⏳ Lọc hết hạn: <strong>{telemetry_info['expired_count']} thông báo cũ</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📚 Văn bản nguồn trích dẫn ({len(sources)} tài liệu xác thực)", expanded=True):
                        for idx, src in enumerate(sources, 1):
                            sim_pct = int(src.get("similarity_score", 0.0) * 100)
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-title">[{idx}] {src['title']}</span>
                                    <span class="badge-active">Đang có hiệu lực</span>
                                </div>
                                <div class="citation-meta">
                                    📅 Ngày ban hành: <strong>{src['publication_date']}</strong> | Hạn: <strong>{src.get('expiration_date') or 'Vô hạn'}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span style="font-size: 0.76rem; color: #047857; font-weight: 600;">Độ liên quan kết hợp: {src.get('combined_score', src['similarity_score'])}</span>
                                    <a href="{src['source_url']}" target="_blank" style="font-size: 0.78rem; color: #7a0c10; font-weight: 600; text-decoration: none;">Xem văn bản gốc ↗</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

        # Store to conversation messages
        current_conv["messages"].append({
            "role": "assistant",
            "content": reply_text,
            "sources": sources,
            "expired_sources": expired_list,
            "telemetry": telemetry_info
        })
        st.rerun()


# --- VIEW 2: TEMPORAL RAG LAB ---
elif st.session_state.active_view == "lab":
    st.markdown("### 🔬 **Phòng thí nghiệm Xếp hạng Thời gian (Temporal RAG Lab)**")
    st.caption("Kiểm chứng thuật toán xếp hạng và lọc thời gian từng bước phục vụ báo cáo khoa học ATI.")

    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        test_q = st.text_input("Câu hỏi kiểm thử:", value="Mức thu học phí Học kỳ 1 năm học 2026-2027")
    with col_l2:
        test_ref = st.date_input("Mô phỏng ngày tham chiếu:", value=simulated_date)

    if st.button("🧪 Phân tích Pipeline từng bước", type="primary"):
        chunks = vector_db_service.query_chunks(test_q, n_results=settings.TOP_K_CHUNKS)
        if not chunks:
            st.warning("Cơ sở dữ liệu đang trống.")
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
                val_status = "❌ Hết hạn" if is_exp else "✅ Hợp lệ"
                rec_boost = calculate_recency_boost(pub_d, test_ref) if not is_exp else 0.0
                sem = c.get("similarity", 0.0)
                final_score = round((0.85 * sem) + rec_boost, 4) if not is_exp else 0.0

                table_rows.append({
                    "Mã": doc_id,
                    "Tiêu đề": title[:38] + "...",
                    "Ban hành": pub_str,
                    "Hạn": exp_str or "Vô hạn",
                    "Cosine Sim": sem,
                    "Hiệu lực": val_status,
                    "Recency Boost": f"+{rec_boost:.2f}",
                    "Final Score": final_score,
                    "Context": "Được chọn" if (not is_exp and sem >= custom_threshold) else "Loại trừ"
                })

            df_lab = pd.DataFrame(table_rows).sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_lab, use_container_width=True)


# --- VIEW 3: ADMIN & INGESTION PORTAL ---
elif st.session_state.active_view == "admin":
    st.markdown("### 🛡️ **Bảng Quản trị & Đồng bộ Cổng Đào tạo (Admin Portal)**")
    admin_key_in = st.text_input("Nhập Khóa Quản trị (Admin Key):", type="password")

    if admin_key_in != settings.ADMIN_API_KEY:
        st.warning("Vui lòng nhập đúng Khóa Quản trị (Mặc định: `hanu_admin_secret_2026`).")
    else:
        st.success("Xác thực Quản trị viên thành công!")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            if st.button("🔄 Đồng bộ ngay (Sync Now)", type="primary", use_container_width=True):
                with st.spinner("Đang kết nối API, tính mã băm SHA-256 và cập nhật ChromaDB..."):
                    sync_res = vector_db_service.sync_knowledge_base()
                    st.success(f"Đồng bộ thành công lúc {sync_res['timestamp'][:19]}!")
                    st.json(sync_res)
                    st.rerun()

        # Metrics
        stats = vector_db_service.get_stats()
        raw_docs = fetch_portal_announcements()
        m1, m2, m3 = st.columns(3)
        m1.metric("Vector Chunks", stats["total_chunks"])
        m2.metric("Văn bản gốc", len(raw_docs))
        m3.metric("Số lần Sync", stats["total_syncs"])

        st.divider()
        st.markdown("#### 📑 Danh mục Văn bản (SHA-256 Hashes)")
        inv = []
        for d in raw_docs:
            h = vector_db_service.get_document_stored_hash(d["id"]) or ""
            inv.append({
                "ID": d["id"],
                "Tiêu đề": d["title"],
                "Ban hành": d.get("publication_date", ""),
                "Hạn": d.get("expiration_date", "Vô hạn"),
                "Mã băm SHA-256": h[:16] + "..." if len(h) > 16 else h
            })
        st.dataframe(pd.DataFrame(inv), use_container_width=True)
