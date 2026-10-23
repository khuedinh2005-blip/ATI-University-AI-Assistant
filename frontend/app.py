"""
Streamlit Web Application for Hanoi University AI Assistant (HANU).
Course: Advanced Topics in Information Technology (ATI) - Fall 2026.

Scientific & Academic User Interface:
1. Academic QA Chat (Telemetry, Temporal Validity Badges, Detailed Citation Cards).
2. Temporal RAG Lab (Interactive ranking inspection, threshold slider, simulation date).
3. Admin & Ingestion Dashboard (Metrics, Document Inventory, SHA-256 Hash status, Sync Logs).
4. System Architecture & Mathematical Specifications.
"""

import sys
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
    page_title="HANU University AI Assistant | ATI Fall 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Academic Design System (CSS) ---
st.markdown("""
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Academic Branding Header */
    .hanu-header {
        background: linear-gradient(135deg, #7a0c10 0%, #a3151a 100%);
        color: #ffffff;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(122, 12, 16, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hanu-header-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hanu-header-subtitle {
        font-size: 0.92rem;
        color: #fde8e8;
        margin: 4px 0 0 0;
        opacity: 0.95;
    }

    /* Real-Time Telemetry Bar */
    .telemetry-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        font-size: 0.82rem;
        color: #475569;
        margin-bottom: 18px;
    }
    .telemetry-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot-blue {
        width: 8px;
        height: 8px;
        background-color: #3b82f6;
        border-radius: 50%;
        display: inline-block;
    }

    /* Prompt Suggestion Card */
    .prompt-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        transition: all 0.2s ease;
        margin-bottom: 8px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .prompt-card:hover {
        border-color: #a3151a;
        box-shadow: 0 3px 10px rgba(163, 21, 26, 0.08);
    }
    .prompt-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #7a0c10;
        background: #fdf2f2;
        padding: 2px 7px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .prompt-title {
        font-size: 0.86rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
        line-height: 1.35;
    }

    /* Citation Box */
    .citation-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #7a0c10;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .citation-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #0f172a;
    }
    .citation-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 6px;
    }
    .score-meter {
        background: #f1f5f9;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #047857;
    }

    /* Badges */
    .badge-active {
        background-color: #ecfdf5;
        color: #059669;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-expired {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Telemetry Pill in Chat */
    .chat-telemetry {
        display: inline-flex;
        gap: 12px;
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        color: #475569;
        margin-top: 6px;
        margin-bottom: 8px;
    }

    /* Metric Cards */
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7a0c10;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Xin chào! Tôi là Trợ lý AI Trường Đại học Hà Nội (HANU Assistant).**\n\n"
                "Tôi được trang bị hệ thống **Temporal-Aware RAG** nhằm tra cứu và giải đáp chính xác "
                "các văn bản, quy chế đào tạo, mức thu học phí và lịch trình học vụ chính thức từ Cổng Quản lý đào tạo.\n\n"
                "Bạn có thể chọn một trong các câu hỏi nghiên cứu bên trên hoặc nhập trực tiếp câu hỏi của mình."
            ),
            "sources": [],
            "telemetry": None
        }
    ]

if "api_key_override" not in st.session_state:
    st.session_state.api_key_override = settings.GEMINI_API_KEY

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


# --- Sidebar: System Status & Parameters ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Hanoi_University_logo.svg/330px-Hanoi_University_logo.svg.png",
        width=85
    )
    st.markdown("### 🎓 **HANU AI ASSISTANT**")
    st.caption("Khoa Công nghệ Thông tin (FIT - HANU)")
    st.caption("Chuyên đề: *Advanced Topics in IT (ATI) - Fall 2026*")
    st.divider()

    # Model & API Credentials
    st.markdown("#### 🧠 **Cấu hình Mô hình (LLM Engine)**")
    user_gemini_key = st.text_input(
        "Gemini API Key:",
        value=st.session_state.api_key_override,
        type="password",
        help="Khóa Google Gemini API (lấy tại https://aistudio.google.com/)"
    )
    if user_gemini_key != st.session_state.api_key_override:
        st.session_state.api_key_override = user_gemini_key

    model_status_badge = "🟢 Sẵn sàng (Live API)" if st.session_state.api_key_override else "🟡 Chế độ mô phỏng (Offline)"
    st.caption(f"**Trạng thái API:** {model_status_badge}")
    st.caption("**Model:** `gemini-3.5-flash-lite` (Google GenAI)")

    st.divider()

    # Temporal Filter Parameters
    st.markdown("#### ⏳ **Tham số Lọc Thời gian (Temporal RAG)**")
    simulated_date = st.date_input(
        "Ngày tham chiếu (Reference Date):",
        value=date.today(),
        help="Thuật toán Validity Filtering sẽ tự động loại bỏ các văn bản có hạn hiệu lực kết thúc trước ngày này."
    )
    custom_threshold = st.slider(
        "Ngưỡng tương đồng tối thiểu (Threshold):",
        min_value=0.30,
        max_value=0.80,
        value=settings.SIMILARITY_THRESHOLD,
        step=0.02,
        help="Nếu điểm tương đồng Cosine của chunk cao nhất nhỏ hơn ngưỡng này, hệ thống sẽ kích hoạt cơ chế chống ảo giác và trả về hướng dẫn liên hệ khoa."
    )

    st.divider()

    # Reset Chat Session
    if st.button("🗑️ Làm mới cuộc hội thoại (New Chat)", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Cuộc trò chuyện đã được làm mới. Mời bạn đặt câu hỏi cần tra cứu.",
                "sources": [],
                "telemetry": None
            }
        ]
        st.rerun()


# --- Top Academic Header ---
st.markdown("""
<div class="hanu-header">
    <div>
        <h1 class="hanu-header-title">Trợ lý AI Trường Đại học Hà Nội (HANU Assistant)</h1>
        <p class="hanu-header-subtitle">Hệ thống Hỏi đáp & Tra cứu Học thuật Thời gian thực (Temporal-Aware RAG Architecture)</p>
    </div>
    <div style="text-align: right; font-size: 0.8rem; background: rgba(255,255,255,0.15); padding: 8px 14px; border-radius: 8px;">
        <strong>Học kỳ:</strong> Fall - 2026<br/>
        <strong>Khoa:</strong> Công nghệ Thông tin (FIT)
    </div>
</div>
""", unsafe_allow_html=True)

# --- Real-Time Telemetry Bar ---
db_count = vector_db_service.collection.count()
last_sync = vector_db_service.sync_logs[0]["timestamp"][:19].replace("T", " ") if vector_db_service.sync_logs else "Khởi tạo sẵn"

st.markdown(f"""
<div class="telemetry-bar">
    <div class="telemetry-item"><span class="status-dot-green"></span> <strong>ChromaDB Vector Store:</strong> Đang hoạt động ({db_count} chunks)</div>
    <div class="telemetry-item">•</div>
    <div class="telemetry-item"><span class="status-dot-blue"></span> <strong>Temporal Filter:</strong> {simulated_date.strftime('%d/%m/%Y')}</div>
    <div class="telemetry-item">•</div>
    <div class="telemetry-item"><strong>Ngưỡng tương đồng:</strong> {custom_threshold}</div>
    <div class="telemetry-item">•</div>
    <div class="telemetry-item"><strong>Lần đồng bộ gần nhất:</strong> {last_sync}</div>
</div>
""", unsafe_allow_html=True)


# --- Tabs Navigation ---
tab_chat, tab_lab, tab_admin, tab_architecture = st.tabs([
    "💬 Hỏi đáp Học thuật (Academic QA)",
    "🔬 Phòng thí nghiệm Xếp hạng Thời gian (Temporal Lab)",
    "🛡️ Quản trị & Đồng bộ Cổng trường (Admin Portal)",
    "🏛️ Đặc tả Kiến trúc & Thuật toán (Architecture)"
])


# =========================================================================
# TAB 1: ACADEMIC CHAT & QA
# =========================================================================
with tab_chat:
    # 4 Categorized Prompt Cards
    st.markdown("##### 📌 **Gợi ý Tra cứu Nhanh (Click để gửi câu hỏi):**")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    selected_prompt = None

    with col_p1:
        st.markdown("""
        <div class="prompt-card">
            <div>
                <span class="prompt-tag">Học phí & Lệ phí</span>
                <div class="prompt-title">Mức thu học phí Học kỳ 1 năm học 2026-2027</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Hỏi mục này ↗", key="btn_p1", use_container_width=True):
            selected_prompt = "Học phí Học kỳ 1 năm học 2026-2027 là bao nhiêu tiền một tín chỉ và thời hạn nộp là khi nào?"

    with col_p2:
        st.markdown("""
        <div class="prompt-card">
            <div>
                <span class="prompt-tag">Tốt nghiệp & CĐR</span>
                <div class="prompt-title">Chuẩn đầu ra và Điều kiện tốt nghiệp CNTT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Hỏi mục này ↗", key="btn_p2", use_container_width=True):
            selected_prompt = "Chuẩn đầu ra và điều kiện để xét tốt nghiệp Cử nhân ngành Công nghệ Thông tin gồm những tiêu chuẩn gì?"

    with col_p3:
        st.markdown("""
        <div class="prompt-card">
            <div>
                <span class="prompt-tag">Học vụ & Thi lại</span>
                <div class="prompt-title">Quy trình đăng ký học lại và thi lại học phần</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Hỏi mục này ↗", key="btn_p3", use_container_width=True):
            selected_prompt = "Các bước cụ thể để đăng ký học lại môn học và điều kiện tham gia thi lại là gì?"

    with col_p4:
        st.markdown("""
        <div class="prompt-card">
            <div>
                <span class="prompt-tag">Khung Đào tạo</span>
                <div class="prompt-title">Khung chương trình đào tạo ngành CNTT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Hỏi mục này ↗", key="btn_p4", use_container_width=True):
            selected_prompt = "Khung chương trình đào tạo ngành Công nghệ Thông tin gồm bao nhiêu tín chỉ và những môn chuyên ngành nào?"

    st.divider()

    # Render Conversation Thread
    for msg in st.session_state.messages:
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

            # Source Citations Grid
            if msg.get("sources"):
                with st.expander(f"📚 Văn bản nguồn tham chiếu ({len(msg['sources'])} tài liệu được xác thực)", expanded=True):
                    for idx, src in enumerate(msg["sources"], 1):
                        is_active = (not src.get("expiration_date")) or (parse_date(src["expiration_date"]) >= simulated_date)
                        status_badge = '<span class="badge-active">Đang có hiệu lực</span>' if is_active else '<span class="badge-expired">Đã hết hạn</span>'
                        sim_pct = int(src.get("similarity_score", 0.0) * 100)

                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-title">[{idx}] {src['title']}</span>
                                <span>{status_badge}</span>
                            </div>
                            <div class="citation-meta">
                                📅 <strong>Ngày ban hành:</strong> {src['publication_date']} | 
                                ⏳ <strong>Hạn hiệu lực:</strong> {src.get('expiration_date') or 'Vô thời hạn'} | 
                                🏷️ <strong>Chuyên mục:</strong> {src.get('category', 'Chung')}
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 6px;">
                                <span class="score-meter">Độ tương đồng ngữ nghĩa: {sim_pct}% (Combined Score: {src.get('combined_score', src['similarity_score'])})</span>
                                <a href="{src['source_url']}" target="_blank" style="font-size: 0.8rem; color: #7a0c10; font-weight: 600; text-decoration: none;">Xem văn bản gốc ↗</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Expired Filter Transparency
            if msg.get("expired_sources"):
                with st.expander(f"⏳ Báo cáo Lọc Thời gian: {len(msg['expired_sources'])} thông báo cũ bị loại trừ tự động", expanded=False):
                    st.caption("Các thông báo dưới đây khớp một phần về ngữ nghĩa nhưng đã hết hiệu lực tính đến ngày tham chiếu, nên đã bị loại trừ để bảo đảm tính chuẩn xác của câu trả lời:")
                    for exp in msg["expired_sources"]:
                        meta = exp.get("metadata", {})
                        st.markdown(f"- ❌ **{meta.get('title')}** (Hết hạn: `{meta.get('expiration_date')}` | Ban hành: `{meta.get('publication_date')}`)")

    # Chat Input Box
    user_input = st.chat_input("Nhập câu hỏi của bạn về quy chế đào tạo, học phí, lịch học tại HANU...")
    prompt_to_send = selected_prompt or user_input

    if prompt_to_send:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(prompt_to_send)

        # Assistant Synthesis
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Đang thực hiện truy vấn ngữ nghĩa, lọc thời gian và tổng hợp phản hồi..."):
                raw_chunks = vector_db_service.query_chunks(prompt_to_send, n_results=settings.TOP_K_CHUNKS)

                if not raw_chunks:
                    reply_text = "⚠️ Cơ sở dữ liệu thông báo đang trống. Vui lòng chuyển sang tab Quản trị và thực hiện 'Đồng bộ ngay'."
                    sources = []
                    expired_list = []
                    telemetry_info = None
                else:
                    # Apply Temporal Filtering & Recency Re-ranking
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
                            "❌ **Hiện tại hệ thống không tìm thấy văn bản hoặc thông báo chính thức phù hợp** "
                            "với câu hỏi của bạn trong cơ sở dữ liệu của nhà trường.\n\n"
                            "💡 **Khuyến nghị:**\n"
                            "- Kiểm tra lại từ khóa hoặc cách diễn đạt câu hỏi.\n"
                            "- Liên hệ trực tiếp **Văn phòng Khoa CNTT (Phòng 201 Nhà C)** hoặc Cổng Quản lý đào tạo (https://qldt.hanu.edu.vn) để được hỗ trợ thủ tục cụ thể."
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

                # Render reply
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
                    with st.expander(f"📚 Văn bản nguồn tham chiếu ({len(sources)} tài liệu)", expanded=True):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-title">[{idx}] {src['title']}</span>
                                    <span class="badge-active">Đang có hiệu lực</span>
                                </div>
                                <div class="citation-meta">
                                    📅 <strong>Ngày ban hành:</strong> {src['publication_date']} | 
                                    ⏳ <strong>Hạn hiệu lực:</strong> {src.get('expiration_date') or 'Vô thời hạn'}
                                </div>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span class="score-meter">Độ liên quan kết hợp: {src.get('combined_score', src['similarity_score'])}</span>
                                    <a href="{src['source_url']}" target="_blank" style="font-size: 0.8rem; color: #7a0c10; font-weight: 600; text-decoration: none;">Xem văn bản gốc ↗</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

        # Store to conversation state
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "sources": sources,
            "expired_sources": expired_list,
            "telemetry": telemetry_info
        })
        st.rerun()


# =========================================================================
# TAB 2: TEMPORAL RAG LAB (SCIENTIFIC BENCHMARK & EXPERIMENT)
# =========================================================================
with tab_lab:
    st.markdown("### 🔬 **Phòng thí nghiệm Xếp hạng Thời gian (Temporal RAG Lab)**")
    st.markdown("""
    Khu vực này cho phép giảng viên, hội đồng đánh giá hoặc nhà nghiên cứu kiểm chứng **từng bước** 
    trong thuật toán xếp hạng thời gian mà nhóm sinh viên đã đề xuất trong Báo cáo giữa kỳ.
    """)

    col_lab1, col_lab2 = st.columns([2, 1])
    with col_lab1:
        test_query = st.text_input(
            "Nhập câu hỏi thử nghiệm:",
            value="Mức thu học phí Học kỳ 1 năm học 2026-2027",
            help="Câu hỏi dùng để chạy phân tích retrieval từng bước"
        )
    with col_lab2:
        lab_ref_date = st.date_input("Mô phỏng ngày tham chiếu:", value=simulated_date, key="lab_date")

    if st.button("🧪 Chạy phân tích Retrieval & Re-ranking", type="primary"):
        with st.spinner("Đang tính toán vector similarity và áp dụng mô hình toán học..."):
            raw_chunks = vector_db_service.query_chunks(test_query, n_results=settings.TOP_K_CHUNKS)

            if not raw_chunks:
                st.warning("Cơ sở dữ liệu đang trống.")
            else:
                table_rows = []
                for c in raw_chunks:
                    meta = c.get("metadata", {})
                    doc_id = meta.get("doc_id", "")
                    title = meta.get("title", "")
                    pub_str = meta.get("publication_date", "")
                    exp_str = meta.get("expiration_date", "")
                    
                    pub_d = parse_date(pub_str)
                    exp_d = parse_date(exp_str)
                    
                    # Check validity
                    is_expired = exp_d and (exp_d < lab_ref_date)
                    validity_status = "❌ Hết hạn (Đã lọc bỏ)" if is_expired else "✅ Hợp lệ"
                    
                    # Calculate recency boost
                    recency_boost = calculate_recency_boost(pub_d, lab_ref_date) if not is_expired else 0.0
                    sem_sim = c.get("similarity", 0.0)
                    combined_score = round((0.85 * sem_sim) + recency_boost, 4) if not is_expired else 0.0
                    
                    table_rows.append({
                        "Mã tài liệu": doc_id,
                        "Tiêu đề thông báo": title[:40] + ("..." if len(title) > 40 else ""),
                        "Ngày ban hành": pub_str,
                        "Hạn hiệu lực": exp_str or "Vô hạn",
                        "Cosine Sim": sem_sim,
                        "Kiểm tra hiệu lực": validity_status,
                        "Recency Boost": f"+{recency_boost:.2f}",
                        "Điểm kết hợp (Final)": combined_score,
                        "Trạng thái": "Được chọn làm Context" if (not is_expired and sem_sim >= custom_threshold) else "Loại trừ"
                    })

                df_results = pd.DataFrame(table_rows)
                # Sort descending by Combined Score
                df_results = df_results.sort_values(by="Điểm kết hợp (Final)", ascending=False).reset_index(drop=True)

                st.markdown("#### 📊 **Bảng Kết quả Phân tích Từng Bước (Pipeline Output):**")
                st.dataframe(df_results, use_container_width=True)

                st.markdown(r"""
                > **Giải thích công thức toán học:**
                > 1. $\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ (được chuẩn hóa từ khoảng cách của ChromaDB).
                > 2. $\text{Validity Filter}$: Nếu $\text{expiration\_date} < \text{reference\_date}$, chunk bị loại bỏ với trọng số $= 0$.
                > 3. $\text{Recency Boost}$: Căn cứ vào khoảng cách ngày $\Delta = \text{reference\_date} - \text{publication\_date}$:
                >    - $\Delta \le 30 \text{ ngày} \rightarrow +0.15$
                >    - $\Delta \le 90 \text{ ngày} \rightarrow +0.10$
                >    - $\Delta \le 180 \text{ ngày} \rightarrow +0.05$
                > 4. $\mathbf{S_{\text{final}}} = 0.85 \times \text{Cosine Sim} + \text{Recency Boost}$.
                """)


# =========================================================================
# TAB 3: ADMIN & INGESTION DASHBOARD
# =========================================================================
with tab_admin:
    st.markdown("### 🛡️ **Bảng điều khiển Quản trị viên (Admin & Ingestion Portal)**")
    st.caption("Quản lý đường ống thu thập dữ liệu (Data Ingestion), đối soát mã băm SHA-256 và giám sát ChromaDB.")

    col_auth1, col_auth2 = st.columns([2, 1])
    with col_auth1:
        admin_key_input = st.text_input(
            "Nhập Khóa bảo mật Quản trị (Admin Secret Key):",
            type="password",
            help="Mặc định trong file .env: hanu_admin_secret_2026"
        )
    with col_auth2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if admin_key_input == settings.ADMIN_API_KEY:
            st.session_state.admin_authenticated = True
            st.success("Xác thực Quản trị viên thành công!")
        elif admin_key_input:
            st.error("Mã bảo mật không chính xác!")

    if st.session_state.admin_authenticated:
        st.divider()

        # Ingestion Trigger Button
        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            if st.button("🔄 Đồng bộ ngay (Trigger Sync Now)", type="primary", use_container_width=True):
                with st.spinner("Đang kết nối API Cổng đào tạo, trích xuất HTML, băm SHA-256 và cập nhật ChromaDB..."):
                    sync_res = vector_db_service.sync_knowledge_base()
                    st.success(f"Đã hoàn thành đồng bộ lúc {sync_res['timestamp'][:19]}!")
                    st.json(sync_res)
                    st.rerun()

        with col_act2:
            st.info(
                "💡 **Cơ chế Tiết kiệm Tài nguyên (Incremental Hashing):** "
                "Hệ thống tự động so khớp mã băm SHA-256 của từng thông báo. Nếu nội dung không thay đổi, "
                "hệ thống sẽ lập tức bỏ qua (Skip) để tránh tính toán lại vector embedding."
            )

        # Metrics Dashboard
        st.markdown("#### 📈 **Chỉ số Vận hành Hệ thống Tri thức**")
        stats = vector_db_service.get_stats()
        raw_portal_docs = fetch_portal_announcements()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Tổng số Vector Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(raw_portal_docs)}</div>
                <div class="metric-label">Văn bản thông báo gốc</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{stats['total_syncs']}</div>
                <div class="metric-label">Số chu kỳ đồng bộ</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">30m</div>
                <div class="metric-label">Chu kỳ quét APScheduler</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Document Inventory Table
        st.markdown("#### 📑 **Danh mục Văn bản Đã Lập Chỉ mục (Document Inventory)**")
        inventory_rows = []
        for doc in raw_portal_docs:
            exp_d = parse_date(doc.get("expiration_date", ""))
            is_valid = (not exp_d) or (exp_d >= date.today())
            stored_hash = vector_db_service.get_document_stored_hash(doc["id"]) or "Chưa lập chỉ mục"

            inventory_rows.append({
                "Mã số (ID)": doc["id"],
                "Tiêu đề văn bản": doc["title"],
                "Chuyên mục": doc.get("category", "Chung"),
                "Ngày ban hành": doc.get("publication_date", ""),
                "Hạn hiệu lực": doc.get("expiration_date", "Vô hạn"),
                "Mã băm SHA-256": stored_hash[:16] + "..." if len(stored_hash) > 16 else stored_hash,
                "Trạng thái": "🟢 Hoạt động" if is_valid else "🔴 Hết hạn"
            })
        st.dataframe(pd.DataFrame(inventory_rows), use_container_width=True)

        # Sync Logs Table
        st.markdown("#### 📜 **Nhật ký Lịch sử Đồng bộ (Sync Audit Logs)**")
        if vector_db_service.sync_logs:
            log_table = []
            for item in vector_db_service.sync_logs:
                log_table.append({
                    "Thời điểm": item["timestamp"][:19].replace("T", " "),
                    "Số văn bản quét": item["total_fetched"],
                    "Thêm mới": item["new"],
                    "Sửa đổi": item["modified"],
                    "Bỏ qua (Trùng SHA-256)": item["skipped"],
                    "Lỗi": item["errors"]
                })
            st.table(log_table)
        else:
            st.write("Chưa có lịch sử đồng bộ nào.")


# =========================================================================
# TAB 4: ARCHITECTURE & MIDTERM REPORT SPECS
# =========================================================================
with tab_architecture:
    st.markdown("### 🏛️ **Đặc tả Kiến trúc Hệ thống & Báo cáo Giữa kỳ**")
    st.markdown("""
    Hệ thống được thiết kế và triển khai đáp ứng chính xác các yêu cầu trong **Báo cáo Giữa kỳ (Midterm Report)** 
    môn học **Advanced Topics in Information Technology (ATI)** - Khoa Công nghệ Thông tin, Đại học Hà Nội (HANU).

    #### 1. Thành phần Công nghệ Đề xuất (Proposed Technology Stack)
    - **Frontend:** Streamlit hỗ trợ giao diện hội thoại đa phương tiện và phòng thí nghiệm kiểm thử thuật toán.
    - **Backend Framework:** FastAPI (RESTful API) và APScheduler (Tác vụ ngầm đồng bộ định kỳ 15-30 phút).
    - **Mô hình Ngôn ngữ Lớn (LLM):** Google Gemini 3.5 Flash Lite API (Grounding strictly on context).
    - **Cơ sở dữ liệu Vector:** ChromaDB (Lưu trữ persistent trên đĩa, độ đo Cosine Distance).
    - **Trích xuất & Xử lý Văn bản:** BeautifulSoup (xử lý bảng biểu HTML) & PyMuPDF (xử lý file đính kèm PDF).

    #### 2. Quy trình 3 Pha Lọc Thời gian (Temporal-Aware Retrieval)
    ```
    ┌──────────────────────┐      Top-20 Chunks      ┌──────────────────────┐
    │   1. Semantic Search  │ ─────────────────────► │  2. Validity Filter  │
    │ (Cosine Sim ChromaDB)│                        │ (exp_date >= today)  │
    └──────────────────────┘                        └──────────┬───────────┘
                                                               │ Remaining Valid
                                                               ▼
    ┌──────────────────────┐      Top-5 Context     ┌──────────────────────┐
    │   4. Prompt Grounding│ ◄───────────────────── │ 3. Recency Re-ranking│
    │  & Gemini Synthesis  │                        │ (Freshness Boost)    │
    └──────────────────────┘                        └──────────────────────┘
    ```

    #### 3. Thông tin Nhóm Sinh viên Thực hiện
    1. **Đinh Thị Hải** — Mã sinh viên: `2301040058`
    2. **Đinh Thị Khuê** — Mã sinh viên: `2301040092`
    3. **Nguyễn Minh Nguyệt** — Mã sinh viên: `2301040142`
    - **Giảng viên hướng dẫn:** Thầy Bùi Minh Đức
    """)
