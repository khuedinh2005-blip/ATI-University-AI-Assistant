"""
Streamlit Web Application for Hanoi University AI Assistant.
Provides:
1. Student Chat Interface (Questions, conversational history, source citations).
2. Admin Control Panel (Protected by Admin Key, 'Sync Now' trigger, sync logs, document stats).
"""

import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import requests
from backend.config import settings
from backend.services.vector_db import vector_db_service
from backend.services.temporal_ranker import rank_and_filter_chunks, parse_date
from backend.services.rag_engine import generate_answer

# Page Configuration
st.set_page_config(
    page_title="HANU AI Assistant - Đại học Hà Nội",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        color: #8b0000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #8b0000;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-valid {
        background-color: #e6f4ea;
        color: #137333;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-expired {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là Trợ lý AI của Trường Đại học Hà Nội (HANU). Tôi có thể giải đáp các thắc mắc của bạn về quy chế đào tạo, mức thu học phí, điều kiện tốt nghiệp, quy trình đăng ký học lại/thi lại dựa trên các văn bản thông báo chính thức.",
            "sources": []
        }
    ]

if "api_key_override" not in st.session_state:
    st.session_state.api_key_override = settings.GEMINI_API_KEY


# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Hanoi_University_logo.svg/330px-Hanoi_University_logo.svg.png", width=90)
    st.markdown("### 🎓 **HANU AI Assistant**")
    st.caption("Khoa Công nghệ Thông tin - FIT HANU")
    st.caption("Môn học: *Advanced Topics in IT (ATI)*")
    st.divider()

    # Model Configuration
    st.markdown("#### ⚙️ Cấu hình Mô hình")
    user_gemini_key = st.text_input(
        "Gemini API Key:",
        value=st.session_state.api_key_override,
        type="password",
        help="Khóa API Gemini (lấy miễn phí tại Google AI Studio: https://aistudio.google.com/)"
    )
    if user_gemini_key != st.session_state.api_key_override:
        st.session_state.api_key_override = user_gemini_key

    # Temporal Settings
    st.markdown("#### ⏳ Bộ lọc Thời gian (Temporal RAG)")
    simulated_date = st.date_input(
        "Ngày tham chiếu (Reference Date):",
        value=date.today(),
        help="Hệ thống tự động loại bỏ các thông báo có hạn kết thúc (expiration_date) trước ngày này."
    )

    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Làm mới cuộc trò chuyện (New Chat)", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Cuộc trò chuyện đã được làm mới. Bạn muốn hỏi về nội dung nào hôm nay?",
                "sources": []
            }
        ]
        st.rerun()


# Main View: Tabs for Student Chat & Admin Panel
tab_chat, tab_admin, tab_about = st.tabs(["💬 Sinh viên hỏi đáp (Chat)", "🛡️ Quản trị viên (Admin Panel)", "ℹ️ Giới thiệu Kiến trúc"])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    st.markdown("<h2 class='main-title'>Trợ lý AI Trường Đại học Hà Nội</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Hệ thống hỏi đáp thông tin đào tạo & văn bản thông báo chính thức (Temporal-Aware RAG)</p>", unsafe_allow_html=True)

    # Prompt Suggestions
    st.markdown("**Gợi ý câu hỏi phổ biến:**")
    col1, col2, col3 = st.columns(3)
    suggestion = None
    with col1:
        if st.button("💰 Học phí Kỳ 1 năm nay là bao nhiêu?", use_container_width=True):
            suggestion = "Học phí Học kỳ 1 năm học 2026-2027 là bao nhiêu tiền một tín chỉ và hạn đóng là khi nào?"
    with col2:
        if st.button("🎓 Điều kiện tốt nghiệp ngành CNTT?", use_container_width=True):
            suggestion = "Chuẩn đầu ra và điều kiện để xét tốt nghiệp Cử nhân ngành Công nghệ Thông tin gồm những gì?"
    with col3:
        if st.button("📝 Thủ tục đăng ký học lại / thi lại?", use_container_width=True):
            suggestion = "Các bước để đăng ký học lại và điều kiện dự thi lại môn học như thế nào?"

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Xem tài liệu nguồn tham chiếu", expanded=False):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Nguồn {idx}: {src['title']}</strong><br/>
                            📅 <em>Ngày ban hành:</em> {src['publication_date']} | <em>Độ liên quan:</em> {src['similarity_score']}<br/>
                            🔗 <a href="{src['source_url']}" target="_blank">Xem văn bản gốc trên cổng trường</a>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Nhập câu hỏi của bạn về quy chế, học phí, lịch học...")
    prompt_to_send = suggestion or user_input

    if prompt_to_send:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Assistant Thinking / Generating
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông báo chính thức và tổng hợp câu trả lời..."):
                # Step 1: Query ChromaDB
                raw_chunks = vector_db_service.query_chunks(prompt_to_send, n_results=settings.TOP_K_CHUNKS)

                if not raw_chunks:
                    reply_text = "Cơ sở dữ liệu thông báo đang trống. Vui lòng vào tab Quản trị viên và nhấn 'Đồng bộ ngay (Sync Now)'."
                    sources_to_show = []
                else:
                    # Step 2: Temporal Filter & Re-ranking
                    selected_chunks, expired_chunks, meets_threshold = rank_and_filter_chunks(
                        chunks=raw_chunks,
                        current_date=simulated_date,
                        top_n=settings.TOP_N_RERANKED,
                        threshold=settings.SIMILARITY_THRESHOLD
                    )

                    if not meets_threshold or not selected_chunks:
                        reply_text = (
                            "Hiện tại hệ thống không tìm thấy thông tin chính thức phù hợp với câu hỏi của bạn "
                            "trong cơ sở dữ liệu các thông báo của nhà trường.\n\n"
                            "Vui lòng kiểm tra lại câu hỏi hoặc liên hệ trực tiếp **Văn phòng Khoa CNTT (Phòng 201 Nhà C)** "
                            "hoặc cổng Quản lý đào tạo (https://qldt.hanu.edu.vn) để được hỗ trợ cụ thể."
                        )
                        sources_to_show = []
                    else:
                        # Step 3: Call RAG Engine
                        gen_result = generate_answer(
                            question=prompt_to_send,
                            chunks=selected_chunks,
                            api_key=st.session_state.api_key_override
                        )
                        reply_text = gen_result["answer"]
                        sources_to_show = gen_result["sources"]

                st.markdown(reply_text)
                if sources_to_show:
                    with st.expander("📚 Xem tài liệu nguồn tham chiếu", expanded=True):
                        for idx, src in enumerate(sources_to_show, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Nguồn {idx}: {src['title']}</strong><br/>
                                📅 <em>Ngày ban hành:</em> {src['publication_date']} | <em>Độ liên quan kết hợp:</em> {src.get('combined_score', src['similarity_score'])}<br/>
                                🔗 <a href="{src['source_url']}" target="_blank">Xem văn bản gốc trên cổng trường</a>
                            </div>
                            """, unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "sources": sources_to_show
        })
        st.rerun()


# ----------------- TAB 2: ADMIN PANEL -----------------
with tab_admin:
    st.markdown("### 🛡️ Bảng điều khiển Quản trị viên (Admin Panel)")
    st.info("Bảng điều khiển cho phép theo dõi tình trạng chỉ mục dữ liệu ChromaDB và kích hoạt đồng bộ hóa theo chu kỳ hoặc thủ công.")

    admin_key_input = st.text_input("Nhập Khóa Quản trị (Admin Secret Key):", type="password")

    if admin_key_input != settings.ADMIN_API_KEY:
        st.warning("⚠️ Vui lòng nhập đúng Khóa Quản trị để mở khóa các tính năng quản lý (Mặc định trong file `.env`: `hanu_admin_secret_2026`).")
    else:
        st.success("✅ Xác thực quyền Quản trị viên thành công!")

        # Sync Button & Status
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Đồng bộ ngay (Sync Now)", type="primary", use_container_width=True):
                with st.spinner("Đang kết nối API Cổng đào tạo và tính toán mã băm SHA-256..."):
                    sync_result = vector_db_service.sync_knowledge_base()
                    st.success(f"Đã hoàn thành đồng bộ lúc {sync_result['timestamp'][:19]}!")
                    st.json(sync_result)
                    st.rerun()

        # Database Stats Cards
        stats = vector_db_service.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng số vector chunks đã lưu", stats["total_chunks"])
        c2.metric("Số lần đồng bộ đã ghi nhận", stats["total_syncs"])
        last_sync_time = stats["last_sync"]["timestamp"][:19] if stats.get("last_sync") else "Chưa đồng bộ"
        c3.metric("Lần đồng bộ gần nhất", last_sync_time)

        st.divider()

        # Sync Logs Table
        st.markdown("#### 📜 Lịch sử đồng bộ (Sync Run Logs)")
        if vector_db_service.sync_logs:
            log_data = []
            for log in vector_db_service.sync_logs:
                log_data.append({
                    "Thời gian": log["timestamp"][:19].replace("T", " "),
                    "Tổng tải về": log["total_fetched"],
                    "Tài liệu mới": log["new"],
                    "Đã sửa đổi": log["modified"],
                    "Bỏ qua (Trùng mã băm)": log["skipped"],
                    "Lỗi": log["errors"]
                })
            st.table(log_data)
        else:
            st.write("Chưa có lịch sử đồng bộ nào.")


# ----------------- TAB 3: ABOUT ARCHITECTURE -----------------
with tab_about:
    st.markdown("### 🏛️ Kiến trúc Hệ thống & Báo cáo Giữa kỳ")
    st.markdown("""
    Hệ thống được thiết kế theo đúng kiến trúc đã bảo vệ trong báo cáo giữa kỳ môn **Advanced Topics in IT (ATI)**:
    - **Frontend:** Streamlit hỗ trợ hội thoại trực quan và bảng điều khiển quản trị.
    - **Backend:** FastAPI (Python) phục vụ các endpoint `/chat`, `/sync`, và `/status`.
    - **Cơ chế Temporal-Aware RAG:**
      1. *Semantic Search:* Truy xuất Top-20 chunks bằng Cosine Similarity trong ChromaDB.
      2. *Validity Filtering:* Loại bỏ triệt để các thông báo có `expiration_date < ngày hiện tại` nhằm tránh cung cấp thông tin học phí hoặc lịch thi cũ đã hết hiệu lực.
      3. *Recency Re-ranking:* Tăng trọng số (boost) cho các thông báo mới ban hành gần đây.
      4. *Relevance Threshold:* Kiểm tra ngưỡng tương đồng, trả về thông báo liên hệ khoa nếu câu hỏi không có căn cứ tài liệu.
      5. *Answer Generation:* Prompt chuyên biệt chuyển cho Gemini 2.5 Flash tổng hợp có trích dẫn nguồn.
    """)
