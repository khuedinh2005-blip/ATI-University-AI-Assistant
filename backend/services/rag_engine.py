"""
RAG Engine Service.
Builds grounded prompts with retrieved context chunks, metadata, and timestamps,
and invokes Gemini 2.5 Flash to generate accurate answers with source citations.
"""

from typing import List, Dict, Any, Optional
from google import genai
from backend.config import settings


SYSTEM_PROMPT = """You are the official Hanoi University AI Assistant (Trợ lý AI Trường Đại học Hà Nội - HANU).
Your mission is to help students and lecturers access verified, official academic regulations, tuition deadlines, graduation standards, and notices.

STRICT INSTRUCTIONS:
1. Ground your answers strictly on the provided "OFFICIAL UNIVERSITY CONTEXT" below.
2. If the context does not contain the answer, explicitly state: "Hiện tại hệ thống không tìm thấy thông tin chính thức về câu hỏi này trong cơ sở dữ liệu các thông báo của trường. Vui lòng liên hệ Văn phòng Khoa CNTT (Phòng 201 Nhà C) hoặc Cổng QLĐT để được hỗ trợ."
3. Do not invent or assume dates, tuition fees, or rules.
4. When stating facts, explicitly reference the document title and publication date.
5. Answer in the same language as the user's question (Vietnamese or English), keeping a polite, helpful academic tone.
"""


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds the context-augmented prompt for Gemini."""
    context_sections = []
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        title = meta.get("title", "Official Announcement")
        pub_date = meta.get("publication_date", "N/A")
        source_url = meta.get("source_url", "")
        text = c.get("text", "")

        context_sections.append(
            f"--- [DOCUMENT {idx}]: {title} (Published: {pub_date}) ---\n"
            f"Source URL: {source_url}\n"
            f"Content:\n{text}\n"
        )

    joined_context = "\n".join(context_sections)

    prompt = f"""OFFICIAL UNIVERSITY CONTEXT:
{joined_context}

---
USER QUESTION:
{question}

Provide an accurate, grounded response referencing the relevant documents and dates from the official context above:"""
    return prompt


def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a grounded response using Gemini 2.5 Flash.
    Returns:
    - answer: LLM generated text
    - sources: list of source document references
    """
    key = api_key or settings.GEMINI_API_KEY

    # Build list of distinct sources
    seen_docs = set()
    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "doc_id": doc_id,
                "title": meta.get("title", "Document"),
                "category": meta.get("category", "General"),
                "publication_date": meta.get("publication_date", ""),
                "expiration_date": meta.get("expiration_date", ""),
                "source_url": meta.get("source_url", ""),
                "similarity_score": c.get("similarity", 0.0),
                "combined_score": c.get("combined_score", 0.0)
            })

    if not key or key.strip() == "" or key == "your_gemini_api_key_here":
        # Helpful offline demonstration fallback when API key has not been configured yet
        sample_context_preview = "\n\n".join([f"• **{s['title']}** (Ngày ban hành: {s['publication_date']}):\n{c['text'][:250]}..." for s, c in zip(sources[:2], chunks[:2])])
        simulated_answer = (
            f"*(Hệ thống đã truy xuất thành công {len(chunks)} đoạn trích phù hợp từ cơ sở dữ liệu)*\n\n"
            f"**Tóm tắt thông tin chính thức tìm thấy:**\n\n"
            f"{sample_context_preview}\n\n"
            f"> 💡 **Lưu ý cho nhà phát triển:** Để kích hoạt phản hồi tổng hợp thời gian thực từ mô hình **Gemini 2.5 Flash**, "
            f"vui lòng thêm `GEMINI_API_KEY` của bạn vào file `.env` hoặc nhập trực tiếp trên thanh điều khiển."
        )
        return {
            "answer": simulated_answer,
            "sources": sources,
            "is_simulated": True
        }

    try:
        client = genai.Client(api_key=key)
        prompt = build_rag_prompt(question, chunks)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.2
            }
        )
        return {
            "answer": response.text,
            "sources": sources,
            "is_simulated": False
        }
    except Exception as e:
        # Fallback with graceful error explanation
        return {
            "answer": f"Lỗi khi kết nối với Gemini API: {str(e)}. Vui lòng kiểm tra lại GEMINI_API_KEY và kết nối mạng.",
            "sources": sources,
            "is_simulated": False,
            "error": str(e)
        }
