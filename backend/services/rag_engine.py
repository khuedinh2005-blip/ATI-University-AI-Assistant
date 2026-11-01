"""
RAG Engine Service.
Builds grounded prompts with retrieved context chunks, metadata, and timestamps,
and invokes Gemini Flash to generate accurate answers with source citations.
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
6. Synthesize the relevant information into a direct answer. Do not mention retrieval,
   chunk counts, similarity scores, vector databases, prompts, models, API keys, or
   internal system steps.
7. Prefer a short structure: answer first, then important deadlines/steps/conditions
   as bullet points, and finish with a brief source note when useful.
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


def _build_extractive_fallback(chunks: List[Dict[str, Any]]) -> str:
    """Provide a useful grounded response when the generation service is unavailable."""
    sections = []
    seen_titles = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Thông báo chính thức")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        publication_date = metadata.get("publication_date", "không rõ ngày")
        text = " ".join(chunk.get("text", "").split())
        if len(text) > 650:
            text = f"{text[:650].rsplit(' ', 1)[0]}..."
        sections.append(f"**{title}** (ban hành {publication_date})\n{text}")

    if not sections:
        return "Hiện chưa có thông tin chính thức phù hợp để trả lời câu hỏi này."

    return (
        "Dưới đây là thông tin phù hợp được tổng hợp từ các thông báo chính thức:\n\n"
        + "\n\n".join(sections)
        + "\n\nBạn nên đối chiếu thời hạn và hướng dẫn trên nguồn chính thức trước khi thực hiện."
    )


def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a grounded response using Gemini Flash.
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

    # Gemini keys do not all share one prefix. Only reject empty/template values.
    is_placeholder_key = not key or key.strip() == "" or key.strip() == "your_gemini_api_key_here"

    if is_placeholder_key:
        simulated_answer = _build_extractive_fallback(chunks)
        return {
            "answer": simulated_answer,
            "sources": sources,
            "is_simulated": True
        }

    try:
        client = genai.Client(api_key=key)
        prompt = build_rag_prompt(question, chunks)

        response = None
        for model_candidate in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_candidate,
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": 0.2
                    }
                )
                break
            except Exception:
                continue

        if response and response.text:
            return {
                "answer": response.text,
                "sources": sources,
                "is_simulated": False
            }
        else:
            raise RuntimeError("Không nhận được phản hồi từ các mô hình Gemini.")

    except Exception as e:
        fallback_answer = (
            _build_extractive_fallback(chunks)
        )
        return {
            "answer": fallback_answer,
            "sources": sources,
            "is_simulated": True,
            "error": str(e)
        }
