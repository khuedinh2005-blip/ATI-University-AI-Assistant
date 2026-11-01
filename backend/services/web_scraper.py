"""
Web Scraper Service for Hanoi University (HANU).
Crawls public notices, academic announcements, and departmental updates from:
- HANU Official Portal (https://hanu.edu.vn/thong-bao-nha-truong, https://hanu.edu.vn/vi/thong-bao)
- Faculty of Information Technology (https://fit.hanu.edu.vn)
- Academic Management System (https://qldt.hanu.edu.vn)

All documents are parsed with BeautifulSoup, categorized, assigned deterministic
SHA-256 IDs, and returned in the standardized announcement schema for ingestion.
"""

import hashlib
import logging
import re
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Standard browser headers to avoid automated request blocks
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

REQUEST_TIMEOUT = 10


def generate_doc_id(source_url: str) -> str:
    """
    Generates a deterministic document ID from the SHA-256 hash of the source URL.
    Format: 'hanu-web-' + sha256(url)[:12]
    """
    normalized_url = source_url.strip().lower()
    url_hash = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:12]
    return f"hanu-web-{url_hash}"


def detect_category(title: str, text: str = "") -> str:
    """
    Detects announcement category based on title and content keywords.
    Categories: 'Tuition', 'Graduation', 'Regulations', 'Curriculum',
                'Scholarship', 'Admissions', 'Facility', or 'Announcement'.
    """
    combined = f"{title} {text}".lower()

    if any(k in combined for k in ["học phí", "hoc phi", "lệ phí", "miễn giảm", "kinh phí", "tiền học", "nộp tiền"]):
        return "Tuition"
    if any(k in combined for k in ["tốt nghiệp", "xét tốt nghiệp", "chuẩn đầu ra", "bằng tốt nghiệp", "đồ án", "khóa luận"]):
        return "Graduation"
    if any(k in combined for k in ["quy chế", "quy định", "quy trình", "hướng dẫn", "nội quy", "điều lệ", "thủ tục"]):
        return "Regulations"
    if any(k in combined for k in ["chương trình", "đào tạo", "môn học", "học phần", "thời khóa biểu", "lịch học", "lịch thi", "hoãn thi", "thi lại", "học lại", "đăng ký tín chỉ"]):
        return "Curriculum"
    if any(k in combined for k in ["học bổng", "khen thưởng", "tài trợ", "vinh danh"]):
        return "Scholarship"
    if any(k in combined for k in ["tuyển sinh", "xét tuyển", "nhập học", "chỉ tiêu"]):
        return "Admissions"
    if any(k in combined for k in ["cơ sở vật chất", "phòng máy", "thư viện", "ký túc xá", "bảo trì"]):
        return "Facility"

    return "Announcement"


def extract_date(text: str) -> str:
    """
    Extracts publication date from text using regex patterns.
    Formats supported:
    - YYYY-MM-DD
    - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    - ngày DD tháng MM năm YYYY
    Falls back to current date (YYYY-MM-DD) if no valid date is found.
    """
    if not text:
        return date.today().isoformat()

    # Pattern 1: YYYY-MM-DD
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        year, month, day = match.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            pass

    # Pattern 2: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    match = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text)
    if match:
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            pass

    # Pattern 3: ngày DD tháng MM năm YYYY
    match = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text, re.IGNORECASE)
    if match:
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            pass

    return date.today().isoformat()


def _safe_get_soup(url: str) -> Optional[BeautifulSoup]:
    """
    Sends an HTTP GET request with timeout, handles encoding,
    and returns a BeautifulSoup instance. Returns None on failure.
    """
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None

        # Ensure correct UTF-8 decoding for Vietnamese text
        if response.encoding and response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        elif not response.encoding:
            response.encoding = "utf-8"

        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        logger.warning(f"Connection error while requesting {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error while processing {url}: {e}")
        return None


def scrape_hanu_main_announcements() -> List[Dict[str, Any]]:
    """
    Scrapes announcements from Hanoi University official website:
    Checks https://hanu.edu.vn/thong-bao-nha-truong and https://hanu.edu.vn/vi/thong-bao.
    Returns: List of standardized announcement dicts.
    """
    announcements: List[Dict[str, Any]] = []
    target_urls = [
        "https://hanu.edu.vn/thong-bao-nha-truong",
        "https://hanu.edu.vn/vi/thong-bao"
    ]

    for page_url in target_urls:
        soup = _safe_get_soup(page_url)
        if not soup:
            continue

        # Look for article blocks or links in announcement lists
        article_elements = soup.find_all(["article", "div", "li"], class_=re.compile(r"item|news|post|thong-bao", re.I))
        found_on_page = 0

        for el in article_elements:
            link_tag = el.find("a", href=True)
            if not link_tag:
                continue

            title = link_tag.get_text(strip=True)
            if not title or len(title) < 15:
                # Try finding a heading inside
                heading = el.find(["h2", "h3", "h4"])
                if heading:
                    title = heading.get_text(strip=True)

            if not title or len(title) < 15:
                continue

            raw_href = link_tag["href"]
            source_url = urljoin(page_url, raw_href)

            # Extract date if available in element
            date_text = ""
            date_tag = el.find(["time", "span", "div"], class_=re.compile(r"date|time|created", re.I))
            if date_tag:
                date_text = date_tag.get_text(strip=True)
            else:
                date_text = el.get_text()

            pub_date = extract_date(date_text)
            category = detect_category(title)

            # Build content summary
            snippet = ""
            desc_tag = el.find(["p", "div"], class_=re.compile(r"desc|summary|intro|teaser", re.I))
            if desc_tag:
                snippet = desc_tag.get_text(strip=True)
            else:
                # Fallback to general text inside element excluding title
                snippet = re.sub(r"\s+", " ", el.get_text(strip=True))[:300]

            content_html = f"""
            <div class="announcement-item">
                <h2>{title}</h2>
                <p><strong>Ngày ban hành:</strong> {pub_date}</p>
                <p><strong>Danh mục:</strong> {category}</p>
                <div class="announcement-body">
                    <p>{snippet}</p>
                </div>
                <p><a href="{source_url}">Xem chi tiết thông báo gốc tại HANU</a></p>
            </div>
            """.strip()

            doc_id = generate_doc_id(source_url)
            announcements.append({
                "id": doc_id,
                "title": title,
                "category": category,
                "publication_date": pub_date,
                "expiration_date": "",
                "source_url": source_url,
                "content_html": content_html,
            })
            found_on_page += 1
            if found_on_page >= 15:
                break

        if announcements:
            # If we successfully retrieved items from the first target, don't overwhelm
            break

    return announcements


def scrape_fit_announcements() -> List[Dict[str, Any]]:
    """
    Scrapes announcements from the Faculty of Information Technology (FIT - HANU).
    Target: https://fit.hanu.edu.vn and https://fit.hanu.edu.vn/thong-bao
    Returns: List of standardized announcement dicts.
    """
    announcements: List[Dict[str, Any]] = []
    target_urls = [
        "https://fit.hanu.edu.vn",
        "https://fit.hanu.edu.vn/thong-bao"
    ]

    for base_url in target_urls:
        soup = _safe_get_soup(base_url)
        if not soup:
            continue

        items = soup.find_all(["article", "div"], class_=re.compile(r"post|item|entry|thongbao", re.I))
        # If no items found by class, check general post links
        if not items:
            items = soup.find_all("h3")

        found_count = 0
        for item in items:
            link = item.find("a", href=True) if item.name != "a" else item
            if not link:
                continue

            title = link.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            source_url = urljoin(base_url, link["href"])
            date_text = item.get_text()
            pub_date = extract_date(date_text)
            category = detect_category(title)

            # Extract snippet or summary
            snippet = ""
            desc = item.find(["p", "div"], class_=re.compile(r"summary|excerpt|content", re.I))
            if desc:
                snippet = desc.get_text(strip=True)
            else:
                snippet = re.sub(r"\s+", " ", item.get_text(strip=True))[:300]

            content_html = f"""
            <div class="fit-announcement">
                <h2>{title}</h2>
                <p><strong>Đơn vị:</strong> Khoa Công nghệ Thông tin (FIT - HANU)</p>
                <p><strong>Ngày ban hành:</strong> {pub_date}</p>
                <p><strong>Danh mục:</strong> {category}</p>
                <div class="body-content">
                    <p>{snippet}</p>
                </div>
                <p><a href="{source_url}">Xem chi tiết thông báo trên Khoa CNTT</a></p>
            </div>
            """.strip()

            announcements.append({
                "id": generate_doc_id(source_url),
                "title": title,
                "category": category,
                "publication_date": pub_date,
                "expiration_date": "",
                "source_url": source_url,
                "content_html": content_html,
            })
            found_count += 1
            if found_count >= 15:
                break

        if announcements:
            break

    return announcements


def scrape_qldt_announcements() -> List[Dict[str, Any]]:
    """
    Scrapes academic management announcements from https://qldt.hanu.edu.vn.
    Publicly accessible notices (grades, registration, tuition schedules).
    Returns: List of standardized announcement dicts.
    """
    announcements: List[Dict[str, Any]] = []
    base_url = "https://qldt.hanu.edu.vn"

    soup = _safe_get_soup(base_url)
    if not soup:
        return announcements

    # Check for announcement tables or list groups on QLĐT portal
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            link = row.find("a", href=True)
            if not link:
                continue

            title = link.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            source_url = urljoin(base_url, link["href"])
            row_text = row.get_text()
            pub_date = extract_date(row_text)
            category = detect_category(title)

            content_html = f"""
            <div class="qldt-announcement">
                <h2>{title}</h2>
                <p><strong>Cổng Quản lý Đào tạo (QLĐT - HANU)</strong></p>
                <p><strong>Ngày đăng:</strong> {pub_date}</p>
                <p><strong>Danh mục:</strong> {category}</p>
                <p>Nội dung chi tiết thông báo quản lý đào tạo được niêm yết tại cổng chính thức.</p>
                <p><a href="{source_url}">Xem thông báo gốc trên QLĐT HANU</a></p>
            </div>
            """.strip()

            announcements.append({
                "id": generate_doc_id(source_url),
                "title": title,
                "category": category,
                "publication_date": pub_date,
                "expiration_date": "",
                "source_url": source_url,
                "content_html": content_html,
            })
            if len(announcements) >= 15:
                break
        if announcements:
            break

    # If no tables found, check anchor links on homepage
    if not announcements:
        for a_tag in soup.find_all("a", href=True):
            title = a_tag.get_text(strip=True)
            if not title or len(title) < 20:
                continue

            href = a_tag["href"]
            if any(ext in href.lower() for ext in [".css", ".js", "javascript:", "#"]):
                continue

            source_url = urljoin(base_url, href)
            pub_date = extract_date(a_tag.parent.get_text() if a_tag.parent else "")
            category = detect_category(title)

            announcements.append({
                "id": generate_doc_id(source_url),
                "title": title,
                "category": category,
                "publication_date": pub_date,
                "expiration_date": "",
                "source_url": source_url,
                "content_html": f"<div><h2>{title}</h2><p>Thông báo từ Cổng Đào tạo HANU.</p><p><a href='{source_url}'>Xem chi tiết</a></p></div>",
            })
            if len(announcements) >= 10:
                break

    return announcements


def fetch_web_announcements() -> List[Dict[str, Any]]:
    """
    Master web scraper aggregator:
    - Scrapes HANU main website, FIT faculty portal, and QLĐT academic portal.
    - Collects all announcements into a unified list.
    - Deduplicates announcements by source_url.
    - Returns standardized list of document dictionaries.
    """
    logger.info("Starting web scraping for Hanoi University public portals...")
    all_announcements: List[Dict[str, Any]] = []

    # 1. Main HANU announcements
    try:
        hanu_items = scrape_hanu_main_announcements()
        logger.info(f"Scraped {len(hanu_items)} items from HANU main portal.")
        all_announcements.extend(hanu_items)
    except Exception as e:
        logger.warning(f"Error during HANU main portal scraping: {e}")

    # 2. FIT HANU announcements
    try:
        fit_items = scrape_fit_announcements()
        logger.info(f"Scraped {len(fit_items)} items from FIT portal.")
        all_announcements.extend(fit_items)
    except Exception as e:
        logger.warning(f"Error during FIT portal scraping: {e}")

    # 3. QLĐT HANU announcements
    try:
        qldt_items = scrape_qldt_announcements()
        logger.info(f"Scraped {len(qldt_items)} items from QLĐT portal.")
        all_announcements.extend(qldt_items)
    except Exception as e:
        logger.warning(f"Error during QLĐT portal scraping: {e}")

    # Deduplicate strictly by source_url
    deduplicated: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    for item in all_announcements:
        url = item.get("source_url", "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduplicated.append(item)

    logger.info(f"Web scraping complete: {len(deduplicated)} unique announcements collected.")
    return deduplicated
