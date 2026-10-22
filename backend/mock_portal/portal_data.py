"""
Mock Academic Management Portal Data for Hanoi University (HANU).
Simulates JSON payloads returned by the university's official API endpoint.
"""

from typing import List, Dict, Any

MOCK_ANNOUNCEMENTS: List[Dict[str, Any]] = [
    {
        "id": "hanu-ann-2026-001",
        "title": "Thông báo thu học phí Học kỳ 1 - Năm học 2026-2027",
        "category": "Tuition",
        "publication_date": "2026-08-15",
        "expiration_date": "2026-12-31",
        "source_url": "https://qldt.hanu.edu.vn/announcements/2026-001",
        "content_html": """
        <div>
            <h2>Quy định mức học phí Học kỳ 1 năm học 2026-2027</h2>
            <p>Trường Đại học Hà Nội (Hanoi University - HANU) thông báo mức thu học phí cho sinh viên chính quy Học kỳ 1 năm học 2026-2027 như sau:</p>
            <table>
                <tr><th>Chương trình đào tạo</th><th>Mức học phí / tín chỉ</th></tr>
                <tr><td>Chương trình chuẩn (Khoa CNTT và các khoa khác)</td><td>750.000 VNĐ / tín chỉ</td></tr>
                <tr><td>Chương trình chất lượng cao / tiên tiến</td><td>1.250.000 VNĐ / tín chỉ</td></tr>
            </table>
            <p><strong>Thời hạn đóng học phí:</strong> Từ ngày 01/09/2026 đến hết ngày 30/10/2026.</p>
            <p><strong>Hình thức nộp:</strong> Chuyển khoản qua cổng thanh toán VNPAY trên cổng Quản lý đào tạo hoặc nộp trực tiếp tại Ngân hàng BIDV theo số tài khoản nhà trường.</p>
            <p><strong>Lưu ý:</strong> Sinh viên không hoàn thành học phí đúng hạn sẽ bị khóa quyền đăng ký thi kết thúc học phần và hủy đăng ký môn học kỳ tiếp theo.</p>
        </div>
        """,
    },
    {
        "id": "hanu-ann-2025-099",
        "title": "Thông báo thu học phí Học kỳ 1 - Năm học 2025-2026 (Đã hết hạn)",
        "category": "Tuition",
        "publication_date": "2025-08-10",
        "expiration_date": "2025-12-31",
        "source_url": "https://qldt.hanu.edu.vn/announcements/2025-099",
        "content_html": """
        <div>
            <h2>Quy định mức học phí Học kỳ 1 năm học 2025-2026</h2>
            <p>Mức thu học phí năm học trước 2025-2026 là 650.000 VNĐ / tín chỉ đối với chương trình chuẩn.</p>
            <p>Hạn đóng: 15/10/2025. Thông báo này đã hết hiệu lực.</p>
        </div>
        """,
    },
    {
        "id": "hanu-ann-2026-002",
        "title": "Quy định Chuẩn đầu ra và Điều kiện tốt nghiệp ngành Công nghệ Thông tin (Khóa 2023 - 2027)",
        "category": "Graduation",
        "publication_date": "2026-01-10",
        "expiration_date": "2027-12-31",
        "source_url": "https://fit.hanu.edu.vn/graduation-requirements-cs",
        "content_html": """
        <div>
            <h2>Điều kiện xét tốt nghiệp Cử nhân Công nghệ Thông tin - Trường Đại học Hà Nội</h2>
            <p>Sinh viên Khoa Công nghệ Thông tin (FIT-HANU) đủ điều kiện công nhận tốt nghiệp khi đáp ứng đầy đủ các tiêu chuẩn sau:</p>
            <ul>
                <li><strong>Tổng số tín chỉ tích lũy:</strong> Tối thiểu 135 tín chỉ theo đúng khung chương trình đào tạo của khóa học.</li>
                <li><strong>Điểm trung bình chung tích lũy (GPA):</strong> Đạt từ 2.00 / 4.0 trở lên.</li>
                <li><strong>Chuẩn ngoại ngữ đầu ra:</strong> Chứng chỉ IELTS Academic tối thiểu 6.0 hoặc chứng chỉ bậc 4 theo Khung năng lực ngoại ngữ 6 bậc dùng cho Việt Nam (VSTEP B2).</li>
                <li><strong>Chứng chỉ Giáo dục thể chất và Giáo dục quốc phòng - an ninh:</strong> Đã hoàn thành và có chứng chỉ hợp lệ.</li>
                <li><strong>Khóa luận tốt nghiệp hoặc Đồ án chuyên ngành:</strong> Đạt điểm từ 5.5 / 10 trở lên hoặc hoàn thành các học phần thay thế khóa luận theo quy định.</li>
                <li>Không bị kỷ luật từ mức đình chỉ học tập trở lên và không bị truy cứu trách nhiệm hình sự tại thời điểm xét tốt nghiệp.</li>
            </ul>
        </div>
        """,
    },
    {
        "id": "hanu-ann-2026-003",
        "title": "Hướng dẫn quy trình đăng ký học lại và thi lại học phần",
        "category": "Regulations",
        "publication_date": "2026-09-01",
        "expiration_date": "2027-06-30",
        "source_url": "https://qldt.hanu.edu.vn/guide/retake-courses",
        "content_html": """
        <div>
            <h2>Quy trình đăng ký học lại và thi lại học phần dành cho sinh viên HANU</h2>
            <p>Dành cho sinh viên có điểm tổng kết học phần dưới F (dưới 4.0/10) hoặc sinh viên muốn đăng ký học cải thiện điểm:</p>
            <ol>
                <li><strong>Bước 1 - Kiểm tra lịch đăng ký:</strong> Theo dõi thời gian mở cổng đăng ký trên website QLĐT (thường diễn ra trong 2 tuần đầu mỗi học kỳ).</li>
                <li><strong>Bước 2 - Đăng ký trên cổng trực tuyến:</strong> Đăng nhập hệ thống Quản lý đào tạo (https://qldt.hanu.edu.vn), vào mục 'Đăng ký học vụ' -> chọn 'Đăng ký học lại / học cải thiện', chọn lớp học phần phù hợp với thời khóa biểu.</li>
                <li><strong>Bước 3 - Nộp học phí học lại:</strong> Học phí học lại tính theo số tín chỉ của học phần đó (mức phí bằng mức học phí tín chỉ hiện hành). Sinh viên nộp trong vòng 7 ngày kể từ khi đăng ký thành công.</li>
                <li><strong>Bước 4 - Điều kiện dự thi lại:</strong> Đối với thi lại, sinh viên phải đạt điều kiện điểm chuyên cần tối thiểu 80% số tiết trên lớp. Đăng ký thi lại tại Văn phòng Khoa trong vòng 1 tuần sau khi công bố điểm thi lần 1.</li>
            </ol>
        </div>
        """,
    },
    {
        "id": "hanu-ann-2026-004",
        "title": "Khung chương trình đào tạo Cử nhân ngành Công nghệ Thông tin",
        "category": "Curriculum",
        "publication_date": "2026-03-01",
        "expiration_date": "2028-12-31",
        "source_url": "https://fit.hanu.edu.vn/curriculum-cs",
        "content_html": """
        <div>
            <h2>Khung chương trình đào tạo Cử nhân ngành Công nghệ Thông tin (HANU FIT)</h2>
            <p>Chương trình đào tạo gồm 135 tín chỉ phân bổ trong 4 năm học (8 học kỳ chính):</p>
            <p><strong>Khối kiến thức cơ sở ngành:</strong> Cơ sở lập trình (C/C++), Lập trình hướng đối tượng (OOP với Java/Python), Cấu trúc dữ liệu và giải thuật, Cơ sở dữ liệu (SQL), Kiến trúc máy tính và Hệ điều hành.</p>
            <p><strong>Khối kiến thức chuyên ngành:</strong> Mạng máy tính và An toàn thông tin, Phát triển ứng dụng Web, Kỹ thuật phần mềm, Trí tuệ nhân tạo (AI), Học máy (Machine Learning), Các chuyên đề nâng cao về CNTT (Advanced Topics in IT - ATI).</p>
            <p><strong>Đồ án & Thực tập:</strong> Thực tập tốt nghiệp doanh nghiệp (3 tín chỉ), Khóa luận tốt nghiệp hoặc Đồ án chuyên ngành (7 tín chỉ).</p>
            <p>Sinh viên có thể xem chi tiết đề cương chi tiết môn học trên cổng đào tạo HANU FIT hoặc liên hệ trợ lý giáo vụ khoa FIT tại Phòng 201 nhà C.</p>
        </div>
        """,
    },
    {
        "id": "hanu-ann-2026-005",
        "title": "Thông báo tạm dừng hoạt động bảo trì phòng máy nhà C (Đã hết hạn)",
        "category": "Facility",
        "publication_date": "2026-09-05",
        "expiration_date": "2026-09-12",
        "source_url": "https://fit.hanu.edu.vn/maintenance-lab-c",
        "content_html": """
        <div>
            <h2>Tạm dừng phòng máy nhà C để nâng cấp đường truyền</h2>
            <p>Các phòng thực hành CNTT tại nhà C sẽ tạm đóng cửa từ 06/09/2026 đến 12/09/2026.</p>
            <p>Thông báo này chỉ áp dụng trong khoảng thời gian trên.</p>
        </div>
        """,
    }
]


def fetch_portal_announcements() -> List[Dict[str, Any]]:
    """
    Simulates calling the University Academic Management HTTP API.
    Returns the list of raw announcement payloads.
    """
    return MOCK_ANNOUNCEMENTS
