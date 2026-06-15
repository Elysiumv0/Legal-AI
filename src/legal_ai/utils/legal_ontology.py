LEGAL_DOMAIN_KEYWORDS = {
    "labor": [
        "lao động", "sa thải", "nghỉ việc", "bảo hiểm xã hội", "bảo hiểm thất nghiệp",
        "lương", "hợp đồng lao động", "thử việc", "kỷ luật", "quyền lợi", "phụ cấp",
        "nghỉ phép", "an toàn vệ sinh lao động", "việc làm", "người lao động",
        "đình công", "thương lượng tập thể", "công đoàn"
    ],
    "enterprise": [
        "doanh nghiệp", "công ty", "thành lập", "giấy phép", "cổ đông", "hội đồng quản trị",
        "pháp nhân", "giải thể", "sáp nhập", "phân chia", "tổ chức", "đăng ký kinh doanh",
        "trách nhiệm hữu hạn", "cổ phần", "hợp danh", "doanh nghiệp tư nhân",
        "cạnh tranh", "đầu tư", "đấu thầu", "thương mại", "kinh doanh",
        "phá sản", "hỗ trợ doanh nghiệp", "trọng tài thương mại", "hợp đồng thương mại",
        "quảng cáo", "kế toán", "kiểm toán", "tài chính doanh nghiệp", "vốn", "cổ phiếu"
    ],
    "civil": [
        "dân sự", "hợp đồng dân sự", "bồi thường thiệt hại", "thừa kế", "hôn nhân",
        "gia đình", "tài sản", "vật quyền", "trách nhiệm dân sự", "giao dịch dân sự",
        "nhà ở", "đất đai", "xây dựng", "quyền sở hữu", "quyền nhân thân",
        "ủy quyền", "đại diện", "giám hộ", "nuôi con", "ly hôn", "tố tụng dân sự"
    ],
    "tax": [
        "thuế", "hóa đơn", "khấu trừ", "GTGT", "giá trị gia tăng", "TNCN", "thu nhập cá nhân",
        "TNDN", "thu nhập doanh nghiệp", "quản lý thuế", "nộp thuế", "truy thu",
        "miễn thuế", "giảm thuế", "hoàn thuế", "kê khai thuế", "quyết toán thuế",
        "phí", "lệ phí", "thuế suất", "căn cứ tính thuế", "doanh thu", "chi phí được trừ"
    ],
    "administrative": [
        "xử phạt", "xử phạt vi phạm", "xử phạt vi phạm hành chính", "vi phạm hành chính",
        "mức phạt", "phạt tiền", "phạt hành chính", "hình thức xử phạt",
        "thanh tra", "kiểm tra", "giấy phép con",
        "thủ tục hành chính", "cải cách hành chính", "một cửa", "công chức",
        "viên chức", "khiếu nại", "tố cáo", "tiếp công dân", "bồi thường nhà nước",
        "trách nhiệm hành chính", "biện pháp xử lý hành chính", "quản lý nhà nước",
        "thống kê", "báo cáo thống kê", "số liệu thống kê",
        "hàng giả", "hàng cấm", "hàng nhập lậu", "buôn lậu", "gian lận thương mại",
        "bảo vệ quyền lợi người tiêu dùng", "sở hữu công nghiệp", "nhãn hiệu", "sáng chế"
    ],
}

QUERY_TYPE_KEYWORDS = {
    "REGULATION": [
        "quy định", "điều", "khoản", "luật", "nghị định", "thông tư", "như thế nào",
        "ra sao", "được quy định", "căn cứ pháp lý", "văn bản nào"
    ],
    "PRACTICE": [
        "thực tiễn", "tòa án", "kiện", "tranh chấp", "án lệ", "phán quyết", "rủi ro",
        "khả năng thắng", "bồi thường bao nhiêu", "phạt bao nhiêu", "hậu quả pháp lý"
    ],
    "COMPLIANCE": [
        "phải làm gì", "thủ tục", "hồ sơ", "quy trình", "cách", "hướng dẫn", "các bước",
        "điều kiện", "yêu cầu", "trình tự", "thời hạn", "nộp ở đâu"
    ],
}


class LegalOntologyMapper:
    @staticmethod
    def get_domain(query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for domain, keywords in LEGAL_DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return "general"
        return max(scores, key=scores.get)
    
    @staticmethod
    def get_query_type(query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            scores[qtype] = sum(1 for kw in keywords if kw in query_lower)
        if not scores or max(scores.values()) == 0:
            return "REGULATION"
        return max(scores, key=scores.get)
    
    @staticmethod
    def get_domain_from_law_name(law_name: str) -> str:
        law_name_lower = law_name.lower()
        for domain, keywords in LEGAL_DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in law_name_lower:
                    return domain
        return "general"