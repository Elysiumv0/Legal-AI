import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# vLLM là optional — chỉ cần khi dùng QueryExpander / QueryDecomposer
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# ① QUERY ENTITY EXTRACTOR — Regex + Gazetteer (không cần LLM)
# ═══════════════════════════════════════════════════════════════════

# ── Regex patterns ──
RE_LAW_ID = re.compile(
    r'\b(\d{1,4}/\d{4}/(?:NĐ-CP|TT-BTC|TT-BTNMT|TT-BKHCN|'
    r'QH\d+|UBTVQH\d+|CP|TTg|BLĐTBXH|[A-Z0-9\-]+))\b',
    re.IGNORECASE
)
RE_ARTICLE = re.compile(r'(?:Điều|Khoản|Chương|Mục)\s+(\d+[a-zđ]*)', re.IGNORECASE)
RE_QUANTITY = re.compile(r'(\d+\s*(?:phút|giờ|ngày|tháng|năm|tuần|triệu|tỷ|nghìn|%))', re.IGNORECASE)
# NOTE: RE_PENALTY được define 1 lần duy nhất ở dòng ~125, bao gồm
# cả các pattern cấu trúc (từ khối cũ) và pattern ngữ cảnh.
# Xem RE_PENALTY bên dưới.

# ── Bảng chuẩn hóa: có dấu → không dấu ──
_ACCENT_MAP = str.maketrans(
    "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
)

def normalize_vn(text: str) -> str:
    """Chuẩn hóa tiếng Việt: lowercase + bỏ dấu."""
    return text.lower().translate(_ACCENT_MAP)

# ── Gazetteer dictionaries ──
LEGAL_SUBJECTS = [
    "doanh nghiệp nhỏ và vừa", "doanh nghiệp siêu nhỏ", "doanh nghiệp tư nhân",
    "công ty cổ phần", "công ty trách nhiệm hữu hạn", "công ty TNHH",
    "công ty hợp danh", "hộ kinh doanh", "hợp tác xã", "tổ chức tín dụng",
    "thương nhân", "tổ chức xúc tiến thương mại", "nhà đầu tư nước ngoài",
    "người lao động", "người sử dụng lao động", "nhân viên", "lao động nữ",
    "lao động chưa thành niên", "người khuyết tật", "người giúp việc gia đình",
    "chủ doanh nghiệp tư nhân", "chủ sở hữu", "cổ đông", "thành viên góp vốn",
    # === THÊM ALIAS ===
    "công ty nhỏ và vừa",
    "công ty sản xuất nhỏ và vừa",
    "cơ sở ươm tạo", "khu làm việc chung",
    "doanh nghiệp khởi nghiệp sáng tạo",
    "chủ quản nền tảng thương mại điện tử",
    "tổ chức cung cấp dịch vụ",
    "đơn vị phụ thuộc",
    "văn phòng đại diện",
    "chi nhánh",
]

LEGAL_ROLES = [
    "giám đốc", "tổng giám đốc", "hội đồng quản trị", "thành viên hội đồng quản trị",
    "chủ tịch hội đồng quản trị", "người đại diện theo pháp luật",
    "người đại diện theo ủy quyền", "ban kiểm soát", "kiểm soát viên",
    "kế toán trưởng", "chủ tịch ủy ban nhân dân", "chủ tịch ubnd",
    "thanh tra viên", "cán bộ công đoàn", "chủ hộ kinh doanh",
]

ADMIN_BODIES = [
    "ủy ban nhân dân tỉnh", "ủy ban nhân dân cấp xã", "ubnd",
    "sở lao động - thương binh và xã hội", "sở lao động", "sở kế hoạch và đầu tư",
    "cục sở hữu trí tuệ", "bộ tài chính", "bộ tư pháp", "bộ công thương",
    "tổng cục thuế", "cơ quan thuế", "cơ quan hải quan",
    "cơ quan đăng ký kinh doanh", "phòng đăng ký kinh doanh",
    "bảo hiểm xã hội", "tòa án", "viện kiểm sát", "cơ quan nhà nước có thẩm quyền",
    "ngân sách nhà nước", "cổng thông tin điện tử",
]

LEGAL_ACTIONS = [
    "chuyển nhượng", "góp vốn", "mua bán", "sáp nhập", "hợp nhất",
    "giải thể", "phá sản", "thành lập", "đăng ký", "cấp phép", "gia hạn",
    "ủy quyền", "chuyển đổi", "tạm ngừng", "tạm đình chỉ",
    "thu hồi", "đình chỉ", "xử phạt", "xử lý vi phạm", "cưỡng chế",
    "đào tạo", "bồi dưỡng", "hỗ trợ", "hưởng ưu đãi", "miễn thuế", "giảm thuế",
    "hoàn thuế", "khấu trừ", "kê khai", "nộp thuế", "quyết toán",
    "thanh tra", "kiểm tra", "khiếu nại", "tố cáo", "khởi kiện",
    "bảo hành", "bồi thường", "sa thải", "kỷ luật", "thử việc",
    "thương lượng", "hòa giải", "đối thoại", "đình công",
    "xử lý", "khắc phục", "bị xử lý",
    "bị phạt", "bị xử phạt", "yêu cầu", "cung cấp",
    "thông báo", "chấm dứt", "tiếp nhận", "sửa đổi",
    "chấp thuận", "xác nhận", "bảo lãnh", "vay vốn",
    "nâng cao", "tham gia", "áp dụng", "thay đổi",
]

LEGAL_OBJECTS = [
    "hóa đơn điện tử", "hóa đơn", "chứng từ", "biên lai", "sổ sách kế toán",
    "báo cáo tài chính", "báo cáo tình hình tài chính", "bảng cân đối tài khoản",
    "bảo hiểm xã hội", "bảo hiểm thất nghiệp", "bảo hiểm y tế",
    "giấy phép kinh doanh", "giấy chứng nhận đăng ký doanh nghiệp",
    "giấy chứng nhận đầu tư", "văn bằng bảo hộ", "bằng bảo hộ",
    "hợp đồng lao động", "hợp đồng thương mại", "hợp đồng chuyển nhượng",
    "hợp đồng sử dụng đối tượng sở hữu công nghiệp", "hợp đồng nhượng quyền",
    "quyền sở hữu trí tuệ", "quyền tác giả", "quyền sở hữu công nghiệp",
    "nhãn hiệu", "sáng chế", "kiểu dáng công nghiệp", "tên thương mại",
    "chỉ dẫn địa lý", "bí mật kinh doanh", "mã số thuế", "mã số doanh nghiệp",
    "quản trị doanh nghiệp", "chi phí", "vốn", "tài sản", "cổ phiếu", "cổ phần",
    "nội quy lao động", "quy chế dân chủ", "thẻ an toàn", "chữ ký số",
]

PENALTIES = [
    "phạt tiền", "phạt bao nhiêu", "bị phạt", "xử phạt vi phạm hành chính",
    "cảnh cáo", "đình chỉ hoạt động", "tước quyền sử dụng",
    "truy cứu trách nhiệm hình sự", "bồi thường", "bồi thường thiệt hại",
    "thu hồi giấy phép", "phạt tù", "bị truy tố",
    "xử lý", "khắc phục", "khắc phục hậu quả",
    "bị xử lý", "bị xử phạt", "bị xử lý như thế nào",
    "bị phạt bao nhiêu", "hình thức xử phạt",
]

RE_PENALTY = re.compile(
    r'((?:bị\s+)?(?:xử phạt|xử lý|phạt)\s+(?:như thế nào|vi phạm|bao nhiêu)'
    r'|khắc phục(?:\s+hậu quả)?'
    r'|phạt\s+(?:tiền|bao nhiêu)'
    r'|cảnh\s+cáo'
    r'|đình\s+chỉ\s+hoạt\s+động'
    r'|tước\s+quyền\s+sử\s+dụng'
    r'|truy\s+cứu\s+trách\s+nhiệm\s+hình\s+sự'
    r'|bồi\s+thường(?:\s+thiệt\s+hại)?'
    r'|thu\s+hồi\s+giấy\s+phép)',
    re.IGNORECASE
)

LAW_NAME_ALIASES = {
    "luật doanh nghiệp": "59/2020/QH14",
    "luật hỗ trợ doanh nghiệp nhỏ và vừa": "04/2017/QH14",
    "luật hỗ trợ dnnvv": "04/2017/QH14",
    "luật đầu tư": "61/2020/QH14",
    "luật sở hữu trí tuệ": "50/2005/QH11",
    "bộ luật dân sự": "91/2015/QH13",
    "bộ luật lao động": "45/2019/QH14",
    "luật cạnh tranh": "23/2018/QH14",
    "luật thương mại": "36/2005/QH11",
    "luật quản lý thuế": "38/2019/QH14",
    "luật thuế giá trị gia tăng": "48/2024/QH15",
    "luật thuế thu nhập doanh nghiệp": "14/2008/QH12",
    "luật thuế thu nhập cá nhân": "04/2007/QH12",
    "luật kế toán": "88/2015/QH13",
    "luật phá sản": "51/2014/QH13",
    "luật đấu thầu": "22/2023/QH15",
    "luật đất đai": "31/2024/QH15",
    "luật nhà ở": "27/2023/QH15",
    "luật xây dựng": "50/2014/QH13",
    "luật an toàn vệ sinh lao động": "84/2015/QH13",
    "luật bảo hiểm xã hội": "58/VBHN-VPQH",
    "luật việc làm": "38/2013/QH13",
    "luật trọng tài thương mại": "54/2010/QH12",
    "luật quảng cáo": "16/2012/QH13",
    "luật thống kê": "89/2015/QH13",
    "bộ luật tố tụng dân sự": "92/2015/QH13",
}

# Mapping từ legal_subjects → law_id (khi query không nhắc tên luật)
SUBJECT_TO_LAW = {
    "doanh nghiệp nhỏ và vừa": "04/2017/QH14",
    "doanh nghiệp siêu nhỏ": "88/2015/QH13",
    "doanh nghiệp tư nhân": "59/2020/QH14",
    "công ty cổ phần": "59/2020/QH14",
    "công ty trách nhiệm hữu hạn": "59/2020/QH14",
    "công ty tnhh": "59/2020/QH14",
    "công ty hợp danh": "59/2020/QH14",
    "hộ kinh doanh": "59/2020/QH14",
    "hợp tác xã": "23/2023/QH15",          # Luật Hợp tác xã 2023
    "tổ chức tín dụng": "47/2010/QH12",     # Luật Các tổ chức tín dụng
    "thương nhân": "36/2005/QH11",           # Luật Thương mại
    "tổ chức xúc tiến thương mại": "36/2005/QH11",
    "nhà đầu tư nước ngoài": "61/2020/QH14", # Luật Đầu tư
    "người lao động": "45/2019/QH14",
    "lao động nữ": "45/2019/QH14",
    "lao động chưa thành niên": "45/2019/QH14",
    "người khuyết tật": "45/2019/QH14",      # Bộ luật Lao động (lao động là người khuyết tật)
    "người sử dụng lao động": "45/2019/QH14",
    "người giúp việc gia đình": "45/2019/QH14",
    "chủ doanh nghiệp tư nhân": "59/2020/QH14",
    "cổ đông": "59/2020/QH14",
    "thành viên góp vốn": "59/2020/QH14",
    "nhân viên": "45/2019/QH14",
    "chủ sở hữu": "59/2020/QH14",
    # === ALIAS ===
    "công ty nhỏ và vừa": "04/2017/QH14",
    "công ty sản xuất nhỏ và vừa": "04/2017/QH14",
    "cơ sở ươm tạo": "04/2017/QH14",
    "khu làm việc chung": "04/2017/QH14",
    "doanh nghiệp khởi nghiệp sáng tạo": "04/2017/QH14",
}


@dataclass
class QueryEntities:
    """Entity trích xuất từ 1 câu query — input cho Decompose + Expand."""
    law_ids: List[str] = field(default_factory=list)
    law_names: List[str] = field(default_factory=list)
    article_nums: List[str] = field(default_factory=list)
    legal_subjects: List[str] = field(default_factory=list)
    legal_roles: List[str] = field(default_factory=list)
    admin_bodies: List[str] = field(default_factory=list)
    legal_actions: List[str] = field(default_factory=list)
    legal_objects: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)
    quantities: List[str] = field(default_factory=list)
    is_compound: bool = False

    def to_context_str(self) -> str:
        """Tạo context string để inject vào prompt Decompose/Expand."""
        parts = []
        if self.law_names:
            parts.append(f"Luật liên quan: {', '.join(self.law_names)}")
        if self.law_ids:
            parts.append(f"Số hiệu: {', '.join(self.law_ids)}")
        if self.article_nums:
            parts.append(f"Điều/khoản: {', '.join(self.article_nums)}")
        if self.legal_subjects:
            parts.append(f"Chủ thể: {', '.join(self.legal_subjects)}")
        if self.legal_roles:
            parts.append(f"Chức danh: {', '.join(self.legal_roles)}")
        if self.admin_bodies:
            parts.append(f"Cơ quan: {', '.join(self.admin_bodies)}")
        if self.legal_actions:
            parts.append(f"Hành vi: {', '.join(self.legal_actions)}")
        if self.legal_objects:
            parts.append(f"Đối tượng: {', '.join(self.legal_objects)}")
        if self.penalties:
            parts.append(f"Chế tài: {', '.join(self.penalties)}")
        if self.quantities:
            parts.append(f"Định lượng: {', '.join(self.quantities)}")
        return "; ".join(parts) if parts else ""

    def get_primary_law_id(self, query: str = "") -> Optional[str]:
        """Trả về law_id có khả năng cao nhất, dùng để filter retrieval.
        
        Khi có nhiều law_id, chọn cái xuất hiện sớm nhất trong câu hỏi gốc
        (theo vị trí text) thay vì theo thứ tự dict insertion — tránh lỗi
        filter sai âm thầm khi câu hỏi nhắc nhiều luật.
        """
        all_ids = self.get_all_law_ids()
        if not all_ids:
            return None
        if len(all_ids) == 1 or not query:
            return all_ids[0]
        # Chọn law_id mà alias/tên xuất hiện SỚM NHẤT trong câu hỏi gốc
        q_norm = normalize_vn(query)
        best_id, best_pos = all_ids[0], len(q_norm)
        for lid in all_ids:
            # Tìm vị trí match sớm nhất cho law_id này trong câu hỏi
            for alias, alias_lid in LAW_NAME_ALIASES.items():
                if alias_lid.lower() == lid.lower():
                    pos = q_norm.find(normalize_vn(alias))
                    if pos != -1 and pos < best_pos:
                        best_pos = pos
                        best_id = lid
            # Cũng check law_id trực tiếp (vd "38/2019/QH14" trong text)
            pos = query.lower().find(lid.lower())
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_id = lid
        return best_id

    def get_all_law_ids(self) -> List[str]:
        """Trả về TẤT CẢ law_ids có thể suy ra, không chỉ 1.
        
        Hữu ích khi câu hỏi liên quan đến nhiều luật (cross-law query),
        giúp tránh mất tài liệu quan trọng do hard filter chỉ lấy 1 luật.
        """
        ids = list(self.law_ids)  # Copy để không mutate
        # Từ law_names → resolve qua alias
        for name in self.law_names:
            name_lower = name.lower()
            for alias, lid in LAW_NAME_ALIASES.items():
                if alias in name_lower or name_lower in alias:
                    if lid not in ids:
                        ids.append(lid)
        # Fallback: suy từ legal_subjects
        if not ids:
            for subj in self.legal_subjects:
                subj_norm = normalize_vn(subj)
                for key, lid in SUBJECT_TO_LAW.items():
                    if normalize_vn(key) in subj_norm or subj_norm in normalize_vn(key):
                        if lid not in ids:
                            ids.append(lid)
        return ids


class QueryEntityExtractor:
    """
    Trích xuất entity pháp lý từ câu hỏi, dùng Regex + Gazetteer.
    Không cần LLM, latency ~0ms — chạy cho mọi query trước khi Decompose.
    """

    @staticmethod
    def extract(query: str) -> QueryEntities:
        q = query.lower()
        q_norm = normalize_vn(query)  # Không dấu để fuzzy match
        entities = QueryEntities()

        # ── Regex (dùng query gốc có dấu) ──
        entities.law_ids = [m.group(1).upper() for m in RE_LAW_ID.finditer(query)]
        entities.article_nums = [m.group(1) for m in RE_ARTICLE.finditer(query)]
        entities.quantities = [m.group(1) for m in RE_QUANTITY.finditer(query)]
        entities.penalties = [m.group(0).lower() for m in RE_PENALTY.finditer(query)]

        # ── Helper: match gazetteer với normalize ──
        def match_gazetteer(gazetteer: list[str], query_norm: str) -> list[str]:
            """Match dài nhất trước, tránh substring overlap."""
            sorted_items = sorted(gazetteer, key=len, reverse=True)
            matched = set()
            matched_norm = set()
            for item in sorted_items:
                item_norm = normalize_vn(item)
                if item_norm in query_norm:
                    # Tránh overlap: nếu item ngắn hơn đã bị item dài hơn bao phủ
                    if not any(item_norm in m for m in matched_norm if m != item_norm):
                        matched.add(item)
                        matched_norm.add(item_norm)
            return sorted(matched, key=len, reverse=True)

        entities.legal_subjects = match_gazetteer(LEGAL_SUBJECTS, q_norm)
        entities.legal_roles = match_gazetteer(LEGAL_ROLES, q_norm)
        entities.admin_bodies = match_gazetteer(ADMIN_BODIES, q_norm)
        entities.legal_actions = match_gazetteer(LEGAL_ACTIONS, q_norm)
        entities.legal_objects = match_gazetteer(LEGAL_OBJECTS, q_norm)
        # Merge regex + gazetteer cho penalties
        penalty_gaz = match_gazetteer(PENALTIES, q_norm)
        for p in penalty_gaz:
            if p.lower() not in entities.penalties:
                entities.penalties.append(p)

        # ── Law names: match alias → resolve law_id ──
        for alias, lid in LAW_NAME_ALIASES.items():
            alias_norm = normalize_vn(alias)
            if alias_norm in q_norm:
                entities.law_names.append(alias)
                if lid not in entities.law_ids:
                    entities.law_ids.append(lid)

        # ── Compound detection ──
        entities.is_compound = QueryEntityExtractor._detect_compound(query)

        return entities

    @staticmethod
    def _detect_compound(query: str) -> bool:
        """Heuristic: câu hỏi compound nếu có nhiều ý hỏi độc lập."""
        # ≥2 dấu hỏi → chắc chắn compound
        if query.count('?') >= 2:
            return True

        # Chuẩn hóa không dấu để tìm từ hỏi (tránh lỗi accent)
        q_norm = normalize_vn(query)

        # Từ hỏi tiếng Việt (không dấu) — chia thành 2 nhóm
        # Nhóm 1: Từ hỏi đơn lẻ (cần \b để tránh match substring)
        single_q_words = [
            r'\bgi\b', r'(?:vi|tai)\s+sao\b', r'bao nhieu',
            r'nhu the nao', r'the nao', r'\bai\b', r'o dau',
            r'khi nao', r'lam sao',
        ]
        # Nhóm 2: Cặp từ hỏi có/không, đã/chưa, phải/không (yêu cầu gần nhau ≤15 từ)
        pair_q_words = [
            r'\bco\b(?:\s+\S+){0,15}\s+\bkhong\b',
            r'\bda\b(?:\s+\S+){0,10}\s+\bchua\b',
            r'\bphai\b(?:\s+\S+){0,10}\s+\bkhong\b',
        ]

        q_positions = []
        for w in single_q_words + pair_q_words:
            m = re.search(w, q_norm)
            if m:
                q_positions.append(m.start())

        if not q_positions:
            return False

        first_q_pos = min(q_positions)

        # Nếu "và" xuất hiện SAU từ hỏi đầu tiên VÀ có ≥2 từ hỏi → compound
        if first_q_pos < len(q_norm):
            after_q = q_norm[first_q_pos:]
            if re.search(r'\bva\b', after_q) and len(q_positions) >= 2:
                return True
            # Có ≥1 dấu phẩy sau từ hỏi + câu có dấu hỏi
            if after_q.count(',') >= 1 and '?' in query:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════
# ② QUERY EXPANDER — Có sẵn, giữ nguyên
# ═══════════════════════════════════════════════════════════════════

class QueryExpander:
    def __init__(self, llm, n_variants: int = 2):
        if not HAS_VLLM:
            raise ImportError("vllm is required for QueryExpander. Install with: pip install vllm")
        self.llm       = llm
        self.n_variants = n_variants
        self.sampling  = SamplingParams(temperature=0.3, max_tokens=150)

    def expand_batch(self, queries: list[str], entity_contexts: list[str] | None = None) -> list[list[str]]:
        if entity_contexts is None:
            entity_contexts = [""] * len(queries)
        prompts = [self._build_prompt(q, ctx) for q, ctx in zip(queries, entity_contexts)]
        outputs = self.llm.generate(prompts, self.sampling)
        results = []
        for original, output in zip(queries, outputs):
            text     = output.outputs[0].text.strip()
            variants = [l.strip() for l in text.split('\n') if l.strip()]
            variants = variants[:self.n_variants]
            results.append([original] + variants)
        return results

    def _build_prompt(self, query: str, entity_context: str = "") -> str:
        ctx_line = f"\nTHỰC THỂ ĐÃ XÁC ĐỊNH: {entity_context}\n" if entity_context else ""
        return (
            f"<|im_start|>system\n"
            f"Viết lại câu hỏi pháp lý sau thành {self.n_variants} cách "
            f"dùng thuật ngữ pháp lý chính xác hơn. Mỗi cách 1 dòng, "
            f"không đánh số, không giải thích.\n"
            f"{ctx_line}"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{query}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ═══════════════════════════════════════════════════════════════════
# ③ QUERY DECOMPOSER — Tách compound query, có entity context
# ═══════════════════════════════════════════════════════════════════

class QueryDecomposer:
    """
    Phân rã câu hỏi compound thành sub-queries nguyên tử.
    Dùng entity context từ NER để giữ đúng chủ thể/luật.
    """

    def __init__(self, llm, max_sub_queries: int = 3):
        if not HAS_VLLM:
            raise ImportError("vllm is required for QueryDecomposer. Install with: pip install vllm")
        self.llm = llm
        self.max_sub = max_sub_queries
        self.sampling = SamplingParams(temperature=0.1, max_tokens=300)

    def decompose_batch(
        self,
        queries: list[str],
        entities_list: list[QueryEntities] | None = None,
    ) -> list[list[str]]:
        """Decompose 1 batch queries, có entity context."""
        if entities_list is None:
            entities_list = [QueryEntityExtractor.extract(q) for q in queries]

        prompts = []
        for query, entities in zip(queries, entities_list):
            prompts.append(self._build_prompt(query, entities))

        outputs = self.llm.generate(prompts, self.sampling)
        results = []
        for original, output in zip(queries, outputs):
            text = output.outputs[0].text.strip()
            sub_queries = self._parse_output(text, original)
            results.append(sub_queries)
        return results

    def _build_prompt(self, query: str, entities: QueryEntities) -> str:
        ctx = entities.to_context_str()
        ctx_block = (
            f"THỰC THỂ PHÁP LÝ ĐÃ XÁC ĐỊNH TRONG CÂU HỎI:\n"
            f"{ctx}\n\n"
        ) if ctx else ""

        return (
            f"<|im_start|>system\n"
            f"Bạn là chuyên gia pháp lý Việt Nam. Phân tích câu hỏi sau.\n"
            f"Nếu câu hỏi chứa NHIỀU Ý HỎI ĐỘC LẬP (nối bằng 'và', 'cùng', 'sau đó'), "
            f"hãy tách thành các câu hỏi con ĐƠN GIẢN, MỖI CÂU CHỈ HỎI 1 Ý.\n"
            f"Nếu câu hỏi đã là 1 ý duy nhất, trả về nguyên văn câu hỏi đó.\n\n"
            f"QUY TẮC:\n"
            f"- Mỗi sub-query phải là câu hỏi HOÀN CHỈNH, có chủ ngữ rõ ràng\n"
            f"- KẾ THỪA chủ thể và luật từ THỰC THỂ PHÁP LÝ ĐÃ XÁC ĐỊNH bên dưới\n"
            f"- KHÔNG thêm thông tin không có trong câu gốc\n"
            f"- Mỗi sub-query 1 dòng, không đánh số\n"
            f"- Tối đa {self.max_sub} sub-query\n"
            f"- Nếu câu hỏi đơn giản: trả về ĐÚNG 1 dòng là câu hỏi gốc\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{ctx_block}"
            f"CÂU HỎI: {query}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_output(self, text: str, original: str) -> list[str]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Loại bỏ prefix số (1., 2., -)
        lines = [re.sub(r'^[\d\.\-\*\s]+', '', l).strip() for l in lines]
        # Giữ dòng đủ dài (bỏ yêu cầu '?' — LLM có thể trả sub-query dạng mệnh lệnh)
        lines = [l for l in lines if len(l) > 15]
        if not lines:
            return [original]
        # Luôn giữ original làm sub-query đầu tiên
        if original not in lines:
            lines.insert(0, original)
        return lines[:self.max_sub + 1]


# ═══════════════════════════════════════════════════════════════════
# ④ PIPELINE TÍCH HỢP: NER → Decompose → Expand
# ═══════════════════════════════════════════════════════════════════

class QueryPipeline:
    """
    Pipeline hoàn chỉnh:
        1. NER (QueryEntityExtractor) — regex+gazetteer, 0ms
        2. Decompose (QueryDecomposer) — LLM, chỉ nếu compound
        3. Expand (QueryExpander) — LLM, mỗi sub-query → N variants
    Output: list[dict] với key 'query' và 'entity_context' cho mỗi variant.
    """

    def __init__(
        self,
        llm,
        n_variants: int = 2,
        max_sub_queries: int = 3,
        force_decompose: bool = False,
    ):
        self.extractor = QueryEntityExtractor()
        self.expander = QueryExpander(llm, n_variants=n_variants) if llm is not None else None
        self.decomposer = QueryDecomposer(llm, max_sub_queries=max_sub_queries) if llm is not None else None
        self.force_decompose = force_decompose

    def process_single(self, query: str) -> dict:
        """
        Xử lý 1 query → trả về dict:
        {
            'original': str,
            'entities': QueryEntities,
            'sub_queries': list[str],
            'expanded_queries': list[dict],  # [{query, entity_context, source}, ...]
        }
        """
        # ── Bước 1: NER ──
        entities = self.extractor.extract(query)

        # ── Bước 2: Decompose (nếu compound và có LLM) ──
        if self.decomposer and (entities.is_compound or self.force_decompose):
            sub_queries = self.decomposer.decompose_batch(
                [query], [entities]
            )[0]
        else:
            sub_queries = [query]

        # ── Bước 3: Extract entity cho từng sub-query ──
        sub_entities = [self.extractor.extract(sq) for sq in sub_queries]

        # ── Bước 4: Expand từng sub-query (nếu có LLM) ──
        expanded_queries = []
        if self.expander:
            entity_contexts = [e.to_context_str() for e in sub_entities]
            for sq, ctx in zip(sub_queries, entity_contexts):
                variants = self.expander.expand_batch([sq], [ctx])[0]
                for v in variants:
                    expanded_queries.append({
                        "query": v,
                        "entity_context": ctx,
                        "source": "sub_query" if sq != query else "original",
                    })
        else:
            # Không có LLM → dùng sub_queries trực tiếp
            for sq in sub_queries:
                e = self.extractor.extract(sq)
                expanded_queries.append({
                    "query": sq,
                    "entity_context": e.to_context_str(),
                    "source": "sub_query" if sq != query else "original",
                })

        return {
            "original": query,
            "entities": entities,
            "sub_queries": sub_queries,
            "expanded_queries": expanded_queries,
        }

    def process_batch(self, queries: list[str]) -> list[dict]:
        """Xử lý batch queries."""
        return [self.process_single(q) for q in queries]