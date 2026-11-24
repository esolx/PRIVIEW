"""
EE (External Extended) 속성 사전
================================
외부에서 접근 가능하여 데이터셋 속성과 결합 시 노출 위험을 높이는 속성 목록.

분류 기준
---------
- 가용성 (Availability)  : 해당 정보가 외부에 실제로 존재하는 비율 (0~1)
- 접근성 (Accessibility) : 일반인이 해당 정보에 접근할 수 있는 용이성 (0~1)
- 검색비용 (Search Effort, 역수) : 해당 정보를 찾는 데 드는 비용이 낮을수록 높음 (0~1)
- EE 점수 (ee_score)     : 세 축의 평균 → 0.7 이상이면 EE, 미만이면 ER로 분류

사용 방법
---------
from ee_attribute_dictionary import EE_DICT, get_ee_attributes, get_er_attributes, match_columns

# 데이터셋 컬럼명과 EE 사전 매칭
matched = match_columns(["age", "gender", "zipcode", "blood_type"])

EE/ER 분류 기준
---------------
EE (External Extended) : 가용성·접근성·검색비용 세 축 모두 0.7 이상
ER (External Restricted): 세 축 중 하나라도 0.7 미만

참고 문헌
---------
- Sweeney (1997/2002): 준식별자 개념 및 k-익명성
- Golle (2006): 인구통계 속성의 재식별 고유성
- Rocher et al. (2019): Nature Communications — 불완전 데이터셋 재식별 가능성
- PLOS ONE (2018): SNS 기반 개인정보 노출 실증 연구
- GDPR Local (2026): 국내 GDPR 적용 가이드라인
- CPRS (2025): Contemporary Privacy Risk Study
- Re-ID Review (2017): 재식별 위험도 리뷰 연구
- Nature Comm. (2019): 일과 패턴·이동 데이터 재식별
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

EE_THRESHOLD = 0.7  # 이 점수 이상이면 EE, 미만이면 ER


@dataclass
class EEAttribute:
    """EE/ER 속성 항목 하나를 표현하는 데이터 클래스."""
    category: str           # 카테고리 (인구통계, 위치, 이동, 직업, 경제, 온라인 행동, 사회관계, 신체, 외형, 소비, 취향, 시간, 행태)
    attr_en: str            # 영문 속성명 (컬럼명 매칭에 사용)
    attr_ko: str            # 한글 속성명
    aliases: list[str]      # 컬럼명 매칭에 사용할 추가 키워드 (소문자, 언더스코어 포함)
    availability: float     # 가용성 (0~1)
    accessibility: float    # 접근성 (0~1)
    search_effort: float    # 검색비용 역수 (0~1, 높을수록 찾기 쉬움)
    example: str            # 외부 노출 경로 예시
    source: str             # 참고 문헌
    notes: str = ""         # 추가 비고

    @property
    def ee_score(self) -> float:
        """EE 종합 점수 (세 축 평균)."""
        return round((self.availability + self.accessibility + self.search_effort) / 3, 3)

    @property
    def ee_type(self) -> str:
        """EE 또는 ER 분류."""
        return "EE" if self.ee_score >= EE_THRESHOLD else "ER"

    @property
    def is_ee(self) -> bool:
        return self.ee_score >= EE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "attr_en": self.attr_en,
            "attr_ko": self.attr_ko,
            "aliases": self.aliases,
            "availability": self.availability,
            "accessibility": self.accessibility,
            "search_effort": self.search_effort,
            "ee_score": self.ee_score,
            "ee_type": self.ee_type,
            "example": self.example,
            "source": self.source,
            "notes": self.notes,
        }


# =============================================================================
# EE 속성 사전 전체 목록
# =============================================================================

EE_DICT: list[EEAttribute] = [

    # -------------------------------------------------------------------------
    # 인구통계 (Demographic)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="인구통계",
        attr_en="Age", attr_ko="나이",
        aliases=["age", "나이", "연령", "age_years", "current_age", "age_at_enrollment"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="SNS 프로필·공개 이력서 기재",
        source="Sweeney 1997; Golle 2006",
        notes="Sweeney(1997)에서 우편번호·성별·생년월일 조합의 87%가 개인 식별 가능함을 입증"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Gender", attr_ko="성별",
        aliases=["gender", "sex", "성별", "gender_code", "sex_code", "gender_identity"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="프로필 사진·이름에서 추론 가능",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Date of Birth", attr_ko="생년월일",
        aliases=["dob", "date_of_birth", "birth_date", "birthdate", "생년월일", "birth_dt"],
        availability=0.95, accessibility=0.95, search_effort=0.9,
        example="선거인 명부·공공기록",
        source="Sweeney 1997",
        notes="생년월일 전체보다 연도+월만 노출되어도 결합 위험 존재"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Birth Year", attr_ko="출생연도",
        aliases=["birth_year", "year_of_birth", "출생연도", "yob", "birth_yr"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="SNS 자기소개 빈번 노출",
        source="Golle 2006"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Nationality", attr_ko="국적",
        aliases=["nationality", "국적", "citizenship", "country_of_birth", "country_of_origin"],
        availability=0.95, accessibility=0.95, search_effort=0.95,
        example="공개 프로필·입국 기록",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Ethnicity / Race", attr_ko="민족·인종",
        aliases=["ethnicity", "race", "ethnic_group", "민족", "인종", "ethnic_background"],
        availability=0.85, accessibility=0.8, search_effort=0.8,
        example="사진 추론·이름 분석으로 예측 가능",
        source="PLOS ONE 2018",
        notes="법정 민감정보(개보법 제23조 인종·민족)와 중복될 수 있음 — 민감속성 탐지에서 이미 탐지된 경우 EE 매칭 제외"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="First Name", attr_ko="이름(名)",
        aliases=["first_name", "given_name", "이름", "name_first", "fname", "forename"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 프로필 어디서나 존재",
        source="GDPR Local 2026"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Last Name", attr_ko="성(姓)",
        aliases=["last_name", "family_name", "surname", "성", "lname", "name_last"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 프로필 어디서나 존재",
        source="GDPR Local 2026"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Full Name", attr_ko="성명",
        aliases=["full_name", "name", "성명", "patient_name", "user_name", "member_name", "customer_name"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="검색엔진 1회 조회로 확인",
        source="Sweeney 1997"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Language", attr_ko="사용 언어",
        aliases=["language", "언어", "lang", "spoken_language", "preferred_language", "native_language"],
        availability=0.9, accessibility=0.9, search_effort=0.95,
        example="글·댓글 분석으로 자동 추론",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Marital Status", attr_ko="혼인 여부",
        aliases=["marital_status", "혼인", "marriage_status", "civil_status", "married"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="SNS 관계 상태 공개",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Number of Children", attr_ko="자녀 수",
        aliases=["num_children", "number_of_children", "children_count", "자녀수", "자녀 수", "미혼자녀수", "child_count", "num_kids"],
        availability=0.7, accessibility=0.7, search_effort=0.7,
        example="SNS 게시물 추론 가능",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="인구통계",
        attr_en="Religion", attr_ko="종교",
        aliases=["religion", "종교", "faith", "religious_affiliation", "belief"],
        availability=0.75, accessibility=0.75, search_effort=0.75,
        example="SNS 프로필 자기 기재",
        source="PLOS ONE 2018",
        notes="법정 민감정보(사상·신념)와 중복 가능 — 이미 B-1 탐지 시 EE 매칭 제외"
    ),

    # -------------------------------------------------------------------------
    # 위치, 이동 (Location & Mobility)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="위치·이동",
        attr_en="ZIP Code / Postal Code", attr_ko="우편번호",
        aliases=["zip_code", "zipcode", "postal_code", "zip", "우편번호", "postcode"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 주소록·배달 정보",
        source="Sweeney 1997",
        notes="Sweeney(1997): 우편번호+성별+생년월일만으로 미국 인구 87% 식별 가능"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="City / District", attr_ko="시·구",
        aliases=["city", "district", "시", "구", "town", "municipality", "city_name", "residence_city"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="SNS 위치 태그·프로필 거주지",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="State / Province", attr_ko="도·광역시",
        aliases=["state", "province", "region", "도", "광역시", "state_code", "province_code"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 주소·이력서",
        source="Golle 2006"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="Country", attr_ko="국가",
        aliases=["country", "국가", "country_code", "nation", "residence_country"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 프로필 어디서나 존재",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="Home Address", attr_ko="자택 주소",
        aliases=["address", "home_address", "street_address", "주소", "residence_address", "addr"],
        availability=0.9, accessibility=0.85, search_effort=0.85,
        example="선거인 명부·온라인 전화번호부",
        source="Sweeney 1997"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="GPS Coordinates (approx.)", attr_ko="GPS 좌표(대략)",
        aliases=["latitude", "longitude", "lat", "lon", "lng", "gps", "coordinates", "location", "geo_location"],
        availability=0.75, accessibility=0.7, search_effort=0.7,
        example="SNS 체크인·사진 메타데이터",
        source="Nature Comm. 2019",
        notes="정밀 좌표가 아닌 대략적 위치도 일상 패턴 분석으로 개인 식별 가능 (de Montjoye et al., 2013)"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="Workplace Location", attr_ko="직장 위치",
        aliases=["workplace", "work_location", "office_address", "직장주소", "work_address", "employer_address"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="LinkedIn 프로필·기업 홈페이지",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="위치·이동",
        attr_en="Commute Origin / Destination", attr_ko="출퇴근 경로",
        aliases=["commute_origin", "commute_dest", "home_to_work", "출퇴근", "commute_route"],
        availability=0.65, accessibility=0.6, search_effort=0.6,
        example="교통카드 데이터·위치 기록 분석",
        source="Nature Comm. 2019"
    ),

    # -------------------------------------------------------------------------
    # 직업, 경제 (Occupation & Finance)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="직업·경제",
        attr_en="Occupation / Job Title", attr_ko="직업·직책",
        aliases=["occupation", "job_title", "position", "role", "직업", "직책", "title", "job", "profession"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="LinkedIn·공개 이력서",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Employer / Company Name", attr_ko="회사명",
        aliases=["employer", "company", "company_name", "organization", "회사", "직장명", "employer_name", "workplace_name"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="LinkedIn·기업 홈페이지 직원 소개",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Industry / Sector", attr_ko="업종",
        aliases=["industry", "sector", "업종", "field", "business_type", "industry_code"],
        availability=0.95, accessibility=0.95, search_effort=0.95,
        example="LinkedIn·기업 등록 정보",
        source="Re-ID Review 2017"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Education Level", attr_ko="학력",
        aliases=["education", "education_level", "degree", "학력", "educational_background", "highest_education"],
        availability=0.9, accessibility=0.9, search_effort=0.9,
        example="LinkedIn·공개 이력서·동문 네트워크",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="School / University Name", attr_ko="학교명",
        aliases=["school", "university", "college", "학교", "학교명", "institution", "school_name", "alma_mater"],
        availability=0.95, accessibility=0.95, search_effort=0.95,
        example="LinkedIn·동문 DB",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Income Range (approximate)", attr_ko="소득 구간",
        aliases=["income", "salary", "income_range", "소득", "급여", "wage", "annual_income", "salary_range"],
        availability=0.7, accessibility=0.65, search_effort=0.65,
        example="공개 연봉 정보·직종별 통계",
        source="Re-ID Review 2017"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Homeownership Status", attr_ko="주택 소유 여부",
        aliases=["homeownership", "home_ownership", "주택소유", "property_ownership", "owns_home"],
        availability=0.7, accessibility=0.65, search_effort=0.65,
        example="부동산 공개 등기 DB",
        source="Re-ID Review 2017"
    ),
    EEAttribute(
        category="직업·경제",
        attr_en="Vehicle Ownership", attr_ko="차량 소유",
        aliases=["vehicle_ownership", "car_ownership", "차량소유", "owns_vehicle", "has_car"],
        availability=0.7, accessibility=0.65, search_effort=0.65,
        example="자동차 등록 공공기록",
        source="Re-ID Review 2017"
    ),

    # -------------------------------------------------------------------------
    # 온라인 행동 (Online Behavior)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="온라인 행동",
        attr_en="Username / Handle", attr_ko="사용자명",
        aliases=["username", "user_id", "handle", "screen_name", "사용자명", "user_handle", "login_id", "account_name"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="검색엔진 1회 조회",
        source="GDPR Local 2026"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Email Address (domain)", attr_ko="이메일 도메인",
        aliases=["email", "email_address", "이메일", "email_domain", "e_mail"],
        availability=0.9, accessibility=0.9, search_effort=0.9,
        example="이름+회사로 이메일 추론",
        source="GDPR Local 2026",
        notes="전체 이메일 주소는 직접 식별자이지만, 도메인 부분만으로도 소속 기관 추론 가능"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Social Media Profile URL", attr_ko="SNS 프로필 URL",
        aliases=["social_media", "sns", "profile_url", "facebook", "instagram", "twitter", "linkedin_url", "social_profile"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="검색엔진에서 즉시 확인",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Public Post Content", attr_ko="공개 게시물",
        aliases=["post_content", "public_post", "게시물", "post", "content", "published_content"],
        availability=0.95, accessibility=0.95, search_effort=0.95,
        example="SNS·블로그 크롤링 가능",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Profile Photo", attr_ko="프로필 사진",
        aliases=["profile_photo", "profile_image", "photo", "picture", "avatar", "프로필사진"],
        availability=1.0, accessibility=1.0, search_effort=1.0,
        example="공개 SNS·역이미지 검색",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Blog / Website URL", attr_ko="블로그·웹사이트",
        aliases=["blog", "website", "url", "homepage", "web_url", "blog_url", "personal_website"],
        availability=0.9, accessibility=0.9, search_effort=0.9,
        example="검색엔진에서 직접 확인",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="IP Address (ISP-level)", attr_ko="IP 주소(ISP 수준)",
        aliases=["ip_address", "ip", "ip_addr", "client_ip", "source_ip"],
        availability=0.85, accessibility=0.8, search_effort=0.8,
        example="공개 IP 조회 서비스",
        source="GDPR Local 2026"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Device Type / OS", attr_ko="기기·운영체제",
        aliases=["device_type", "operating_system", "device_os", "platform", "user_agent", "os_type", "client_os"],
        availability=0.8, accessibility=0.8, search_effort=0.85,
        example="User-Agent 헤더·공개 API",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="온라인 행동",
        attr_en="Following / Follower List", attr_ko="팔로우·팔로워 목록",
        aliases=["followers", "following", "follower_count", "following_count", "팔로워", "connections"],
        availability=0.9, accessibility=0.9, search_effort=0.9,
        example="공개 계정 목록 열람",
        source="CPRS 2025"
    ),

    # -------------------------------------------------------------------------
    # 사회관계 (Social Relations)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="사회관계",
        attr_en="Relationship Status", attr_ko="연애·결혼 상태",
        aliases=["relationship_status", "marital_status", "연애상태", "relationship", "partner_status"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="SNS 프로필 공개 기재",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="사회관계",
        attr_en="Family Members (names)", attr_ko="가족 구성원 이름",
        aliases=["family_members", "family", "spouse_name", "parent_name", "가족", "family_name_list"],
        availability=0.8, accessibility=0.8, search_effort=0.75,
        example="SNS 태그·가족 게시물",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="사회관계",
        attr_en="Social Network Connections", attr_ko="소셜 연결망",
        aliases=["social_network", "connections", "network_size", "소셜네트워크", "contact_list"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="LinkedIn 1촌 공개",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="사회관계",
        attr_en="Group / Association Membership", attr_ko="단체·협회 가입",
        aliases=["group_membership", "association", "club", "organization_membership", "단체", "협회"],
        availability=0.8, accessibility=0.8, search_effort=0.8,
        example="SNS·단체 홈페이지 공개",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="사회관계",
        attr_en="Alumni Network", attr_ko="동창 네트워크",
        aliases=["alumni", "alumni_network", "school_network", "동창", "동문"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="동문 DB·SNS 그룹",
        source="CPRS 2025"
    ),

    # -------------------------------------------------------------------------
    # 신체, 외형 (Physical Appearance)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="신체·외형",
        attr_en="Height (approximate)", attr_ko="키(대략)",
        aliases=["height", "키", "stature", "body_height", "height_cm"],
        availability=0.7, accessibility=0.65, search_effort=0.65,
        example="프로필·사진에서 추론",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="신체·외형",
        attr_en="Physical Appearance (visible)", attr_ko="외모 특성",
        aliases=["appearance", "physical_description", "외모", "looks", "facial_features"],
        availability=0.85, accessibility=0.85, search_effort=0.9,
        example="공개 사진 역이미지 검색",
        source="CPRS 2025"
    ),
    EEAttribute(
        category="신체·외형",
        attr_en="Hair Color / Style", attr_ko="머리 색·스타일",
        aliases=["hair_color", "hair_style", "머리색", "hair", "hair_type"],
        availability=0.75, accessibility=0.75, search_effort=0.8,
        example="공개 사진에서 확인",
        source="PLOS ONE 2018"
    ),

    # -------------------------------------------------------------------------
    # 소비, 취향 (Consumption & Preference)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="소비·취향",
        attr_en="Hobby / Leisure Activity", attr_ko="취미 활동",
        aliases=["hobby", "hobbies", "leisure", "취미", "interests", "activities", "pastimes"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="SNS 게시물·그룹 가입",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="소비·취향",
        attr_en="Sports Team / Fan Affiliation", attr_ko="응원 팀·팬덤",
        aliases=["sports_team", "favorite_team", "fan", "팬", "응원팀", "sports_affiliation"],
        availability=0.9, accessibility=0.9, search_effort=0.9,
        example="SNS 공개 기재·해시태그",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="소비·취향",
        attr_en="Music Taste", attr_ko="음악 취향",
        aliases=["music_taste", "music_preference", "favorite_music", "음악", "genre_preference"],
        availability=0.85, accessibility=0.85, search_effort=0.85,
        example="스트리밍 공개 플레이리스트",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="소비·취향",
        attr_en="Food Preferences / Diet", attr_ko="식품 선호·식단",
        aliases=["food_preference", "diet", "dietary_preference", "식단", "음식", "food_habit", "eating_habit"],
        availability=0.75, accessibility=0.75, search_effort=0.75,
        example="SNS 음식 게시물·리뷰",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="소비·취향",
        attr_en="Travel Preferences", attr_ko="여행 선호",
        aliases=["travel", "travel_preference", "여행", "travel_history", "visited_countries"],
        availability=0.8, accessibility=0.8, search_effort=0.8,
        example="SNS 여행 게시물·체크인",
        source="CPRS 2025"
    ),

    # -------------------------------------------------------------------------
    # 시간, 행태 (Temporal & Behavioral)
    # -------------------------------------------------------------------------
    EEAttribute(
        category="시간·행태",
        attr_en="Daily Routine Pattern", attr_ko="일과 패턴",
        aliases=["daily_routine", "routine", "일과", "schedule", "daily_pattern", "activity_pattern"],
        availability=0.7, accessibility=0.65, search_effort=0.65,
        example="SNS 활동 시간대 분석",
        source="Nature Comm. 2019",
        notes="de Montjoye et al.(2013): 4개 시공간 좌표만으로 95%의 개인 식별 가능"
    ),
    EEAttribute(
        category="시간·행태",
        attr_en="Working Hours", attr_ko="근무 시간대",
        aliases=["working_hours", "work_hours", "근무시간", "work_schedule", "business_hours"],
        availability=0.75, accessibility=0.75, search_effort=0.8,
        example="게시물 타임스탬프 분석",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="시간·행태",
        attr_en="Posting Time Preference", attr_ko="게시 선호 시간",
        aliases=["posting_time", "post_time", "게시시간", "activity_time", "active_hours"],
        availability=0.85, accessibility=0.9, search_effort=0.95,
        example="공개 타임라인 자동 분석",
        source="PLOS ONE 2018"
    ),
    EEAttribute(
        category="시간·행태",
        attr_en="Sleep Schedule (approx.)", attr_ko="수면 패턴(대략)",
        aliases=["sleep_schedule", "sleep_pattern", "수면패턴", "sleep_time", "bedtime"],
        availability=0.65, accessibility=0.6, search_effort=0.6,
        example="게시 시각 패턴 분석",
        source="PLOS ONE 2018"
    ),
]


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_ee_attributes() -> list[EEAttribute]:
    """EE 점수 0.7 이상인 속성만 반환 (실질적 외부 연결 위험 있는 속성)."""
    return [a for a in EE_DICT if a.is_ee]


def get_er_attributes() -> list[EEAttribute]:
    """EE 점수 0.7 미만인 속성 반환 (외부 연결 장벽 있는 속성)."""
    return [a for a in EE_DICT if not a.is_ee]


def get_by_category(category: str) -> list[EEAttribute]:
    """카테고리로 속성 필터링."""
    return [a for a in EE_DICT if a.category == category]


def _normalize(text: str) -> str:
    """컬럼명 정규화: camelCase 분리, 소문자, 언더스코어·공백·하이픈 → 언더스코어."""
    t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(text).strip())   # zipCode → zip_Code
    return re.sub(r"[\s\-]+", "_", t.lower())


_HANGUL = re.compile(r"[가-힣]")


def _tokens(norm: str) -> list:
    """정규화된 이름 → 토큰 목록 (영문/숫자는 단어 단위, 한글은 덩어리 그대로)."""
    return [t for t in re.split(r"[^0-9a-z가-힣]+", norm) if t]


def _alias_partial_hit(alias_norm: str, col_norm: str) -> bool:
    """[v9] 부분 일치 규칙.

    예전 규칙(alias in col  또는  col in alias, 단순 문자열 포함)은 다음 오탐을 냈다:
      message·dosage·average_score·storage_gb → Age('age' 포함),
      code·x → Gender, recipient·shipping → IP('ip' 포함),
      구매금액·시험점수·가구원수 → 시·구('구'/'시' 한 글자), 성적·성과급 → 성(姓).
    새 규칙:
      · 영문 alias : 토큰 경계에서만. alias 토큰 열이 컬럼 토큰 열 안에 연속으로
                    있어야 함 (age_band ✓, usage_count ✗)
      · 한글 alias : 2글자 이상만 부분 포함 허용 (한 글자 '시','구','도','성' 은 완전 일치만)
      · 컬럼명이 alias 안에 포함되는 역방향 매칭은 하지 않는다 (code → zip_code 류 오탐)
    """
    if not alias_norm:
        return False
    if _HANGUL.search(alias_norm):
        return len(alias_norm.replace("_", "")) >= 2 and alias_norm in col_norm
    a_tok, c_tok = _tokens(alias_norm), _tokens(col_norm)
    if not a_tok:
        return False
    m = len(a_tok)
    return any(c_tok[i:i + m] == a_tok for i in range(len(c_tok) - m + 1))


def match_columns(
    columns: list[str],
    ee_only: bool = True,
    score_threshold: float = EE_THRESHOLD,
    already_flagged: Optional[list[str]] = None,
) -> dict[str, Optional[EEAttribute]]:
    """
    데이터셋 컬럼명 목록을 EE 사전과 매칭.

    Parameters
    ----------
    columns : list[str]
        데이터셋의 컬럼명 목록
    ee_only : bool
        True이면 EE 점수 threshold 이상인 속성만 매칭 대상으로 사용
    score_threshold : float
        EE 분류 임계값 (기본값 0.7)
    already_flagged : list[str] | None
        상위 파이프라인(법정 민감정보 탐지 레이어)에서 이미 탐지된 컬럼명 목록.
        여기 포함된 컬럼은 EE 매칭을 건너뛰고 None을 반환.

    Returns
    -------
    dict[str, Optional[EEAttribute]]
        {컬럼명: 매칭된 EEAttribute 또는 None}
        already_flagged에 포함된 컬럼은 항상 None.

    Examples
    --------
    >>> flagged = legal_detector.run(dataset)   # 상위 파이프라인 결과
    >>> results = match_columns(["age", "gender", "diagnosis_code", "zipcode"],
    ...                         already_flagged=flagged)
    >>> for col, attr in results.items():
    ...     if attr:
    ...         print(f"{col} → {attr.attr_ko} (EE점수: {attr.ee_score}, 유형: {attr.ee_type})")
    """
    # 상위 파이프라인에서 넘어온 컬럼명을 정규화해서 집합으로 만들기
    # set으로 만들면 'in' 검사가 O(1) — 리스트보다 빠름
    flagged_norms: set[str] = (
        {_normalize(c) for c in already_flagged} if already_flagged else set()
    )

    candidates = [a for a in EE_DICT if a.ee_score >= score_threshold] if ee_only else EE_DICT
    result: dict[str, Optional[EEAttribute]] = {}

    for col in columns:
        col_norm = _normalize(col)

        # 상위 파이프라인에서 이미 탐지된 컬럼이면 매칭 없이 바로 None
        if col_norm in flagged_norms:
            result[col] = None
            continue

        matched = None

        # 1순위: 정규화된 영문 속성명 완전 일치
        for attr in candidates:
            if _normalize(attr.attr_en) == col_norm:
                matched = attr
                break

        # 2순위: aliases 완전 일치
        if not matched:
            for attr in candidates:
                if col_norm in [_normalize(a) for a in attr.aliases]:
                    matched = attr
                    break

        # 3순위: aliases 부분 일치 (토큰 경계 ,  한글 2글자 이상, _alias_partial_hit 참고)
        #   여러 속성이 걸리면 가장 긴 alias 를 고른다 (더 구체적인 쪽:
        #   '이메일주소' 에서 '주소' 보다 '이메일').
        if not matched:
            best_len = 0
            for attr in candidates:
                for alias in attr.aliases:
                    a_norm = _normalize(alias)
                    if len(a_norm) > best_len and _alias_partial_hit(a_norm, col_norm):
                        matched, best_len = attr, len(a_norm)

        result[col] = matched

    return result


def summarize(columns: list[str], already_flagged: Optional[list[str]] = None) -> dict:
    """
    컬럼 목록에 대한 EE 매칭 요약 통계 반환.

    Parameters
    ----------
    already_flagged : list[str] | None
        상위 파이프라인에서 이미 탐지된 법정 민감정보 컬럼명 목록.
        해당 컬럼은 EE 매칭을 건너뛰고 skipped_legal로 별도 집계.

    Returns
    -------
    dict with keys:
        - total: 전체 컬럼 수
        - matched_ee: EE 매칭 컬럼 수
        - matched_er: ER 매칭 컬럼 수
        - skipped_legal: 법정 탐지로 스킵된 컬럼 수
        - unmatched: 미매칭 컬럼 수
        - matched_columns: {컬럼명: attr 정보} 딕셔너리
        - unmatched_columns: 미매칭 컬럼명 목록
        - skipped_columns: 스킵된 컬럼명 목록
    """
    flagged_norms: set[str] = (
        {_normalize(c) for c in already_flagged} if already_flagged else set()
    )

    all_match = match_columns(columns, ee_only=False, already_flagged=already_flagged)
    ee_match  = {c: a for c, a in all_match.items() if a and a.is_ee}
    er_match  = {c: a for c, a in all_match.items() if a and not a.is_ee}
    skipped   = [c for c in columns if _normalize(c) in flagged_norms]
    unmatched = [c for c, a in all_match.items()
                 if a is None and _normalize(c) not in flagged_norms]

    return {
        "total":           len(columns),
        "matched_ee":      len(ee_match),
        "matched_er":      len(er_match),
        "skipped_legal":   len(skipped),
        "unmatched":       len(unmatched),
        "matched_columns": {c: a.to_dict() for c, a in all_match.items() if a},
        "unmatched_columns": unmatched,
        "skipped_columns": skipped,
    }


if __name__ == "__main__":
    # 상위 파이프라인(법정 민감정보 탐지 레이어)에서 탐지된 결과라고 가정
    flagged_by_legal_detector = ["religion", "diagnosis_code"]

    test_cols = [
        "age", "gender", "zipcode", "diagnosis_code", "income",
        "user_id", "email", "latitude", "job_title", "blood_type",
        "sleep_hours", "religion", "hobby", "posting_time"
    ]

    print("=" * 60)
    print("EE 사전 매칭 테스트")
    print("=" * 60)

    summary = summarize(test_cols, already_flagged=flagged_by_legal_detector)
    print(f"\n총 컬럼      : {summary['total']}개")
    print(f"EE 매칭      : {summary['matched_ee']}개")
    print(f"ER 매칭      : {summary['matched_er']}개")
    print(f"법정 스킵    : {summary['skipped_legal']}개  ← {summary['skipped_columns']}")
    print(f"미매칭       : {summary['unmatched']}개")

    print("\n[매칭 결과]")
    results = match_columns(test_cols, ee_only=False,
                            already_flagged=flagged_by_legal_detector)
    for col, attr in results.items():
        if col in summary["skipped_columns"]:
            print(f"  {col:20s} → [법정 스킵]")
        elif attr:
            print(f"  {col:20s} → [{attr.ee_type}] {attr.attr_ko} (점수: {attr.ee_score})")
        else:
            print(f"  {col:20s} → 미매칭")

    print(f"\n전체 EE 속성 수: {len(get_ee_attributes())}개")
    print(f"전체 ER 속성 수: {len(get_er_attributes())}개")
