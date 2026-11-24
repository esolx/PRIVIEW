"""
keyword_match.py
1단계 키워드 매칭.
목적별 규칙(RULES)의 필요어·금지어와 전역 금지 키워드(DANGER_KEYWORDS)를
형태소 매칭으로 한 번에 검사한다.
부정문·맥락 판별은 규칙(is_negated)과 LLM(is_safe_context_by_llm)을 함께 사용한다.
LLM은 키워드가 실제로 텍스트에 있을 때만 호출하여 비용을 아낀다.
"""

from konlpy.tag import Okt

from .rules import RULES, DANGER_KEYWORDS
from .preprocess import is_negated
from .llm_check import is_safe_context_by_llm


okt = Okt()


def _matched(keyword, tokens, text_no_space):
    """
    키워드가 텍스트에 있는지 형태소 기준으로 확인한다.
    단일 단어는 형태소 토큰에서, 복합어는 붙여쓴 원문에서 찾는다.

    입력:
        keyword (str): 찾을 키워드
        tokens (list): 형태소 토큰 리스트
        text_no_space (str): 띄어쓰기를 제거한 원문
    출력:
        bool: 발견되면 True
    """
    # 단일 단어는 형태소 토큰에 정확히 일치할 때만 (부분 일치 오탐 방지)
    if keyword in tokens:
        return True
    # 복합어(공백/여러 글자)는 붙여쓴 원문에서 확인
    if keyword.replace(" ", "") in text_no_space:
        return True
    return False


def _is_real_hit(text, keyword):
    """
    매칭된 키워드가 실제 문제되는 맥락인지 확인한다.
    부정문이거나(규칙) 안전한 맥락이면(LLM) 문제 없는 것으로 본다.

    입력:
        text (str): 정리된 텍스트
        keyword (str): 매칭된 키워드
    출력:
        bool: 실제로 문제되는 맥락이면 True
    """
    if is_negated(text, keyword):
        return False
    if is_safe_context_by_llm(text, keyword):
        return False
    return True


def keyword_match(purpose, text):
    """
    1단계 키워드 매칭으로 목적 적합성을 검사한다.

    입력:
        purpose (str): 선택한 목적
        text (str): 목적 설명 문서의 텍스트
    출력:
        dict: {"미충족항목": list, "위험키워드": dict}
    """

    tokens = okt.morphs(text)
    text_no_space = text.replace(" ", "")

    미충족항목 = []
    위험키워드 = {}

    # --- 목적별 규칙 검사 (필요어 / 금지어) ---
    for rule in RULES[purpose]:

        # 금지어: 매칭되고 실제 문제 맥락이면 미충족
        if "금지어" in rule:
            hit = [w for w in rule["금지어"]
                   if _matched(w, tokens, text_no_space)
                   and _is_real_hit(text, w)]
            if hit:
                미충족항목.append(rule["항목"])

        # 필요어: 하나도 없으면 미충족
        if "필요어" in rule:
            found = [w for w in rule["필요어"]
                     if _matched(w, tokens, text_no_space)]
            if not found:
                미충족항목.append(rule["항목"])

    # --- 전역 금지 키워드 검사 ---
    for category, keywords in DANGER_KEYWORDS.items():
        found = []
        for keyword in keywords:
            # 먼저 매칭 확인 (텍스트에 없으면 LLM을 부를 필요 없음)
            if not _matched(keyword, tokens, text_no_space):
                continue
            # 매칭된 것만 부정문, 맥락 판별
            if is_negated(text, keyword) or is_safe_context_by_llm(text, keyword):
                continue
            found.append(keyword)
        if found:
            위험키워드[category] = found

    return {"미충족항목": 미충족항목, "위험키워드": 위험키워드}