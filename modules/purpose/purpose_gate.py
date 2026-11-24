"""
purpose_gate.py
목적외사용 판정의 최종 게이트.
문서 텍스트와 사용자가 선택한 목적을 받아
통과 또는 반려 여부와 그 근거를 반환한다.
다른 모듈에서 이 파일의 purpose_gate 함수를 호출하여 사용한다.
"""

from .preprocess import clean_text, is_meaningful
from .keyword_match import keyword_match
from .rules import RULES


def make_result(통과여부, 반려사유=None, 미충족항목=None, 위험키워드=None):
    """
    판정 결과 딕셔너리를 생성한다.
    반려 관련 값이 전달되지 않으면 빈 값으로 채워
    통과와 반려 모두 동일한 형태를 유지한다.

    입력:
        통과여부 (bool): 게이트 통과 여부
        반려사유 (list): 반려 사유 목록
        미충족항목 (list): 충족하지 못한 항목
        위험키워드 (dict): 분류별 탐지된 금지 키워드
    출력:
        dict: 판정 결과
    """

    return {
        "통과여부": 통과여부,
        "반려사유": 반려사유 or [],
        "미충족항목": 미충족항목 or [],
        "위험키워드": 위험키워드 or {},
    }


def purpose_gate(purpose, text):
    """
    목적외사용 여부를 판정한다.

    입력:
        purpose (str): 사용자가 선택한 가명처리 목적
        text (str): 목적 설명 문서에서 추출한 텍스트
    출력:
        dict: 판정 결과 (make_result 형태)
    """

    # 0단계: 허용 목적을 선택했는지 확인
    if purpose not in RULES:
        return make_result(False, ["허용 목적이 선택되지 않음"])

    # 문서 텍스트를 정리한다
    text = clean_text(text)

    # 1단계 준비: 판정에 사용할 수 있는 문서인지 확인
    valid, token_count = is_meaningful(text)
    if not valid:
        return make_result(False, ["문서 내용이 부족하여 판정 불가"])

    # 1단계: 키워드 매칭 (필요어/금지어 + 전역 금지 키워드)
    #         부정문 판별은 규칙과 LLM을 함께 사용한다 (2단계 LLM 확인)
    #         하나라도 걸리면 반려하며, 걸린 항목을 모두 모아서 보여준다
    결과 = keyword_match(purpose, text)
    미충족항목 = 결과["미충족항목"]
    위험키워드 = 결과["위험키워드"]

    반려사유 = []
    if 미충족항목:
        반려사유.append("필요 요소 미충족")
    if 위험키워드:
        반려사유.append("금지 키워드 탐지")

    # 반려 사유가 하나라도 있으면 반려
    if 반려사유:
        return make_result(False, 반려사유, 미충족항목, 위험키워드)

    # 모두 통과
    return make_result(True)