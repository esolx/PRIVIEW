"""
column_check.py
데이터셋의 컬럼을 검사하여 목적과 상관없는 민감 컬럼을 골라내는 모듈.
실제 제거는 하지 않고 대상 목록만 반환한다. (2단계 가명처리에서 사용)
"""

from .sensitive_columns import SENSITIVE_KEYWORDS, PURPOSE_ALLOWED_SENSITIVE


def find_sensitive_columns(columns):
    """
    컬럼 목록에서 민감 컬럼을 찾는다.
    컬럼 이름에 민감 키워드가 포함되어 있으면 민감 컬럼으로 본다.

    입력:
        columns (list): 컬럼 이름 리스트
    출력:
        list: 민감 컬럼 이름 리스트
    """

    sensitive = []
    for column in columns:
        for keyword in SENSITIVE_KEYWORDS:
            if keyword in column:
                sensitive.append(column)
                break
    return sensitive


def find_pseudonym_targets(purpose, columns):
    """
    목적과 상관없는 민감 컬럼을 가명처리 대상으로 골라낸다.
    민감 컬럼이면서 해당 목적에 필요하지 않은 컬럼만 대상으로 한다.

    입력:
        purpose (str): 선택한 목적
        columns (list): 데이터셋 컬럼 이름 리스트
    출력:
        list: 가명처리 대상 컬럼 이름 리스트
    """

    sensitive = find_sensitive_columns(columns)
    allowed = PURPOSE_ALLOWED_SENSITIVE.get(purpose, [])

    targets = []
    for column in sensitive:
        keep = False
        for keyword in allowed:
            if keyword in column:
                keep = True
                break
        if not keep:
            targets.append(column)

    return targets