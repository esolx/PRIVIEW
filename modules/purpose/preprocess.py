"""
preprocess.py
추출된 문서 텍스트를 판정에 사용할 수 있는 형태로 다듬고,
판정에 사용할 만한 내용을 담고 있는지 검증하는 모듈.
PDF 입력과 직접 입력 텍스트 모두에 공통으로 적용된다.
"""

import re

from konlpy.tag import Okt


# 형태소 분석기는 한 번만 생성하여 재사용한다
okt = Okt()

# 의미 있는 문서로 인정하는 최소 토큰 수
MIN_MEANINGFUL_TOKENS = 20


def clean_text(text):
    """
    추출한 텍스트를 정리한다.
    문서 레이아웃 때문에 생긴 줄바꿈과 중복 공백을 제거하여
    문서마다 판정 결과가 달라지지 않도록 한다.

    입력:
        text (str): 원본 텍스트
    출력:
        str: 정리된 텍스트
    """

    # 줄바꿈을 공백으로 바꾼다
    text = text.replace("\n", " ")

    # 연속된 공백을 하나로 합친다
    text = re.sub(r"\s+", " ", text)

    # 앞뒤 공백을 제거한다
    text = text.strip()

    return text


def is_meaningful(text, min_tokens=MIN_MEANINGFUL_TOKENS):
    """
    문서 텍스트가 판정에 사용할 만한 내용을 담고 있는지 확인한다.
    명사, 동사, 형용사의 개수를 세어 기준 미만이면 무의미한 입력으로 본다.

    입력:
        text (str): 정리된 문서 텍스트
        min_tokens (int): 의미 있는 문서로 인정할 최소 토큰 수
    출력:
        tuple: (판별 결과 bool, 실제 토큰 수 int)
    """

    # 형태소를 품사와 함께 분석한다
    tagged = okt.pos(text)

    # 명사, 동사, 형용사만 추린다
    meaningful = [word for word, pos in tagged
                  if pos in ("Noun", "Verb", "Adjective")]

    count = len(meaningful)

    return count >= min_tokens, count


# 부정 표현으로 볼 어미와 표현
NEGATION_WORDS = ["않", "안 ", "없", "미사용", "제외", "금지", "배제", "불가", "지양"]


def split_sentences(text):
    """
    텍스트를 문장 단위로 분리한다.

    입력:
        text (str): 정리된 텍스트
    출력:
        list: 문장 리스트
    """

    # 마침표, 물음표, 느낌표를 기준으로 나눈다
    sentences = re.split(r"[.!?]\s*", text)

    # 빈 문자열은 제외한다
    return [s.strip() for s in sentences if s.strip()]


def is_negated(text, keyword):
    """
    키워드가 부정 표현과 함께 사용되었는지 확인한다.
    키워드가 포함된 문장 안에 부정 표현이 있으면 부정으로 판단한다.

    입력:
        text (str): 정리된 텍스트
        keyword (str): 확인할 키워드
    출력:
        bool: 부정 표현과 함께 쓰였으면 True
    """

    for sentence in split_sentences(text):

        # 키워드가 없는 문장은 건너뛴다
        if keyword not in sentence:
            continue

        # 같은 문장 안에 부정 표현이 있는지 확인한다
        for negation in NEGATION_WORDS:
            if negation in sentence:
                return True

    return False