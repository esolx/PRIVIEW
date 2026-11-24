"""
llm_check.py
LLM(OpenAI GPT)을 이용해 금지 키워드가 실제로 문제되는 맥락에서
쓰였는지 판단하는 모듈.
두 가지를 함께 확인한다.
  1. 부정문 여부 ("사용하지 않는다" 등)
  2. 의도 여부 (실제 그 용도인가, 단순히 연구·통계의 주제인가)
규칙 기반 매칭의 한계(문맥·의도 파악 불가)를 보완한다.
API 키가 없으면 판단하지 않고, 규칙 기반 결과를 그대로 사용한다.
"""

import os

from openai import OpenAI


# API 키는 환경 변수에서 읽어온다 (코드에 직접 쓰지 않는다)
API_KEY = os.environ.get("OPENAI_API_KEY")

# API 키가 있을 때만 클라이언트를 생성한다
client = OpenAI(api_key=API_KEY) if API_KEY else None


def is_safe_context_by_llm(text, keyword):
    """
    LLM에게 키워드가 '안전한 맥락'에서 쓰였는지 판단하게 한다.
    부정문이거나, 실제 사용 의도가 아니라 연구·통계의 주제일 경우
    안전한 맥락으로 보아 True를 반환한다.

    입력:
        text (str): 계획서 텍스트
        keyword (str): 확인할 금지 키워드
    출력:
        bool: 안전한 맥락이면 True (금지어로 보지 않음)
    """

    # API 키가 없으면 판단하지 않고 False를 반환한다
    # (규칙 기반 결과만 사용하게 되어, 규칙만 있을 때와 동일하게 동작한다)
    if client is None:
        return False

    prompt = (
        f"다음은 가명정보 활용 계획서의 일부입니다.\n"
        f"이 계획서에서 '{keyword}'라는 단어가 아래 중 어디에 해당하는지 판단해주세요.\n\n"
        f"(A) 실제로 그 행위를 하겠다는 의도로 쓰임 "
        f"(예: 이 데이터로 개인을 평가하거나 마케팅에 활용하겠다)\n"
        f"(B) 부정문으로 쓰임 (예: 그 용도로 사용하지 않겠다)\n"
        f"(C) 연구나 통계의 주제·대상으로만 언급됨 "
        f"(예: 평가 제도나 마케팅 현상 자체를 분석하는 연구)\n\n"
        f"(A)이면 '위험', (B)나 (C)이면 '안전'이라고만 답해주세요.\n\n"
        f"계획서: {text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        return "안전" in answer
    except Exception as e:
        # API 호출에 실패하면 안전하게 규칙 기반 결과를 따른다
        print("LLM 호출 실패:", e)
        return False