# -*- coding: utf-8 -*-
"""
llm_labeler_openai.py — OpenAI GPT API로 컬럼 프로파일에 카테고리 초벌 라벨을
                        붙이는 드라이버 (전체 라벨링 버전).

Claude 버전(llm_labeler.py)과 인터페이스·출력 포맷이 동일하다.
어느 쪽으로 라벨링하든 이후 흐름(검수 → apply_feedback 재학습)은 그대로 이어진다.

역할: 정답이 없는 상태에서 각 컬럼(프로파일 텍스트)을 GPT에게 보내
      config의 19개 코드(18 민감 + NON_SENSITIVE) 중 하나로 강제 분류하고,
      신뢰도까지 함께 받아 학습용 (텍스트, 라벨, 신뢰도) 쌍을 만든다.

핵심 설계:
    · function calling + strict:true → 출력이 반드시 {category: enum, confidence}
      스키마를 따름 (LLM이 멋대로 라벨 만드는 것 방지 = 라벨 노이즈 차단)
    · n-shot 예시 고정 → 모든 입력에 같은 예시 재사용(논문 방식)
    · 전체 라벨링 → label_profiles가 모든 컬럼을 순차 처리
    · 배치 모드 지원 → 대량이면 Batch API(.jsonl)로 50% 절감

사용법:
    from exposure_pipeline.llm_labeler_openai import OpenAILabeler
    from exposure_pipeline.features import profile_dataframe
    import pandas as pd

    labeler = OpenAILabeler(model="gpt-4o-mini")
    profiles = profile_dataframe(pd.read_csv("data.csv"))
    results = labeler.label_profiles(profiles)      # 전체 라벨링
    labeler.to_csv(results, "llm_labels.csv")

주의: OPENAI_API_KEY 환경변수가 필요하다. (pip install openai)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .features import profile_to_text
from .config import CATEGORIES, NEGATIVE_CODE, SENSITIVE_CODES, LEGAL_CODES

# --- 라벨 공간: config에서 자동 유도 (18 민감 + NON_SENSITIVE) ---
LABEL_CODES = list(CATEGORIES.keys())   # 19개


def _build_category_definitions() -> str:
    """카테고리 정의 텍스트. 모든 프롬프트에 동일하게 들어간다."""
    lines = ["다음은 컬럼을 분류할 카테고리 정의다:\n"]
    for code, cat in CATEGORIES.items():
        ko = getattr(cat, "name_ko", "")
        en = getattr(cat, "name_en", "")
        kws = ", ".join(list(getattr(cat, "keywords", ()))[:8])
        lines.append(f"- {code} ({ko} / {en}): 예시 키워드 = {kws}")
    return "\n".join(lines)


# --- n-shot 예시 (고정). 대표 오탐 유형을 예시로 넣어 경계 학습 유도 ---
FEWSHOT_EXAMPLES = [
    ("Patient ID Patient ID Patient ID P00001 P00002 P00003 (전부 고유 식별자)",
     "NON_SENSITIVE"),
    ("Heart Rate Heart Rate Heart Rate 72 88 95 101 (심박수 수치)",
     "HEALTH"),
    ("Diagnosis Diagnosis Diagnosis diabetes asthma hypertension",
     "HEALTH"),
    ("region code region code region code 1 2 17 23 (지역 코드값)",
     "LOCATION"),
    ("email email email user1@ex.com user2@ex.com (이메일 주소)",
     "COMMUNICATION"),
]


SYSTEM_INSTRUCTION = (
    "너는 데이터 컬럼 분류기다. 주어진 컬럼의 이름과 표본 값을 보고, "
    "그 컬럼이 담는 정보를 카테고리 중 정확히 하나로 분류한다. "
    "반드시 label_column 함수를 호출해 답하라. "
    "중요 규칙: (1) 식별자(_id, _key, 일련번호 등)는 그 자체로는 NON_SENSITIVE다. "
    "(2) 컬럼명이 민감어라도 값이 무의미한 코드/정수면 값 근거를 우선하라. "
    "(3) 확신이 낮으면 confidence를 낮게 매겨라 — 이후 사람이 그 컬럼을 검수한다."
)


def _classify_function() -> Dict:
    """OpenAI function calling 스키마 (strict 모드). enum으로 19개 코드 강제."""
    return {
        "type": "function",
        "function": {
            "name": "label_column",
            "description": "컬럼을 정확히 하나의 카테고리로 분류하고 신뢰도를 매긴다.",
            "strict": True,   # ★ 스키마 100% 준수 강제
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": LABEL_CODES,   # ★ 19개 코드로 강제
                        "description": "컬럼이 담는 정보의 카테고리",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "이 분류에 대한 확신도(0~1)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "한 줄 근거 (컬럼명 기준인지 값 기준인지 명시)",
                    },
                },
                # strict 모드는 모든 property가 required 여야 한다
                "required": ["category", "confidence", "reason"],
            },
        },
    }


def _build_messages(cat_defs: str, fewshot: str, text: str) -> List[Dict]:
    """system(정의+예시) + user(대상 컬럼) 메시지 구성.

    정의·예시를 프롬프트 앞쪽 고정 프리픽스로 두면 prompt caching 이득도 본다.
    """
    fewshot_block = "다음은 분류 예시다:\n" + "\n".join(
        f"컬럼: {t}\n→ {code}" for t, code in FEWSHOT_EXAMPLES
    )
    system = f"{SYSTEM_INSTRUCTION}\n\n{cat_defs}\n\n{fewshot_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"컬럼:\n{text}"},
    ]


@dataclass
class LabelResult:
    column: str
    category: str
    confidence: float
    reason: str
    text: str            # 라벨을 단 프로파일 텍스트 (재학습 입력과 동일해야 함)


class OpenAILabeler:
    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None, max_retries: int = 5):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai 필요")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=max_retries,
        )
        self.model = model
        self._cat_defs = _build_category_definitions()
        self._fewshot = ""  # _build_messages 안에서 렌더

    # --- 단건 라벨링 ---
    def label_one(self, text: str, column: str = "") -> LabelResult:
        messages = _build_messages(self._cat_defs, self._fewshot, text)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[_classify_function()],
            tool_choice={"type": "function",
                         "function": {"name": "label_column"}},  # ★ 강제
            max_completion_tokens=256,
            temperature=0,
        )
        return self._parse(resp, text, column)

    def _parse(self, resp, text: str, column: str) -> LabelResult:
        cat, conf, reason = NEGATIVE_CODE, 0.0, ""
        msg = resp.choices[0].message
        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
            cat = args.get("category", NEGATIVE_CODE)
            conf = float(args.get("confidence", 0.0))
            reason = args.get("reason", "")
        return LabelResult(column=column, category=cat, confidence=conf,
                           reason=reason, text=text)

    # --- 전체 라벨링 ---
    def label_profiles(self, profiles) -> List[LabelResult]:
        """profile_dataframe 결과 리스트 전체를 순차 라벨링."""
        out = []
        for p in profiles:
            text = profile_to_text(p)
            out.append(self.label_one(text, column=p.name))
        return out

    # --- 대량 배치 라벨링 (Batch API, 50% 저렴) ---
    def write_batch_jsonl(self, profiles, path: str = "batch_requests.jsonl") -> str:
        """Batch API 제출용 .jsonl 파일 생성.

        제출:  batch_input = client.files.create(file=open(path,'rb'), purpose='batch')
               client.batches.create(input_file_id=batch_input.id,
                                      endpoint='/v1/chat/completions',
                                      completion_window='24h')
        """
        tool = _classify_function()
        with open(path, "w", encoding="utf-8") as f:
            for i, p in enumerate(profiles):
                text = profile_to_text(p)
                messages = _build_messages(self._cat_defs, self._fewshot, text)
                line = {
                    "custom_id": f"col-{i}::{p.name}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "tools": [tool],
                        "tool_choice": {"type": "function",
                                        "function": {"name": "label_column"}},
                        "max_tokens": 256,
                        "temperature": 0,
                    },
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return path

    def parse_batch_output(self, jsonl_path: str) -> List[LabelResult]:
        """Batch API 결과 .jsonl을 LabelResult 리스트로 파싱."""
        out = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                custom_id = rec.get("custom_id", "")
                column = custom_id.split("::", 1)[1] if "::" in custom_id else custom_id
                cat, conf, reason = NEGATIVE_CODE, 0.0, ""
                try:
                    body = rec["response"]["body"]
                    msg = body["choices"][0]["message"]
                    if msg.get("tool_calls"):
                        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
                        cat = args.get("category", NEGATIVE_CODE)
                        conf = float(args.get("confidence", 0.0))
                        reason = args.get("reason", "")
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
                out.append(LabelResult(column=column, category=cat,
                                       confidence=conf, reason=reason, text=""))
        return out

    # --- 출력 저장 (Claude 버전과 동일 포맷) ---
    def to_csv(self, results: Sequence[LabelResult], path: str):
        import pandas as pd
        rows = [{
            "column": r.column,
            "llm_category": r.category,
            "llm_confidence": round(r.confidence, 3),
            "is_sensitive": r.category in SENSITIVE_CODES,
            "is_legal": r.category in LEGAL_CODES,
            "reason": r.reason,
            "TRUE_LABEL": "",   # 검수자가 확인/수정하는 칸
        } for r in results]
        df = pd.DataFrame(rows).sort_values("llm_confidence")  # 저신뢰 먼저 검수
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df

    def to_corrections(self, results: Sequence[LabelResult],
                       min_confidence: float = 0.0) -> Dict[str, str]:
        """{컬럼: 카테고리} 딕셔너리. apply_feedback에 바로 넣을 수 있다."""
        return {r.column: r.category for r in results
                if r.confidence >= min_confidence}
