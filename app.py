# -*- coding: utf-8 -*-
"""
가명정보 위험 분석 통합 대시보드 - 백엔드.

파이프라인 순서(게이트 우선):
  1) 목적 외 사용 게이트(사전판정)  →  반려 시 이후 단계 전면 중단
  2) 통과 시 재식별 위험, 집계 추론 위험 및 노출 위험 분석
  3) 세 위험 모듈의 데이터셋 등급 중 최고 등급으로 통합 위험도 산출
  4) 적정성 검토서(.docx) 생성
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import warnings

# scikit-learn 1.8이 Windows 병렬 작업 프로세스에서 매번 찍는 안내성 경고 (결과와 무관).
# 작업 프로세스는 따로 뜨므로 환경변수로도 넘겨 준다.
_SKLEARN_PARALLEL_WARNING = "`sklearn.utils.parallel.delayed` should be used"
warnings.filterwarnings("ignore", message=_SKLEARN_PARALLEL_WARNING, category=UserWarning)
_pw = f"ignore:{_SKLEARN_PARALLEL_WARNING}:UserWarning"
if _pw not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(x for x in (os.environ.get("PYTHONWARNINGS"), _pw) if x)
# Windows의 joblib 작업 프로세스는 부모의 -W 옵션(sys.warnoptions)을 넘겨받아 시작하므로 여기에도 넣는다
if _pw not in sys.warnoptions:
    sys.warnoptions.append(_pw)

import pandas as pd

# Windows 콘솔이 출력하지 못하는 문자(√, ✓ 등)가 로그에 있어도 서버가 멈추지 않도록
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# matplotlib 한글 폰트를 프로세스 전역으로 먼저 설정(모든 모듈 시각화에 적용)
from modules.common import mpl_korean
mpl_korean.setup()

from modules.integration.final_risk import compute_integrated_risk_from_modules
from modules.integration.id_plots import generate_identification_plots
from modules.integration.adequacy_report import generate_adequacy_report
from modules.repseudonymization import METHOD_LABELS, process_dataframe, suggest_method

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

PURPOSE_LABEL_TO_CODE = {
    "통계 작성": "통계작성",
    "과학적 연구": "과학적연구",
    "공익적 기록보존": "공익적기록보존",
}
GRADE_KO = {"H": "고위험", "M": "중위험", "L": "저위험"}

# 집계 모듈이 생성하는 PNG 파일의 제목/설명(대시보드 실험 시각화 탭에 노출)
AGG_PLOT_META = {
    "risk_distribution_pie.png": ("최종 위험도 분포", "조합별 최종 위험 등급의 비율입니다."),
    "risk_accuracy_distribution.png": ("위험도와 정확도 분포", "조합별 평균 정확도 분포와 위험 구간을 함께 나타냅니다."),
    "top20_high_risk_combinations.png": ("상위 고위험 조합 20", "추론 정확도가 높은 상위 조합을 정렬해 보여줍니다."),
    "model_accuracy_boxplot.png": ("모델별 조합 정확도 분포", "10종 분류기의 조합별 정확도 분포(박스플롯)입니다."),
}

EXPOSURE_GRADE_TO_CODE = {"High": "H", "Middle": "M", "Medium": "M", "Low": "L"}
# 최신 노출 카테고리의 표시 이름을 그대로 사용한다. 화면 구성은 유지한다.
from exposure_pipeline.config import CATEGORIES as EXPOSURE_CATEGORIES
EXPOSURE_CATEGORY_KO = {code: category.name_ko for code, category in EXPOSURE_CATEGORIES.items()}



# ────────────────────────────── 유틸 ──────────────────────────────
def _save_upload(file_storage, job_dir: Path, fallback_name: str) -> Path:
    name = secure_filename(file_storage.filename or fallback_name) or fallback_name
    path = job_dir / name
    file_storage.save(path)
    return path


def _read_text_table(path: Path, sep: str = ",") -> pd.DataFrame:
    """CSV/TSV 읽기. 한글 Excel이 저장한 CP949 파일과 BOM 붙은 UTF-8도 읽는다."""
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949(한글 Excel) CSV로 저장해 주세요.") from last_error


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_text_table(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return _read_text_table(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("지원하지 않는 데이터 형식입니다. CSV, XLSX, XLS, TSV, JSON, Parquet를 사용해 주세요.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(v) for v in value.to_dict("records")]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and value != value:   # NaN (노출 v9.6 확정·비적용) → null
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


def _set_progress(out_dir: Path, stage: str, title: str, description: str,
                  state: str = "running") -> None:
    """현재 실행 중인 실제 분석 단계를 화면 조회용 파일에 기록한다."""
    payload = {
        "stage": stage,
        "title": title,
        "description": description,
        "state": state,
    }
    # 화면이 1초마다 이 파일을 읽으므로, 임시 파일에 다 쓴 뒤 한 번에 바꿔 끼운다.
    # (그대로 덮어쓰면 읽는 순간 빈 파일·반쪽 파일을 읽어 진행 조회가 500 오류가 남)
    target = out_dir / "progress.json"
    tmp = out_dir / f"progress.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:      # Windows: 화면이 마침 파일을 읽는 중이면 잠깐 기다렸다 다시
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)      # 진행 표시는 다음 단계에서 다시 갱신되므로 이번 것은 건너뜀


def _read_progress(path: Path):
    """진행 파일 읽기. 바꿔 끼우는 순간과 겹치면 잠깐 뒤 다시 읽고, 그래도 안 되면 None."""
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError, FileNotFoundError):
            time.sleep(0.05)
    return None


def _progress_response(path: Path, missing_message: str):
    if not path.exists():
        return jsonify({"ok": False, "message": missing_message}), 404
    payload = _read_progress(path)
    if payload is None:   # 작업은 계속 진행 중 — 이번 조회만 건너뜀 (화면은 다음 조회에서 갱신)
        return jsonify({"ok": False, "message": "진행 상태를 갱신하고 있습니다."}), 202
    return jsonify({"ok": True, **payload})


def _purpose_display_details(purpose_code: str, plan_path: Path,
                             purpose_result: dict) -> dict:
    """판정값은 바꾸지 않고 계획서에서 확인한 근거를 화면 표시용으로 정리한다."""
    from modules.purpose.document_reader import extract_text_from_pdf
    from modules.purpose.preprocess import clean_text, is_meaningful, okt
    from modules.purpose.rules import RULES
    from modules.purpose.llm_check import API_KEY, client

    raw_text = extract_text_from_pdf(str(plan_path))
    text = clean_text(raw_text)
    _, token_count = is_meaningful(text)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    unmet = set(purpose_result.get("미충족항목", []))
    danger = purpose_result.get("위험키워드", {}) or {}
    title_map = {
        "특정 개인 식별·추적 목적이 아닌가": "특정 개인 식별이나 추적 목적이 아닌가",
        "1:1 타깃 마케팅·결정에 직접 사용되는가": "개인별 타깃 마케팅이나 의사결정에 직접 사용되는가",
        "검증·분석 방법이 명시되어 있는가": "검증 및 분석 방법이 명시되어 있는가",
    }
    used_lines: set[int] = set()

    def evidence_for(criterion: str, keywords: list[str]) -> str:
        nouns = [word for word in okt.nouns(criterion) if len(word) >= 2]
        terms = list(dict.fromkeys(keywords + nouns))
        candidates = []
        for index, line in enumerate(lines):
            compact = line.replace(" ", "")
            score = sum(3 for word in keywords if word.replace(" ", "") in compact)
            score += sum(1 for word in nouns if word.replace(" ", "") in compact)
            if score:
                candidates.append((score, index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        index = next((idx for _score, idx in candidates if idx not in used_lines), None)
        if index is None:
            purpose_terms = ["목적", "활용", "분석", "연구", "통계", "보존", "방법"]
            index = next((idx for idx, line in enumerate(lines)
                          if idx not in used_lines and any(term in line for term in purpose_terms)), None)
        if index is None:
            index = next((idx for idx in range(len(lines)) if idx not in used_lines), None)
        if index is None:
            return "계획서에서 관련 내용을 확인하지 못했습니다."
        used_lines.add(index)
        selected = [lines[index]]
        if len(lines[index]) < 90 and index + 1 < len(lines) and index + 1 not in used_lines:
            selected.append(lines[index + 1])
            used_lines.add(index + 1)
        return "\n".join(selected)[:360]

    criteria = []
    for rule in RULES.get(purpose_code, []):
        keywords = list(rule.get("필요어", [])) + list(rule.get("금지어", []))
        criterion = rule.get("항목", "")
        criteria.append({
            "criterion": criterion,
            "display_title": title_map.get(criterion, criterion.replace("·", " 및 ")),
            "status": "미충족" if criterion in unmet else "확인 완료",
            "evidence": evidence_for(criterion, keywords),
        })

    rule_count = len(criteria)
    satisfied = max(0, rule_count - len(unmet))
    llm_active = bool(API_KEY and client is not None)
    return {
        "criteria": criteria,
        "analysis_summary": {
            "형태소수": token_count,
            "검토항목수": rule_count,
            "충족항목수": satisfied,
            "충족률": round((satisfied / rule_count * 100) if rule_count else 100.0, 1),
            "위험표현수": sum(len(values) for values in danger.values()),
            "LLM사용": llm_active,
        },
        "llm": {
            "api_key_configured": bool(API_KEY),
            "api_call_active": llm_active,
            "mode": "규칙 및 OpenAI GPT" if llm_active else "규칙 기반 판정",
            "description": (
                "OpenAI GPT가 규칙에서 확인된 표현의 부정 문맥과 실제 사용 의도를 추가로 판별했습니다."
                if llm_active else
                "OPENAI_API_KEY가 설정되지 않아 규칙 기반 판정만 적용했습니다."
            ),
        },
    }


def _find_job_dataset(job_id: str) -> Path:
    if not job_id.isalnum():
        raise ValueError("작업 ID를 확인해 주세요.")
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise ValueError("위험 평가에 사용한 데이터셋을 찾을 수 없습니다.")
    analysis_copy = job_dir / "dataset_for_analysis.csv"
    if analysis_copy.exists():
        return analysis_copy
    candidates = [path for path in job_dir.iterdir()
                  if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet"}
                  and "for_purpose" not in path.name and "for_aggregation" not in path.name]
    if not candidates:
        raise ValueError("위험 평가에 사용한 데이터셋을 찾을 수 없습니다.")
    return sorted(candidates)[0]


def _load_result_with_current_integration(job_id: str) -> dict:
    path = RESULT_DIR / job_id / "integrated_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    old = result.get("integrated") or {}
    if old and old.get("score_kind") != "ordinal_grade":
        df = _load_dataframe(_find_job_dataset(job_id))
        result["integrated"] = compute_integrated_risk_from_modules(result.get("modules") or {}, df)
        result.setdefault("warnings", []).append(
            "이전 통합식으로 저장된 결과를 현재 등급 통합 기준으로 다시 표시했습니다. "
            "기존에 출력한 검토서는 재생성되지 않으므로 새 분석 후 내려받아 주세요.")
        result["report"] = {"available": False, "docx_url": None}
    return result


def _build_sensitive_detection_pipeline():
    """노출 위험 모듈과 동일한 민감정보 판정 파이프라인을 구성한다."""
    from exposure_pipeline.runtime import build_detector

    module_dir = BASE_DIR / "exposure_pipeline"
    pipeline, _, _ = build_detector(
        model_path=str(module_dir / "rf_model.joblib"),
        calibration_path=str(module_dir / "calibration_holdout.csv"),
        lang="both", learned_dir=str(module_dir / "learned"),
    )
    return pipeline


def _build_risk_comparison(before: dict, after: dict, processing_summary: list[dict]) -> dict:
    before_integrated = before.get("integrated") or {}
    after_integrated = after.get("integrated") or {}
    before_components = {row.get("module"): row for row in before_integrated.get("components", [])}
    after_components = {row.get("module"): row for row in after_integrated.get("components", [])}
    module_rows, attribute_rows = [], []
    for module in ("identification", "aggregation", "exposure"):
        old = before_components.get(module, {})
        new = after_components.get(module, {})
        old_score = old.get("score") if old else None
        new_score = new.get("score") if new else None
        reduction = old_score - new_score if old_score is not None and new_score is not None else None
        module_rows.append({
            "module": module,
            "label": old.get("label") or new.get("label") or
                     {"identification": "식별 위험", "aggregation": "집계 위험", "exposure": "노출 위험"}.get(module, module),
            "before_grade": old.get("grade"), "after_grade": new.get("grade"),
            "before_score": round(old_score, 4) if old_score is not None else None,
            "after_score": round(new_score, 4) if new_score is not None else None,
            "reduction": round(reduction, 4) if reduction is not None else None,
            "before_attribute": old.get("top_attribute"), "after_attribute": new.get("top_attribute"),
            "comparable": reduction is not None,
        })
        old_attrs = {row.get("attribute"): row for row in old.get("attributes", [])}
        new_attrs = {row.get("attribute"): row for row in new.get("attributes", [])}
        for attribute in set(old_attrs) & set(new_attrs):
            old_row, new_row = old_attrs.get(attribute, {}), new_attrs.get(attribute, {})
            if old_row.get("metric_label") != new_row.get("metric_label"):
                continue
            if old_row.get("risk_score") is None or new_row.get("risk_score") is None:
                continue
            old_value = float(old_row["risk_score"])
            new_value = float(new_row["risk_score"])
            if old_value - new_value > 0.00005:
                attribute_rows.append({
                    "module": module, "module_label": old.get("label") or new.get("label") or module,
                    "attribute": attribute, "before_score": round(old_value, 4),
                    "after_score": round(new_value, 4), "reduction": round(old_value - new_value, 4),
                    "before_grade": old_row.get("grade"), "after_grade": new_row.get("grade"),
                    "metric_label": old_row["metric_label"],
                })
    before_score = before_integrated.get("score")
    after_score = after_integrated.get("score")
    before_complete = before_integrated.get("completeness", "complete") == "complete" and len(before_components) == 3
    after_complete = after_integrated.get("completeness", "complete") == "complete" and len(after_components) == 3
    comparable = before_complete and after_complete
    reduction = before_score - after_score if comparable and before_score is not None and after_score is not None else None
    increased_modules = [row["label"] for row in module_rows
                         if row["reduction"] is not None and row["reduction"] < 0]
    note = (f"{', '.join(increased_modules)}의 자체 판정 등급이 상승했습니다. "
            "속성별 수치 감소와 데이터셋 등급은 서로 다른 판단이므로 해당 모듈의 세부 결과를 확인해 주세요."
            if increased_modules else
            "속성별 수치와 데이터셋 등급은 기준이 다르므로 각각 확인해 주세요.")
    return {
        "before_grade": before_integrated.get("grade"), "after_grade": after_integrated.get("grade"),
        "before_grade_label": before_integrated.get("grade_label"),
        "after_grade_label": after_integrated.get("grade_label"),
        "before_score": before_score, "after_score": after_score,
        "absolute_reduction": round(reduction, 4) if reduction is not None else None,
        "improvement_pct": None,
        "improved": bool(reduction is not None and reduction > 0.00005),
        "comparable": comparable,
        "score_kind": "ordinal_grade",
        "comparison_note": note,
        "modules": module_rows,
        "improved_attributes": sorted(attribute_rows, key=lambda row: row["reduction"], reverse=True)[:20],
        "processing_summary": processing_summary,
    }


def _recommended_columns(job_id: str, df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    result_path = RESULT_DIR / job_id / "integrated_result.json"
    if not result_path.exists():
        return [], {}
    result = _load_result_with_current_integration(job_id)
    reasons: dict[str, str] = {}
    for component in (result.get("integrated") or {}).get("components", []):
        column = str(component.get("top_attribute", ""))
        if column in df.columns and component.get("grade") in {"M", "H"}:
            reasons[column] = (
                f"{component.get('label', '통합 위험')}의 최고위험 속성으로 "
                f"{component.get('grade_label', '위험')} 판정 확인됨")
    for column in (result.get("gate") or {}).get("pseudonym_targets", []):
        if column in df.columns:
            reasons[str(column)] = "목적 적합성 판정에서 가명처리 검토 대상으로 확인됨"
    identification = (result.get("modules") or {}).get("identification") or {}
    for row in identification.get("attributes", []):
        column = str(row.get("attribute", ""))
        if column in df.columns and row.get("final_grade") == "H":
            reasons[column] = f"식별 위험이 {GRADE_KO.get(row.get('final_grade'), '확인 필요')}으로 확인됨"
    aggregation = (result.get("modules") or {}).get("aggregation") or {}
    agg_grades = aggregation.get("attribute_grades") or {}
    agg_scores = aggregation.get("attribute_scores") or {}
    # [상위 절반 버전] 집계 고위험 속성 중 H 비율 상위 절반(올림)만 자동 추천
    high = sorted((c for c, g in agg_grades.items() if g == "H" and c in df.columns),
                  key=lambda c: -float(agg_scores.get(c) or 0))
    for column in high[:math.ceil(len(high) / 2)]:
        score = agg_scores.get(column)
        reasons[column] = ("집계 위험 H 비율 상위 절반"
                           + (f" (H 비율 {float(score) * 100:.0f}%)" if score is not None else ""))
    exposure = (result.get("modules") or {}).get("exposure") or {}
    for column in exposure.get("determined_attributes") or []:
        if str(column) in df.columns:
            reasons[str(column)] = "노출 지표가 확정 상태(유효값 1종 등)로 우선 검토가 필요한 컬럼"
    if exposure.get("dataset_grade") == "H":
        for row in exposure.get("top_actions", []):
            column = str(row.get("attribute", ""))
            if column in df.columns:
                reasons[column] = "노출 위험 임계를 초과하여 우선 처리가 필요한 컬럼"
    return list(reasons), reasons


def _repseudonymization_profile(df: pd.DataFrame, recommended: list[str], reasons: dict[str, str]) -> dict:
    return {
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "recommended_columns": recommended,
        "columns": [{
            "name": str(column),
            "dtype": str(df[column].dtype),
            "unique": int(df[column].nunique(dropna=True)),
            "missing_pct": round(float(df[column].isna().mean() * 100), 2),
            "recommended": str(column) in recommended,
            "reason": reasons.get(str(column), ""),
            "suggested_method": suggest_method(df[column], str(column)),
        } for column in df.columns],
        "methods": METHOD_LABELS,
    }


def _build_attribute_risk_report(result: dict, df: pd.DataFrame) -> dict:
    """1단계의 속성별 H/M/L 결과를 2단계 최종 파이프라인 입력으로 변환한다."""
    from modules.pseudonymization_pipeline import AttributeRisk, RiskLevel

    modules = result.get("modules") or {}
    id_grades = {
        str(row.get("attribute")): str(row.get("final_grade", "L"))
        for row in (modules.get("identification") or {}).get("attributes", [])
    }
    agg_grades = {
        str(column): str(grade)
        for column, grade in ((modules.get("aggregation") or {}).get("attribute_grades") or {}).items()
    }
    exposure = modules.get("exposure") or {}
    exposure_scores = exposure.get("attribute_scores") or {}
    exposure_components = exposure.get("risk_components") or {}
    exposure_determined = set(map(str, exposure.get("determined_attributes") or []))

    def level(value: str) -> RiskLevel:
        return RiskLevel(value if value in {"H", "M", "L"} else "L")

    def exposure_level(score: float) -> RiskLevel:
        score = float(score or 0)
        return RiskLevel.H if score >= 0.50 else RiskLevel.M if score >= 0.25 else RiskLevel.L

    report = {}
    matrix_by_attr = {
        str(row.get("attribute")): row
        for row in (exposure.get("exposure_matrix") or [])
    }
    def _score(value):
        """노출 v9.6: 확정·비적용 성분은 NaN/None — 수치가 아니면 None."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    for column in map(str, df.columns):
        component = exposure_components.get(column) or {}
        internal = _score(component.get("internal_risk", exposure_scores.get(column)))
        external = _score(component.get("external_risk", exposure_scores.get(column)))
        internal_score, external_score = internal or 0.0, external or 0.0
        matrix = matrix_by_attr.get(column, {})
        # 확정(유효값 1종 등)은 v9.6 규칙상 우선 검토 대상 → 노출(분포) H. 비적용은 점수 없음 → L
        dist_level = RiskLevel.H if column in exposure_determined else exposure_level(internal_score)
        report[column] = AttributeRisk(
            reid=level(id_grades.get(column, "L")),
            aggregation=level(agg_grades.get(column, "L")),
            exposure_dist=dist_level,
            exposure_link=exposure_level(external_score),
            purpose_fit=True,
            needed_for_purpose=True,
            ee_high=float(matrix.get("ee_match", 0) or 0) >= 0.50,
            risk_scores={
                "reid": next((float(row.get("risk_score", row.get("individual_risk_score", 0)) or 0)
                              for row in (modules.get("identification") or {}).get("attributes", [])
                              if str(row.get("attribute")) == column), 0.0),
                "aggregation": float(((modules.get("aggregation") or {}).get("attribute_scores") or {}).get(column, 0) or 0),
                # 확정 속성의 H 는 점수가 아니라 상태에서 온 것이므로 내부 점수를 넣지 않는다
                **({"exposure_dist": internal} if internal is not None and column not in exposure_determined else {}),
                **({"exposure_link": external} if external is not None else {}),
            },
        )
    return report


def _run_final_repseudonymization(source_job_id: str, selected_columns: set[str],
                                  df: pd.DataFrame, out_dir: Path,
                                  target_level: str = None, max_rounds: int = None,
                                  progress=None) -> dict:
    """제공된 2단계 최종 추천 파이프라인으로 목표 등급보다 높은 속성을 재처리한다 (대시보드용).
    [v24] 라운드 루프(run_repseudonymization_rounds) 결과 dict를 돌려준다."""
    result_path = RESULT_DIR / source_job_id / "integrated_result.json"
    if not result_path.exists():
        raise ValueError("1단계 위험 평가 결과를 찾을 수 없습니다.")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    return run_repseudonymization_rounds(
        source_result, selected_columns, df, out_dir,
        target_level=target_level or DEFAULT_TARGET_LEVEL,
        max_rounds=max_rounds or DEFAULT_MAX_ROUNDS, progress=progress)


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _mapping_file(module_dir: Path, ascii_name: str, korean_name: str) -> Path:
    """매핑 CSV 경로. 영문 이름 사본을 먼저 쓴다.
    Windows에서 압축을 풀면 한글 파일명이 깨질 수 있어 영문 이름 사본을 함께 둔다."""
    for name in (ascii_name, korean_name):
        if (module_dir / name).exists():
            return module_dir / name
    raise FileNotFoundError(f"매핑 파일을 찾을 수 없습니다: {module_dir / ascii_name}")


def run_repseudonymization_from_result(source_result: dict, selected_columns: Optional[set],
                                       df: pd.DataFrame, out_dir: Path,
                                       save_trial_data: bool = True,
                                       target_level: str = "M",
                                       original_df: Optional[pd.DataFrame] = None,
                                       agg_thresholds: Optional[dict] = None):
    """1단계 결과(dict)를 받아 강도 탐색 재가명처리를 실행하고 강도별 기록을 저장한다.

    [v24] target_level: "M"(v23과 같음, H만 처리) 또는 "L"(M 속성까지 처리, L까지 강도 탐색).
          재평가는 '목표보다 높은 축'을 모두 다시 계산한다 (v23은 H였던 축만).
    [v24] original_df: 강도 탐색을 시작할 원래 값. 라운드 2부터는 df(직전 라운드 처리본)를
          바탕으로 쓰고, 처리할 컬럼은 원래 값에서 다시 탐색한다. None이면 df.

    반환: (처리된 DataFrame, 컬럼별 요약, ProcessingReport, 속성별 처리 전 위험)
    저장: out_dir/strength_search_log.csv, candidate_comparison.csv, strength_search_log.json,
          out_dir/trials/<컬럼>/<기법>__<단계>.csv,
          out_dir/reassessment/<컬럼>/<기법>__<단계>/trial_result.json (+ error.log)
    """
    from modules.pseudonymization_pipeline import (
        AXIS_LABEL, PlanDoc, RiskLevel, AttributeRisk, run_pipeline,
        load_kcd8_mapping, load_combined_occupation_mapping, load_combined_region_mapping,
        get_base_seed, get_current_trial,
    )
    from modules.pseudonymization_pipeline.models import _LEVEL_ORDER
    from modules.integration.strength_log import save_strength_search, safe_name

    target = RiskLevel(str(target_level).upper())
    original_df = df if original_df is None else original_df

    def above(risk, axis) -> bool:
        """이 축이 목표 등급보다 높은가 → 재평가 대상"""
        return _LEVEL_ORDER[getattr(risk, axis).value] > _LEVEL_ORDER[target.value]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_report = _build_attribute_risk_report(source_result, df)
    if selected_columns:
        risk_report = {column: risk for column, risk in risk_report.items()
                       if column in selected_columns}

    module_dir = BASE_DIR / "modules" / "pseudonymization_pipeline"
    kcd = load_kcd8_mapping(str(_mapping_file(module_dir, "kcd8_189_disease_mapping.csv", "189개질환_KCD8_검증매핑.csv")))
    occupation = load_combined_occupation_mapping(
        str(_mapping_file(module_dir, "occupation_mapping_ko.csv", "직업분류_매핑테이블.csv")),
        str(module_dir / "ISCO-08_occupation_mapping.csv"),
    )
    region = load_combined_region_mapping(
        str(_mapping_file(module_dir, "region_mapping_ko.csv", "법정동_지역분류_매핑테이블.csv")),
        str(module_dir / "beopjeongdong_region_mapping_EN.csv"),
    )

    analysis_config = source_result.get("analysis_config") or {}
    aggregation_session = None

    def get_aggregation_session():
        nonlocal aggregation_session
        if aggregation_session is None:
            from modules.integration.aggregation_reassessment import AggregationReassessment
            target_col = str((source_result.get("modules", {}).get("aggregation") or {}).get("target_col", ""))
            if not target_col or target_col not in df.columns:
                raise ValueError("집계 재평가의 타깃 컬럼을 찾을 수 없습니다.")
            thresholds = (agg_thresholds or analysis_config.get("agg_fixed_thresholds")
                          or (source_result.get("modules", {}).get("aggregation") or {}).get("thresholds"))
            aggregation_session = AggregationReassessment(
                df, target_col, out_dir / "aggregation_baseline",
                int(analysis_config.get("agg_min_combo", 2)),
                int(analysis_config.get("agg_max_combo", 3)),
                fixed_thresholds=_reassess_thresholds(thresholds),   # 처리 전 경계 + 재평가 안심 완화
            )
        return aggregation_session

    def reassess_fn(changed_df: pd.DataFrame, attr: str) -> AttributeRisk:
        original = risk_report.get(attr, AttributeRisk())
        reid = original.reid
        aggregation = original.aggregation
        exposure_dist = original.exposure_dist
        exposure_link = original.exposure_link
        risk_scores = dict(original.risk_scores)
        errors: dict[str, str] = {}
        tracebacks: list[str] = []

        trial = get_current_trial()
        if trial:
            attempt_dir = (out_dir / "reassessment" / safe_name(attr)
                           / f"{safe_name(trial.get('technique', ''))}__{safe_name(trial.get('rung', ''))}")
        else:
            attempt_dir = out_dir / "reassessment" / safe_name(attr) / uuid.uuid4().hex
        attempt_dir.mkdir(parents=True, exist_ok=True)

        if above(original, "reid"):
            try:
                id_result = _run_identification(changed_df, int(analysis_config.get("id_max_combo", 3)), 10,
                                                reassess=True)
                row = next((x for x in id_result.get("attributes", [])
                            if str(x.get("attribute")) == attr), None)
                if row is None:
                    raise ValueError(f"식별 재평가 결과에 '{attr}' 속성이 없습니다.")
                reid = RiskLevel(str(row.get("final_grade", "H")))
                risk_scores["reid"] = float(row.get("risk_score", row.get("individual_risk_score", 0)) or 0)
            except Exception as exc:
                reid = RiskLevel.H
                risk_scores.pop("reid", None)
                errors["reid"] = _error_text(exc)
                tracebacks.append("[reid]\n" + traceback.format_exc())

        if above(original, "aggregation"):
            try:
                agg_result = get_aggregation_session().evaluate(changed_df, attr)
                aggregation = RiskLevel(str(agg_result["attribute_grade"]))
                risk_scores["aggregation"] = float(agg_result["attribute_score"])
            except Exception as exc:
                aggregation = RiskLevel.H
                risk_scores.pop("aggregation", None)
                errors["aggregation"] = _error_text(exc)
                tracebacks.append("[aggregation]\n" + traceback.format_exc())

        if above(original, "exposure_dist") or above(original, "exposure_link"):
            try:
                exposure_result, _ = _run_exposure(changed_df, attempt_dir / "exposure")
                component = (exposure_result.get("risk_components") or {}).get(attr)
                if component is None:
                    raise ValueError("재평가한 속성의 노출 지표를 찾을 수 없습니다.")
                determined = set(map(str, exposure_result.get("determined_attributes") or []))
                for axis, key in (("exposure_dist", "internal_risk"), ("exposure_link", "external_risk")):
                    if not above(original, axis):
                        continue
                    raw = component.get(key)
                    score = None if raw is None else float(raw)
                    if score is None or score != score:
                        # 노출 v9.6: 확정·비적용은 수치가 없음 — '낮아짐'으로 보지 않고 H 유지, 사유 기록
                        state = "확정(우선 검토)" if attr in determined else "비적용"
                        new_level = RiskLevel.H
                        errors[axis] = f"노출 지표가 수치가 아님: {state}"
                        risk_scores.pop(axis, None)
                    else:
                        new_level = RiskLevel.H if score >= 0.50 else RiskLevel.M if score >= 0.25 else RiskLevel.L
                        risk_scores[axis] = score
                    if axis == "exposure_dist":
                        exposure_dist = new_level
                    else:
                        exposure_link = new_level
            except Exception as exc:
                risk_scores.pop("exposure_dist", None)
                risk_scores.pop("exposure_link", None)
                exposure_dist = RiskLevel.H if above(original, "exposure_dist") else exposure_dist
                exposure_link = RiskLevel.H if above(original, "exposure_link") else exposure_link
                for axis in ("exposure_dist", "exposure_link"):
                    if above(original, axis):
                        errors[axis] = _error_text(exc)
                tracebacks.append("[exposure]\n" + traceback.format_exc())

        new_risk = AttributeRisk(
            reid=reid, aggregation=aggregation,
            exposure_dist=exposure_dist, exposure_link=exposure_link,
            risk_scores=risk_scores, errors=errors,
        )
        trial_record = {
            "attribute": attr, **trial,
            "before": {a: getattr(original, a).value for a in AXIS_LABEL},
            "after": {a: getattr(new_risk, a).value for a in AXIS_LABEL},
            "after_scores": risk_scores, "errors": errors,
        }
        (attempt_dir / "trial_result.json").write_text(
            json.dumps(_json_safe(trial_record), ensure_ascii=False, indent=2), encoding="utf-8")
        if tracebacks:
            (attempt_dir / "error.log").write_text("\n\n".join(tracebacks), encoding="utf-8")
        return new_risk

    def commit_fn(committed_df: pd.DataFrame, attr: str):
        if aggregation_session is not None:
            aggregation_session.evaluate(committed_df, attr, commit=True)

    processed, report = run_pipeline(
        original_df=original_df,
        pseudo_df=df,
        risk_report=risk_report,
        domain_mapping=kcd,
        plan_doc=PlanDoc(),
        reassess_fn=reassess_fn,
        commit_fn=commit_fn,
        occupation_mapping=occupation,
        region_mapping=region,
        target_level=target,
    )

    save_strength_search(report, original_df, risk_report, out_dir, base_seed=get_base_seed(),
                         save_trial_data=save_trial_data, target_level=target.value)

    summary = []
    for item in report.reprocessed:
        log = report.search_log.get(item.attribute, [])
        summary.append({
            "column": item.attribute,
            "method_label": item.technique.value,
            "attribute_type": item.attr_type.value,
            "risk_axis": AXIS_LABEL.get(item.dominant_axis, item.dominant_axis),
            "intensity": item.intensity,
            "target_level": target.value,
            "reached_level": next((c["extra"].get("reached_level") for c in
                                   report.candidate_comparison.get(item.attribute, [])
                                   if c["strategy"] is item), None),
            "before_risk": {a: risk_report[item.attribute].__getattribute__(a).value for a in AXIS_LABEL},
            "protection": report.protection_degree.get(item.attribute, "-"),
            "utility_loss": round(float(report.utility_loss.get(item.attribute, 0)), 4),
            "attempts": len(log),
            "reassessment_errors": sum(1 for e in log if e.get("errors")),
            "before_unique": int(original_df[item.attribute].nunique(dropna=True)),
            "after_unique": int(processed[item.attribute].nunique(dropna=True)),
            "changed_rows": int((original_df[item.attribute].astype("string").fillna("<NA>") !=
                                 processed[item.attribute].astype("string").fillna("<NA>")).sum()),
        })
    return processed, summary, report, risk_report


# ────────────────────────────── 모듈 실행 ──────────────────────────────
def _run_purpose(purpose_code: str, plan_path: Path, csv_path: Path) -> dict:
    from modules.purpose.main import run as purpose_run
    result = purpose_run(purpose_code, str(plan_path), str(csv_path))
    decision = result.get("판정")
    output = {
        "status": "success",
        "purpose_code": purpose_code,
        "purpose_label": next((k for k, v in PURPOSE_LABEL_TO_CODE.items() if v == purpose_code), purpose_code),
        "decision": decision,
        "passed": decision == "통과",
        "rejection_reasons": result.get("반려사유", []),
        "unmet_items": result.get("미충족항목", []),
        "danger_keywords": result.get("위험키워드", {}),
        "pseudonym_targets": result.get("가명처리대상", []),
    }
    output["display_details"] = _purpose_display_details(purpose_code, plan_path, result)
    output["analysis_summary"] = output["display_details"]["analysis_summary"]
    return output


# 재평가 전용 식별 조합 등급 기준 (처음 평가는 기본값 high 0.7 / low 0.0 그대로).
# 상위 10개 위험 조합 등장비율이 low 이하면 저위험, high 이상이면 고위험.
REASSESS_ID_COMBO_CUTS = {"combo_high_cut": 0.8, "combo_low_cut": 0.3}


def _run_identification(df: pd.DataFrame, max_combo_size: int, top_n: int,
                        reassess: bool = False) -> dict:
    from modules.correctmatch_merged.pipeline import assess_reidentification_risk, analyze_all_combinations
    from modules.correctmatch_merged.risk_classification import classify_attribute_risk, dataset_level_risk

    max_combo_size = max(1, min(max_combo_size, len(df.columns)))
    base = assess_reidentification_risk(df, method="quantile")
    combo_df = analyze_all_combinations(
        df, method="quantile", min_size=1, max_size=max_combo_size, k_anon_k=3)
    effective_top_n = min(top_n, len(combo_df))
    cuts = REASSESS_ID_COMBO_CUTS if reassess else {}
    attribute_df = classify_attribute_risk(combo_df, top_n=effective_top_n, **cuts)
    dataset_grade = dataset_level_risk(attribute_df)

    count_col = f"top{effective_top_n}_appearance_count"
    ratio_col = f"top{effective_top_n}_appearance_ratio"
    attribute_view = attribute_df.rename(columns={count_col: "top_appearance_count",
                                                  ratio_col: "top_appearance_ratio"})
    top_combos = combo_df.head(12).copy()
    top_combos["combination"] = top_combos["combination"].apply(list)
    grade_counts = attribute_df["final_grade"].value_counts().to_dict()
    return {
        "status": "success",
        "dataset_grade": dataset_grade,
        "dataset_grade_label": GRADE_KO.get(dataset_grade, dataset_grade),
        "uniqueness": round(float(base["uniqueness"]), 6),
        "correctness": round(float(base["correctness"]), 6),
        "attribute_grade_counts": {g: int(grade_counts.get(g, 0)) for g in ["H", "M", "L"]},
        "attributes": attribute_view.to_dict("records"),
        "top_n": effective_top_n,
        "top_combinations": top_combos.to_dict("records"),
        "max_combo_size": max_combo_size,
        "k_anonymity": 3,
        "combo_cuts": {"combo_high_cut": cuts.get("combo_high_cut", 0.7),
                       "combo_low_cut": cuts.get("combo_low_cut", 0.0)},
    }


def _run_aggregation(data_path: Path, target_col: str, min_combo_size: int,
                     max_combo_size: int, out_dir: Path, fixed_thresholds: Optional[dict] = None) -> dict:
    """[v25] fixed_thresholds: 처리 전(1단계) 경계. 재평가에서 넘기면 처리본으로 경계를 다시 뽑지 않는다."""
    from modules.aggregation_risk_system import AggregationConfig, AggregationRiskAnalyzer

    config = AggregationConfig(
        data_file=str(data_path), target_col=target_col, out_dir=str(out_dir),
        min_combo_size=min_combo_size, max_combo_size=max_combo_size, save_every=100,
        fixed_thresholds=fixed_thresholds or None,
    )
    analyzer = AggregationRiskAnalyzer(config)
    summary = analyzer.run()

    classified_path = out_dir / "final_risk_classification.csv"
    risk_df = pd.read_csv(classified_path) if classified_path.exists() else pd.DataFrame()
    top_rows = []
    if not risk_df.empty:
        cols = [c for c in ["Feature_Combination", "Num_Features", "Mean_Accuracy", "Mean_F1", "Final_Risk"]
                if c in risk_df.columns]
        top_rows = (risk_df.sort_values(["Risk_Score_Numeric", "Mean_Accuracy"], ascending=[False, False])
                    .head(15)[cols].to_dict("records"))

    info = summary.get("dataset_level_hml", {})
    attribute_grades = summary.get("attribute_level_hml", {}) or {}
    attribute_path = out_dir / "attribute_risk_levels.csv"
    attribute_scores = dict(summary.get("attribute_level_score") or {})
    if not attribute_scores and attribute_path.exists():
        attributes = pd.read_csv(attribute_path)
        ratio_col = f"top{config.attribute_top_n}_appearance_ratio"
        if ratio_col in attributes.columns:
            attribute_scores = attributes.set_index("attribute")[ratio_col].astype(float).to_dict()
    return {
        "status": "success",
        "target_col": target_col,
        "dataset_grade": info.get("dataset_final_risk"),
        "dataset_grade_label": GRADE_KO.get(info.get("dataset_final_risk"), info.get("dataset_final_risk")),
        "hml_distribution": info.get("hml_aggregation", {}),
        "raw_distribution": info.get("raw_4label_distribution", {}),
        "total_combinations": summary.get("summary", {}).get("total_combinations", 0),
        "above_baseline_pct": summary.get("summary", {}).get("above_baseline_pct", 0),
        "top_combinations": top_rows,
        "models": summary.get("config", {}).get("model_names", []),
        "combo_range": summary.get("config", {}).get("combo_range"),
        "attribute_grades": attribute_grades,
        "attribute_scores": attribute_scores,
        # [v25] 처리 전후를 같은 자로 재기 위한 경계와 그 출처
        "thresholds": summary.get("thresholds") or {},
        "threshold_source": summary.get("threshold_source", "computed"),
        "safe_baseline": summary.get("safe_baseline"),
        "reassess_safe_margin": summary.get("reassess_safe_margin", 0.0),
        "attribute_grade_method": summary.get("attribute_grade_method"),
    }


def _collect_agg_plots(job_id: str, agg_out: Path) -> list:
    plots = []
    for fname, (title, caption) in AGG_PLOT_META.items():
        if (agg_out / fname).exists():
            plots.append({"name": fname, "title": title, "caption": caption,
                          "url": f"/api/plot/{job_id}/aggregation/{fname}"})
    return plots


def _run_exposure(df: pd.DataFrame, out_dir: Path) -> tuple[dict, list]:
    """첨부된 최종 노출 프로세스를 대시보드용 결과 구조로 실행한다."""
    from exposure_pipeline import exposure_stage

    pipeline = _build_sensitive_detection_pipeline()
    raw_result = pipeline.run(df)
    detection_df = pipeline.to_frame(raw_result)
    exposure_out = exposure_stage.run_from_detection(
        df, raw_result, min_tier=("CONFIRMED", "PROBABLE", "WEAK"))
    matrix = exposure_out["matrix"]

    assessment = exposure_stage.assess_dataset(
        matrix, coverage_floor=exposure_stage.COVERAGE_FLOOR) if not matrix.empty else None
    grade_raw = assessment.grade if assessment is not None else "Low"
    grade = EXPOSURE_GRADE_TO_CODE.get(grade_raw, "L")

    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    if not matrix.empty:
        heatmap = exposure_stage.render_heatmap(
            matrix, out_path=str(out_dir / "exposure_heatmap.png"),
            category_map=exposure_out["roles"].category_map)
        if heatmap:
            plots.append({"name": "exposure_heatmap.png", "title": "속성별 노출 위험 지표",
                          "caption": "민감속성별 내부 및 외부 노출 지표입니다."})
        if assessment is not None:
            curve = exposure_stage.render_risk_curve(
                assessment, out_path=str(out_dir / "risk_curve.png"))
            if curve:
                plots.append({"name": "risk_curve.png", "title": "속성별 통합 노출 위험",
                              "caption": "노출 위험이 높은 속성과 데이터셋 등급을 보여줍니다."})

    detection_cols = [column for column in [
        "column", "category", "category_ko", "track", "stage", "tier",
        "is_sensitive", "review_required", "rf_confidence", "reasons"]
        if column in detection_df.columns]
    matrix_records = [] if matrix.empty else (
        matrix.reset_index().rename(columns={"index": "attribute"}).to_dict("records"))
    top_actions = [] if assessment is None else [
        {"rank": rank, "attribute": attribute, "risk": round(float(risk), 4)}
        for rank, attribute, risk in exposure_stage.topk_actions(assessment)]
    attribute_scores = {} if assessment is None else {
        str(attribute): round(float(score), 4)
        for attribute, score in assessment.scores.items() if pd.notna(score)
    }

    return {
        "status": "success",
        "dataset_grade": grade,
        "dataset_grade_label": GRADE_KO[grade],
        "rho": round(float(assessment.rho), 4) if assessment is not None else 0.0,
        "threshold": float(assessment.threshold) if assessment is not None else 0.5,
        "n_exceed": int(assessment.n_exceed) if assessment is not None else 0,
        "n_total": int(assessment.n_total) if assessment is not None else 0,
        "sensitive_columns": list(raw_result.track_a) + list(raw_result.track_b),
        "track_a": list(raw_result.track_a), "track_b": list(raw_result.track_b),
        "review_columns": list(raw_result.review),
        "top_actions": top_actions,
        "attribute_scores": attribute_scores,
        "risk_components": {} if assessment is None else assessment.components.to_dict("index"),
        "detection": detection_df[detection_cols].to_dict("records"),
        "exposure_matrix": matrix_records,
        "category_labels": EXPOSURE_CATEGORY_KO,
        "coverage_floor": float(exposure_stage.COVERAGE_FLOOR),
        # [노출 v9.6] 산출 상태 — 확정(우선 검토), 미산출(순위·ρ 제외), 지표별 상태 수
        "determined_attributes": [] if assessment is None else list(assessment.determined),
        "unscored_attributes": [] if assessment is None else list(assessment.unscored),
        "status_counts": {} if assessment is None or assessment.status_counts.empty
                         else assessment.status_counts.to_dict("index"),
    }, plots


def reassess_dataframe(df: pd.DataFrame, config: dict, target_col: str, out_dir: Path,
                       progress=None) -> tuple[dict, list]:
    """재가명처리본을 다시 평가한다 (식별은 REASSESS_ID_COMBO_CUTS, 집계는 REASSESS_SAFE_MARGIN 재평가 기준 적용)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda *args: None)
    modules, warnings = {}, []

    progress("identification", "식별 위험을 재평가하고 있습니다",
             "재가명처리본의 재식별 가능성을 다시 계산합니다.")
    try:
        modules["identification"] = _run_identification(
            df.copy(), int(config.get("id_max_combo", 3)), 10, reassess=True)
    except Exception as exc:
        modules["identification"] = {"status": "error", "message": str(exc),
                                     "trace": traceback.format_exc(limit=6)}
        warnings.append(f"식별 위험 재평가 실패: {exc}")

    progress("aggregation", "집계 위험을 재평가하고 있습니다",
             "기존 집계 타깃과 조합 기준을 적용해 추론 위험을 다시 계산합니다.")
    try:
        if target_col not in df.columns:
            raise ValueError(f"타깃 컬럼 '{target_col}'이 재가명처리본에 없습니다.")
        agg_csv = out_dir / "dataset_for_aggregation.csv"
        df.to_csv(agg_csv, index=False)
        modules["aggregation"] = _run_aggregation(
            agg_csv, target_col, int(config.get("agg_min_combo", 2)),
            int(config.get("agg_max_combo", 3)), out_dir / "aggregation",
            fixed_thresholds=_reassess_thresholds(config.get("agg_fixed_thresholds")))
        if not config.get("agg_fixed_thresholds"):
            warnings.append("처리 전 집계 경계가 없어 처리본으로 경계를 새로 계산했습니다 "
                            "(v25 이전 1단계 결과). 처리 전후 집계 등급은 같은 기준이 아닙니다.")
    except Exception as exc:
        modules["aggregation"] = {"status": "error", "message": str(exc),
                                  "trace": traceback.format_exc(limit=6)}
        warnings.append(f"집계 위험 재평가 실패: {exc}")

    progress("exposure", "노출 위험을 재평가하고 있습니다",
             "공통 민감정보 판정과 노출 지표를 재가명처리본에 다시 적용합니다.")
    try:
        modules["exposure"], _ = _run_exposure(df.copy(), out_dir / "exposure")
    except Exception as exc:
        modules["exposure"] = {"status": "error", "message": str(exc),
                               "trace": traceback.format_exc(limit=6)}
        warnings.append(f"노출 위험 재평가 실패: {exc}")
    return modules, warnings


# 재평가 전용 집계 '안심' 기준 완화폭 (정확도 기준, 0.10 = 10%p).
# 처음 평가는 영향 없음. 재가명처리 강도 탐색과 처리 후 재평가에서만
# "정확도 ≤ 처리 전 우연 수준 + 이 값"이면 안심으로 본다. 0으로 두면 처음 평가와 같은 기준.
REASSESS_SAFE_MARGIN = 0.10


def _reassess_thresholds(thresholds: Optional[dict]) -> Optional[dict]:
    """처리 전 경계에 재평가 전용 안심 기준 완화폭을 붙인다 (원본 dict는 바꾸지 않음)."""
    if not thresholds:
        return thresholds
    return {**thresholds, "__reassess_safe_margin__": REASSESS_SAFE_MARGIN}


DEFAULT_TARGET_LEVEL = "L"      # v24: 재가명처리 목표 등급 (M이면 v23 동작)
DEFAULT_MAX_ROUNDS = 1      # 재가명처리는 한 번만 실행 (목표 미달이어도 반복하지 않음)


def _attrs_above_target(result: dict, df: pd.DataFrame, target_level: str,
                        selected_columns: Optional[set] = None) -> list[str]:
    """속성별 위험에서 어느 한 축이라도 목표 등급보다 높은 속성."""
    t = _GRADE_ORDER[target_level]
    report = _build_attribute_risk_report(result, df)
    return [column for column, risk in report.items()
            if (not selected_columns or column in selected_columns)
            and any(_GRADE_ORDER[getattr(risk, axis).value] > t
                    for axis in ("reid", "aggregation", "exposure_dist", "exposure_link"))]


_GRADE_ORDER = {"L": 0, "M": 1, "H": 2}


def run_repseudonymization_rounds(source_result: dict, selected_columns: Optional[set],
                                  df: pd.DataFrame, out_dir: Path,
                                  target_level: str = DEFAULT_TARGET_LEVEL,
                                  max_rounds: int = DEFAULT_MAX_ROUNDS,
                                  save_trial_data: bool = True, progress=None) -> dict:
    """[v24] 데이터셋 수준 반복 루프.

    라운드마다 (속성별 강도 탐색 재가명처리 → 처리본 전체 재평가 → 통합 등급 확인)을 하고,
    통합 등급이 목표보다 높으면 재평가 결과로 다시 대상 속성을 골라 다음 라운드를 돈다.
      - 라운드 1은 out_dir 바로 아래(v23과 같은 위치), 라운드 k≥2는 out_dir/round_k 에 저장
      - 라운드 k≥2의 바탕 데이터는 직전 라운드 처리본, 처리 컬럼은 원래 값에서 다시 탐색
      - 멈춤 조건: 목표 도달 / 최대 라운드 / 처리할 속성 없음 / 통합 등급과 목표 초과 속성 수가
        직전 라운드와 같음(개선 없음) / 재평가 일부 실패
    반환 dict: processed, summary(컬럼별 마지막 처리), rounds, final(modules·warnings·integrated),
               utility_tv(표시용), stop_reason
    """
    from modules.integration.utility_tv import tv_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda *args: None)
    target_level = str(target_level).upper()
    max_rounds = max(1, int(max_rounds))
    config = dict(source_result.get("analysis_config") or {})
    target_col = str(config.get("target_col") or
                     ((source_result.get("modules") or {}).get("aggregation") or {}).get("target_col") or "")
    # [v25] 처리 전(1단계) 집계 경계 — 모든 라운드의 탐색·재평가를 이 경계로 잰다
    base_thresholds = ((source_result.get("modules") or {}).get("aggregation") or {}).get("thresholds") or None
    if base_thresholds:
        config["agg_fixed_thresholds"] = base_thresholds

    original = df
    current, current_result = df, source_result
    before_grade = (source_result.get("integrated") or {}).get("grade")
    prev = (before_grade, len(_attrs_above_target(source_result, df, target_level, selected_columns)))
    rounds, summary_by_column = [], {}
    final, stop_reason = None, "최대 라운드 도달"

    for round_no in range(1, max_rounds + 1):
        round_dir = out_dir if round_no == 1 else out_dir / f"round_{round_no}"
        progress("processing", "선택한 컬럼을 재가명처리하고 있습니다",
                 "선택한 처리 방법을 각 컬럼에 적용하고 변경 결과를 계산합니다.")
        processed, summary, _, _ = run_repseudonymization_from_result(
            current_result, selected_columns, current, round_dir,
            save_trial_data=save_trial_data, target_level=target_level, original_df=original,
            agg_thresholds=base_thresholds)
        for row in summary:
            summary_by_column[row["column"]] = {**row, "round": round_no}

        progress("processing", "선택한 컬럼을 재가명처리하고 있습니다",
                 "선택한 처리 방법을 각 컬럼에 적용하고 변경 결과를 계산합니다.")
        modules, warnings = reassess_dataframe(processed, config, target_col, round_dir / "round_assessment")
        integrated = compute_integrated_risk_from_modules(modules, processed)
        round_result = {"analysis_config": config, "modules": modules, "integrated": integrated}
        grade = (integrated or {}).get("grade")
        n_above = (len(_attrs_above_target(round_result, processed, target_level, selected_columns))
                   if grade in _GRADE_ORDER else None)
        info = {"round": round_no, "dir": str(round_dir.relative_to(out_dir)) if round_no > 1 else ".",
                "processed_columns": [row["column"] for row in summary],
                "grade_before": prev[0], "grade_after": grade,
                "module_grades": {name: (modules.get(name) or {}).get("dataset_grade") for name in modules},
                "attributes_above_target": n_above, "warnings": warnings}
        rounds.append(info)
        final = {"modules": modules, "warnings": warnings, "integrated": integrated}
        current, current_result = processed, round_result
        (round_dir / "round_summary.json").write_text(
            json.dumps(_json_safe(info), ensure_ascii=False, indent=2), encoding="utf-8")

        if grade not in _GRADE_ORDER:
            stop_reason = "재평가 일부 실패로 통합 등급을 확인할 수 없어 멈춤"
        elif _GRADE_ORDER[grade] <= _GRADE_ORDER[target_level]:
            stop_reason = f"목표 등급({GRADE_KO[target_level]}) 도달"
        elif not summary:
            stop_reason = "더 처리할 속성이 없음 (목표 초과 속성이 모두 처리 불가 또는 선택 밖)"
        elif (grade, n_above) == prev:
            stop_reason = "직전 라운드와 통합 등급·목표 초과 속성 수가 같아 멈춤 (개선 없음)"
        else:
            prev = (grade, n_above)
            continue
        break

    processed_columns = list(summary_by_column)
    utility = tv_report(original, current, processed_columns, target_col or None)
    payload = {"target_level": target_level, "max_rounds": max_rounds, "stop_reason": stop_reason,
               "grade_before": before_grade, "grade_after": (final.get("integrated") or {}).get("grade"),
               "rounds": rounds, "utility_tv": utility}
    (out_dir / "rounds_summary.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    if utility.get("queries"):
        pd.DataFrame(utility["queries"]).to_csv(out_dir / "utility_tv.csv", index=False, encoding="utf-8-sig")
    return {"processed": current, "summary": list(summary_by_column.values()), **payload, "final": final}


# ────────────────────────────── API ──────────────────────────────
@app.post("/api/inspect")
def inspect_dataset():
    data_file = request.files.get("data_file")
    if not data_file:
        return jsonify({"ok": False, "message": "데이터셋 파일을 선택해 주세요."}), 400
    job_dir = UPLOAD_DIR / f"inspect-{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = _save_upload(data_file, job_dir, "dataset.csv")
        df = _load_dataframe(path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        target_candidates = [c for c in df.columns if df[c].dropna().nunique() == 2]
        return jsonify({
            "ok": True,
            "rows": int(len(df)),
            "columns": [{
                "name": str(c), "dtype": str(df[c].dtype),
                "unique": int(df[c].nunique(dropna=True)),
                "missing_pct": round(float(df[c].isna().mean() * 100), 2),
                "numeric": c in numeric_cols,
            } for c in df.columns],
            "target_candidates": target_candidates,
            "aggregation_numeric_features": numeric_cols,
        })
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/api/analyze")
def analyze():
    data_file = request.files.get("data_file")
    plan_file = request.files.get("plan_file")
    purpose_label = request.form.get("purpose", "")
    target_col = request.form.get("target_col", "")
    if not data_file:
        return jsonify({"ok": False, "message": "데이터셋 파일이 필요합니다."}), 400
    if not plan_file:
        return jsonify({"ok": False, "message": "목적 외 사용 판정을 위한 계획서 PDF가 필요합니다."}), 400
    if purpose_label not in PURPOSE_LABEL_TO_CODE:
        return jsonify({"ok": False, "message": "허용된 처리 목적을 선택해 주세요."}), 400
    if not target_col:
        return jsonify({"ok": False, "message": "집계 위험 분석의 민감정보 타깃 컬럼을 선택해 주세요."}), 400

    try:
        id_max_combo = int(request.form.get("id_max_combo", "3"))
        agg_min_combo = int(request.form.get("agg_min_combo", "2"))
        agg_max_combo = int(request.form.get("agg_max_combo", "3"))
    except ValueError:
        return jsonify({"ok": False, "message": "조합 크기 설정을 확인해 주세요."}), 400

    requested_job_id = request.form.get("client_job_id", "").strip()
    job_id = (requested_job_id if len(requested_job_id) == 32
              and all(ch in "0123456789abcdefABCDEF" for ch in requested_job_id)
              else uuid.uuid4().hex)
    job_dir = UPLOAD_DIR / job_id
    out_dir = RESULT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_progress(out_dir, "preparing", "분석 자료를 준비하고 있습니다",
                  "업로드한 데이터셋과 활용 계획서를 확인합니다.")

    result = {
        "ok": True, "job_id": job_id, "pipeline_halted": False,
        "modules": {}, "plots": {"identification": [], "aggregation": []},
        "report": {"available": False, "docx_url": None}, "warnings": [],
    }

    try:
        data_path = _save_upload(data_file, job_dir, "dataset.csv")
        plan_path = _save_upload(plan_file, job_dir, "plan.pdf")
        if plan_path.suffix.lower() != ".pdf":
            raise ValueError("계획서는 PDF 형식이어야 합니다.")

        df = _load_dataframe(data_path)
        analysis_path = job_dir / "dataset_for_analysis.csv"
        df.to_csv(analysis_path, index=False)
        result["dataset"] = {
            "filename": data_path.name, "rows": int(len(df)),
            "columns": int(len(df.columns)), "column_names": [str(c) for c in df.columns],
        }
        result["analysis_config"] = {
            "purpose_label": purpose_label, "target_col": target_col,
            "id_max_combo": id_max_combo, "agg_min_combo": agg_min_combo,
            "agg_max_combo": agg_max_combo,
        }

        purpose_csv = job_dir / "dataset_for_purpose.csv"
        df.to_csv(purpose_csv, index=False)

        # ── 1단계: 목적 외 사용 게이트(사전판정) ──
        _set_progress(out_dir, "purpose", "목적 적합성을 확인하고 있습니다",
                      "활용 계획서와 선택한 처리 목적을 비교합니다.")
        try:
            gate = _run_purpose(PURPOSE_LABEL_TO_CODE[purpose_label], plan_path, purpose_csv)
        except Exception as exc:
            gate = {
                "status": "error", "message": str(exc), "passed": False,
                "purpose_label": purpose_label, "decision": None,
                "rejection_reasons": [f"게이트 실행 오류: {exc}"], "unmet_items": [],
                "danger_keywords": {}, "pseudonym_targets": [],
                "trace": traceback.format_exc(limit=5),
            }
        result["gate"] = gate

        # 게이트 반려/오류 → 이후 단계 전면 중단
        if not gate.get("passed"):
            result["pipeline_halted"] = True
            result["integrated"] = None
            report_path = out_dir / "adequacy_report.docx"
            try:
                generate_adequacy_report(_json_safe(result), report_path)
                result["report"] = {"available": True, "docx_url": f"/api/report/{job_id}"}
            except Exception as exc:
                result["warnings"].append(f"검토서 생성 실패: {exc}")
            _persist(out_dir, result)
            _set_progress(out_dir, "completed", "목적 적합성 판정을 완료했습니다",
                          "반려 사유를 확인해 주세요.", "completed")
            return jsonify(_json_safe(result))

        # 2단계: 식별 위험, 집계 위험 및 노출 위험 분석
        _set_progress(out_dir, "identification", "식별 위험을 분석하고 있습니다",
                      "준식별자 조합의 유일성과 정확 매칭률을 계산합니다.")
        try:
            ident = _run_identification(df.copy(), max_combo_size=id_max_combo, top_n=10)
            result["modules"]["identification"] = ident
            try:
                id_plot_dir = out_dir / "identification"
                metas = generate_identification_plots(ident, id_plot_dir)
                result["plots"]["identification"] = [
                    {**m, "url": f"/api/plot/{job_id}/identification/{m['name']}"} for m in metas]
            except Exception as exc:
                result["warnings"].append(f"식별 시각화 생성 실패: {exc}")
        except Exception as exc:
            result["modules"]["identification"] = {
                "status": "error", "message": str(exc),
                "trace": traceback.format_exc(limit=5),
                "hint": "식별 위험 모듈은 최초 실행 시 Julia와 CorrectMatch.jl 설치에 시간이 걸릴 수 있습니다."}

        _set_progress(out_dir, "aggregation", "집계 위험을 분석하고 있습니다",
                      "특성 조합별 민감정보 추론 가능성을 평가합니다.")
        try:
            if target_col not in df.columns:
                raise ValueError(f"타깃 컬럼 '{target_col}'을 데이터셋에서 찾을 수 없습니다.")
            aggregation_csv = job_dir / "dataset_for_aggregation.csv"
            df.to_csv(aggregation_csv, index=False)
            agg_out = out_dir / "aggregation"
            agg = _run_aggregation(aggregation_csv, target_col, agg_min_combo, agg_max_combo, agg_out)
            result["modules"]["aggregation"] = agg
            result["plots"]["aggregation"] = _collect_agg_plots(job_id, agg_out)
        except Exception as exc:
            result["modules"]["aggregation"] = {
                "status": "error", "message": str(exc),
                "trace": traceback.format_exc(limit=5),
                "hint": "집계 위험 분석은 타깃 컬럼과 특성 컬럼을 사용합니다. 범주형 특성은 자동으로 인코딩합니다."}

        _set_progress(out_dir, "exposure", "노출 위험을 분석하고 있습니다",
                      "민감정보 탐지와 내부 및 외부 노출 지표를 계산합니다.")
        try:
            exposure_out = out_dir / "exposure"
            exposure, exposure_plots = _run_exposure(df.copy(), exposure_out)
            result["modules"]["exposure"] = exposure
            result["plots"]["exposure"] = [
                {**plot, "url": f"/api/plot/{job_id}/exposure/{plot['name']}"}
                for plot in exposure_plots]
        except Exception as exc:
            result["modules"]["exposure"] = {
                "status": "error", "message": str(exc),
                "trace": traceback.format_exc(limit=5),
                "hint": "노출 위험 분석용 모델과 scikit-learn 버전을 확인해 주세요."}

        # ── 3단계: 세 모듈의 자체 데이터셋 등급 중 최고 등급 ──
        _set_progress(out_dir, "integration", "통합 위험도를 산출하고 있습니다",
                      "식별, 집계 및 노출 위험 결과를 통합합니다.")
        result["integrated"] = compute_integrated_risk_from_modules(result["modules"], df)
        if result["integrated"] is None:
            result["warnings"].append("세 위험 모듈의 속성별 결과가 없어 통합 위험도를 산출하지 못했습니다.")

        # ── 4단계: 적정성 검토서 생성 ──
        _set_progress(out_dir, "report", "적정성 검토서를 생성하고 있습니다",
                      "분석 결과와 시각화를 문서에 정리합니다.")
        report_path = out_dir / "adequacy_report.docx"
        try:
            generate_adequacy_report(_json_safe(result), report_path)
            result["report"] = {"available": True, "docx_url": f"/api/report/{job_id}"}
        except Exception as exc:
            result["warnings"].append(f"검토서 생성 실패: {exc}")
            result["report"] = {"available": False, "docx_url": None}

        _persist(out_dir, result)
        _set_progress(out_dir, "completed", "전체 분석을 완료했습니다",
                      "통합 위험도와 세부 분석 결과를 확인해 주세요.", "completed")
        return jsonify(_json_safe(result))

    except Exception as exc:
        _set_progress(out_dir, "error", "분석을 완료하지 못했습니다",
                      str(exc), "error")
        return jsonify({"ok": False, "message": str(exc),
                        "trace": traceback.format_exc(limit=8)}), 500


def _persist(out_dir: Path, result: dict):
    (out_dir / "integrated_result.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/result/<job_id>")
def get_result(job_id: str):
    if not job_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    path = RESULT_DIR / job_id / "integrated_result.json"
    if not path.exists():
        return jsonify({"ok": False, "message": "결과를 찾을 수 없습니다."}), 404
    try:
        return jsonify(_json_safe(_load_result_with_current_integration(job_id)))
    except (OSError, ValueError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.get("/api/progress/<job_id>")
def get_progress(job_id: str):
    if not job_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    return _progress_response(RESULT_DIR / job_id / "progress.json", "분석 준비 중입니다.")


@app.get("/api/plot/<job_id>/<module>/<path:name>")
def get_plot(job_id: str, module: str, name: str):
    if not job_id.isalnum() or module not in {"aggregation", "identification", "exposure"}:
        return jsonify({"ok": False, "message": "이미지 경로를 확인해 주세요."}), 400
    safe = secure_filename(name)
    plot_path = RESULT_DIR / job_id / module / safe
    if not plot_path.exists():
        return jsonify({"ok": False, "message": "이미지를 찾을 수 없습니다."}), 404
    return send_file(plot_path, mimetype="image/png")


@app.get("/api/report/<job_id>")
def get_report(job_id: str):
    if not job_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    result_path = RESULT_DIR / job_id / "integrated_result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        integrated = saved.get("integrated") or {}
        if integrated and integrated.get("score_kind") != "ordinal_grade":
            return jsonify({"ok": False, "message": "이전 통합식으로 만든 검토서입니다. 새 분석 후 다시 내려받아 주세요."}), 409
    report_path = RESULT_DIR / job_id / "adequacy_report.docx"
    if not report_path.exists():
        return jsonify({"ok": False, "message": "검토서를 찾을 수 없습니다."}), 404
    return send_file(
        report_path,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True, download_name="가명정보_적정성_검토서.docx")


@app.get("/api/repseudonymize/prepare/<job_id>")
def prepare_repseudonymization_from_result(job_id: str):
    try:
        data_path = _find_job_dataset(job_id)
        df = _load_dataframe(data_path)
        recommended, reasons = _recommended_columns(job_id, df)
        result_path = RESULT_DIR / job_id / "integrated_result.json"
        source_result = _load_result_with_current_integration(job_id) if result_path.exists() else {}
        integrated = source_result.get("integrated") or {}
        profile = _repseudonymization_profile(df, recommended, reasons)
        profile["automatic_pipeline"] = True
        profile["methods"] = {
            "auto": "위험축과 자료형에 따른 자동 추천",
            **METHOD_LABELS,
        }
        for column in profile["columns"]:
            column["suggested_method"] = "auto"
        return jsonify({
            "ok": True,
            "source": "risk_result",
            "job_id": job_id,
            "filename": data_path.name,
            "integrated_grade": integrated.get("grade"),
            "integrated_grade_label": integrated.get("grade_label"),
            **profile,
        })
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.post("/api/repseudonymize/inspect")
def inspect_repseudonymization_dataset():
    data_file = request.files.get("data_file")
    if not data_file:
        return jsonify({"ok": False, "message": "재가명처리할 데이터셋을 선택해 주세요."}), 400
    temp_dir = UPLOAD_DIR / f"repseudo-inspect-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_path = _save_upload(data_file, temp_dir, "dataset.csv")
        df = _load_dataframe(data_path)
        return jsonify({
            "ok": True,
            "source": "upload",
            "filename": data_path.name,
            **_repseudonymization_profile(df, [], {}),
        })
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/repseudonymize/progress/<task_id>")
def get_repseudonymization_progress(task_id: str):
    if not task_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    return _progress_response(RESULT_DIR / f"repseudo-{task_id}" / "progress.json",
                              "진행 상태를 준비하고 있습니다.")


@app.get("/api/repseudonymize/reassess/progress/<task_id>")
def get_reassessment_progress(task_id: str):
    if not task_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    return _progress_response(RESULT_DIR / f"repseudo-{task_id}" / "reassessment" / "progress.json",
                              "진행 상태를 준비하고 있습니다.")


@app.post("/api/repseudonymize")
def repseudonymize():
    data_file = request.files.get("data_file")
    source_job_id = request.form.get("job_id", "").strip()
    requested_task_id = request.form.get("client_task_id", "").strip()
    task_id = (requested_task_id if len(requested_task_id) == 32
               and all(ch in "0123456789abcdefABCDEF" for ch in requested_task_id)
               else uuid.uuid4().hex)
    task_upload_dir = UPLOAD_DIR / f"repseudo-{task_id}"
    task_result_dir = RESULT_DIR / f"repseudo-{task_id}"
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    task_result_dir.mkdir(parents=True, exist_ok=True)
    _set_progress(task_result_dir, "preparing", "재가명처리를 준비하고 있습니다",
                  "선택한 컬럼과 처리 방법을 확인합니다.")
    try:
        strategies = json.loads(request.form.get("strategies", "[]"))
        if not isinstance(strategies, list) or not strategies:
            raise ValueError("재가명처리할 컬럼과 처리 방법을 한 개 이상 선택해 주세요.")

        _set_progress(task_result_dir, "loading", "처리할 데이터셋을 불러오고 있습니다",
                      "원본 파일을 유지한 상태로 재가명처리용 사본을 준비합니다.")
        if source_job_id:
            source_path = _find_job_dataset(source_job_id)
            input_path = task_upload_dir / source_path.name
            shutil.copy2(source_path, input_path)
        elif data_file:
            input_path = _save_upload(data_file, task_upload_dir, "dataset.csv")
        else:
            raise ValueError("위험 평가 결과를 불러오거나 데이터셋을 선택해 주세요.")

        df = _load_dataframe(input_path)
        target_level = str(request.form.get("target_level") or DEFAULT_TARGET_LEVEL).upper()
        if target_level not in {"M", "L"}:
            target_level = DEFAULT_TARGET_LEVEL
        _set_progress(task_result_dir, "processing", "선택한 컬럼을 재가명처리하고 있습니다",
                      "선택한 처리 방법을 각 컬럼에 적용하고 변경 결과를 계산합니다.")
        rounds_info, final_assessment = None, None
        if source_job_id:
            automatic = [item for item in strategies if str(item.get("method", "")) == "auto"]
            manual = [item for item in strategies if str(item.get("method", "")) != "auto"]
            processed = df.copy()
            summary = []
            if automatic:
                automatic_columns = {str(item.get("column", "")) for item in automatic}
                rounds_info = _run_final_repseudonymization(
                    source_job_id, automatic_columns, processed, task_result_dir,
                    target_level=target_level,
                    progress=lambda stage, title, desc: _set_progress(task_result_dir, stage, title, desc))
                processed = rounds_info.pop("processed")
                final_assessment = rounds_info.pop("final")
                summary.extend(rounds_info.pop("summary"))
            if manual:
                processed, manual_summary = process_dataframe(processed, manual)
                summary.extend(manual_summary)
                final_assessment = None   # 수동 처리가 더해졌으면 라운드 재평가를 재사용할 수 없음
        else:
            processed, summary = process_dataframe(df, strategies)

        # [v24] TV 유용성 — 표시 전용 (선택·탈락에 쓰지 않음). 수동 처리까지 반영한 최종본 기준
        utility_tv = None
        if source_job_id:
            from modules.integration.utility_tv import tv_report
            source_cfg = (_load_result_with_current_integration(source_job_id).get("analysis_config") or {})
            utility_tv = tv_report(df, processed, [row["column"] for row in summary],
                                   source_cfg.get("target_col") or None)
        if final_assessment is not None:
            (task_result_dir / "final_assessment.json").write_text(
                json.dumps(_json_safe(final_assessment), ensure_ascii=False, indent=2), encoding="utf-8")

        _set_progress(task_result_dir, "saving", "재가명처리 결과 파일을 생성하고 있습니다",
                      "처리된 데이터와 컬럼별 변경 내역을 저장합니다.")
        output_path = task_result_dir / "repseudonymized.csv"
        processed.to_csv(output_path, index=False, encoding="utf-8-sig")

        response = {
            "ok": True,
            "task_id": task_id,
            "source_job_id": source_job_id or None,
            "filename": input_path.name,
            "rows": int(len(processed)),
            "columns": int(len(processed.columns)),
            "processed_columns": len(summary),
            "summary": summary,
            "download_url": f"/api/repseudonymized/{task_id}",
            "reassess_url": f"/api/repseudonymize/reassess/{task_id}" if source_job_id else None,
            "target_level": target_level,
            "rounds": rounds_info,
            "utility_tv": utility_tv,
            "final_assessment_reusable": final_assessment is not None,
        }
        (task_result_dir / "repseudonymization_result.json").write_text(
            json.dumps(_json_safe(response), ensure_ascii=False, indent=2), encoding="utf-8")
        _set_progress(task_result_dir, "completed", "재가명처리를 완료했습니다",
                      f"{len(summary)}개 컬럼의 처리를 마치고 결과 파일을 생성했습니다.", state="completed")
        return jsonify(_json_safe(response))
    except Exception as exc:
        _set_progress(task_result_dir, "error", "재가명처리를 완료하지 못했습니다",
                      str(exc), state="error")
        return jsonify({"ok": False, "message": str(exc), "trace": traceback.format_exc(limit=6)}), 400


@app.post("/api/repseudonymize/reassess/<task_id>")
def reassess_repseudonymized_file(task_id: str):
    """재가명처리본을 같은 세 모듈과 동일한 등급 통합 방식으로 재평가한다."""
    if not task_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    task_dir = RESULT_DIR / f"repseudo-{task_id}"
    reassess_dir = task_dir / "reassessment"
    reassess_dir.mkdir(parents=True, exist_ok=True)
    _set_progress(reassess_dir, "preparing", "위험도 재평가를 준비하고 있습니다",
                  "재가명처리본과 기존 위험 평가 기준을 확인합니다.")
    metadata_path = task_dir / "repseudonymization_result.json"
    output_path = task_dir / "repseudonymized.csv"
    if not metadata_path.exists() or not output_path.exists():
        _set_progress(reassess_dir, "error", "위험도 재평가를 시작하지 못했습니다",
                      "재가명처리 결과를 찾을 수 없습니다.", state="error")
        return jsonify({"ok": False, "message": "재가명처리 결과를 찾을 수 없습니다."}), 404
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_job_id = str(metadata.get("source_job_id") or "")
        if not source_job_id:
            raise ValueError("위험 평가와 연결된 재가명처리 결과만 전후 비교할 수 있습니다.")
        source_path = RESULT_DIR / source_job_id / "integrated_result.json"
        if not source_path.exists():
            raise ValueError("비교할 위험 평가 결과를 찾을 수 없습니다.")
        source_result = _load_result_with_current_integration(source_job_id)
        config = dict(source_result.get("analysis_config") or {})
        config["agg_fixed_thresholds"] = (((source_result.get("modules") or {}).get("aggregation") or {})
                                          .get("thresholds") or None)   # [v25] 처리 전 경계로 재평가
        target_col = str(config.get("target_col") or
                         ((source_result.get("modules") or {}).get("aggregation") or {}).get("target_col") or "")
        if not target_col:
            raise ValueError("집계 타깃 정보를 찾을 수 없습니다.")

        df = _load_dataframe(output_path)
        def progress(stage, title, description):
            _set_progress(reassess_dir, stage, title, description)

        final_path = task_dir / "final_assessment.json"
        if metadata.get("final_assessment_reusable") and final_path.exists():
            # [v24] 마지막 라운드에서 같은 처리본을 이미 전체 재평가했음 → 다시 계산하지 않음
            saved = json.loads(final_path.read_text(encoding="utf-8"))
            modules, warnings = saved.get("modules") or {}, saved.get("warnings") or []
        else:
            modules, warnings = reassess_dataframe(df, config, target_col, reassess_dir, progress)

        _set_progress(reassess_dir, "integration", "통합 위험도를 다시 산출하고 있습니다",
                      "세 위험 영역의 재평가 결과를 같은 통합 기준으로 정리합니다.")
        followup = {
            "ok": True, "task_id": task_id, "source_job_id": source_job_id,
            "dataset": {"filename": output_path.name, "rows": int(len(df)),
                        "columns": int(len(df.columns)), "column_names": list(map(str, df.columns))},
            "modules": modules, "warnings": warnings,
        }
        followup["integrated"] = compute_integrated_risk_from_modules(modules, df)

        _set_progress(reassess_dir, "comparison", "처리 전후 결과를 비교하고 있습니다",
                      "재가명처리 전후의 영역별 등급과 속성별 위험 변화를 정리합니다.")
        followup["comparison"] = _build_risk_comparison(
            source_result, followup, metadata.get("summary") or [])
        followup["comparison"]["utility_tv"] = metadata.get("utility_tv")
        followup["comparison"]["rounds"] = metadata.get("rounds")
        followup["comparison"]["target_level"] = metadata.get("target_level")
        comparison_path = task_dir / "risk_comparison.json"
        comparison_path.write_text(
            json.dumps(_json_safe(followup), ensure_ascii=False, indent=2), encoding="utf-8")
        _set_progress(reassess_dir, "completed", "위험도 재평가를 완료했습니다",
                      "재가명처리 전후 위험도 비교 결과를 확인해 주세요.", state="completed")
        return jsonify(_json_safe(followup))
    except Exception as exc:
        _set_progress(reassess_dir, "error", "위험도 재평가를 완료하지 못했습니다",
                      str(exc), state="error")
        return jsonify({"ok": False, "message": str(exc),
                        "trace": traceback.format_exc(limit=6)}), 400


@app.get("/api/repseudonymized/<task_id>")
def get_repseudonymized_file(task_id: str):
    if not task_id.isalnum():
        return jsonify({"ok": False, "message": "작업 ID를 확인해 주세요."}), 400
    output_path = RESULT_DIR / f"repseudo-{task_id}" / "repseudonymized.csv"
    if not output_path.exists():
        return jsonify({"ok": False, "message": "재가명처리 결과를 찾을 수 없습니다."}), 404
    return send_file(output_path, mimetype="text/csv", as_attachment=True,
                     download_name="재가명처리_결과.csv")


@app.get("/")
def index():
    return render_template("index.html", purposes=list(PURPOSE_LABEL_TO_CODE.keys()))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
