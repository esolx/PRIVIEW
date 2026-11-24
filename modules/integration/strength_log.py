# -*- coding: utf-8 -*-
"""강도 사다리 탐색 결과를 파일로 저장한다.

run_pipeline()이 돌려주는 ProcessingReport의 candidate_comparison(후보 기법별 시도 기록)을
읽어 다음 파일을 만든다.

  strength_search_log.csv   시도 1건 = 1행 (컬럼 × 기법 × 강도 단계)
  candidate_comparison.csv  후보 기법 1개 = 1행 (컬럼 × 기법)
  strength_search_log.json  위 두 표 + 처리 요약을 한 파일에
  trials/<컬럼>/<기법>__<단계>.csv  그 시도의 처리본 (원본값, 처리값)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

AXES = ("reid", "aggregation", "exposure_dist", "exposure_link")
AXIS_KO = {"reid": "식별", "aggregation": "집계",
           "exposure_dist": "노출_분포", "exposure_link": "노출_외부연결"}


def safe_name(text: str) -> str:
    """한글은 살리고 파일 이름에 못 쓰는 문자만 바꾼다. (secure_filename은 한글을 지운다)"""
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text)).strip("._")
    return text or "column"


def _level(risk, axis) -> Optional[str]:
    if risk is None:
        return None
    value = getattr(risk, axis, None)
    return getattr(value, "value", value)


def save_strength_search(report, original_df: pd.DataFrame, risk_report: dict,
                         out_dir, base_seed: Optional[int] = None,
                         save_trial_data: bool = True, target_level: Optional[str] = None) -> dict:
    """ProcessingReport를 CSV/JSON으로 저장하고 저장한 경로를 돌려준다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = out_dir / "trials"

    adopted = {s.attribute: getattr(s.technique, "value", str(s.technique))
               for s in report.reprocessed}

    trial_rows, candidate_rows = [], []
    for attr, candidates in report.candidate_comparison.items():
        before = risk_report.get(attr)
        orig = original_df[attr] if attr in original_df.columns else None
        for cand in candidates:
            technique = getattr(cand["technique"], "value", str(cand["technique"]))
            chosen_rung = (cand.get("extra") or {}).get("chosen_rung")
            technique_adopted = adopted.get(attr) == technique
            h_axes = list(cand.get("h_axes") or [])
            log = cand.get("log") or []

            for order, entry in enumerate(log, start=1):
                new_risk = entry.get("new_risk")
                rung = entry.get("label")
                row = {
                    "column": attr, "technique": technique, "rung_order": order, "rung": rung,
                    "seed": entry.get("seed"), "base_seed": base_seed,
                    "skipped": entry.get("skipped") or "",
                    "change_rate": entry.get("change_rate"),
                    "utility_loss": entry.get("utility"),
                    "target_level": target_level,
                    "achieved": bool(entry.get("achieved")),
                    "achieved_m": bool(entry.get("achieved_m", entry.get("achieved"))),
                    "reached_level": entry.get("reached"),
                    "rung_chosen_in_technique": (rung == chosen_rung) and not cand.get("all_noop"),
                    "technique_adopted": technique_adopted,
                    "final_adopted": technique_adopted and rung == chosen_rung,
                    "h_axes": "+".join(h_axes),
                }
                scores = getattr(new_risk, "risk_scores", {}) if new_risk is not None else {}
                errors = entry.get("errors") or {}
                for axis in AXES:
                    ko = AXIS_KO[axis]
                    row[f"{ko}_before"] = _level(before, axis)
                    row[f"{ko}_after"] = _level(new_risk, axis)
                    row[f"{ko}_after_score"] = scores.get(axis) if axis in h_axes else None
                    row[f"{ko}_error"] = errors.get(axis, "")
                row["has_error"] = bool(errors)

                processed = entry.get("processed")
                if save_trial_data and processed is not None and orig is not None:
                    path = trials_dir / safe_name(attr) / f"{safe_name(technique)}__{safe_name(rung)}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"original": orig.values, "processed": pd.Series(processed).values}) \
                        .to_csv(path, index=False, encoding="utf-8-sig")
                    row["trial_data"] = str(path.relative_to(out_dir))
                else:
                    row["trial_data"] = ""
                trial_rows.append(row)

            candidate_rows.append({
                "column": attr, "technique": technique,
                "all_noop": bool(cand.get("all_noop")),
                "target_level": target_level,
                "achieved": bool(cand.get("achieved")),
                "achieved_m": bool(cand.get("achieved_m", cand.get("achieved"))),
                "reached_level": cand.get("reached"),
                "chosen_rung": chosen_rung,
                "m_rung": (cand.get("extra") or {}).get("m_rung"),
                "utility_loss": cand.get("ul"),
                "composite_score": None if cand.get("score") in (None, float("inf")) else cand.get("score"),
                "n_trials": sum(1 for e in log if not e.get("skipped")),
                "n_skipped": sum(1 for e in log if e.get("skipped")),
                "n_errors": sum(1 for e in log if e.get("errors")),
                "adopted": technique_adopted,
            })

    trial_df = pd.DataFrame(trial_rows)
    cand_df = pd.DataFrame(candidate_rows)
    trial_csv = out_dir / "strength_search_log.csv"
    cand_csv = out_dir / "candidate_comparison.csv"
    trial_df.to_csv(trial_csv, index=False, encoding="utf-8-sig")
    cand_df.to_csv(cand_csv, index=False, encoding="utf-8-sig")

    payload = {
        "base_seed": base_seed,
        "target_level": target_level,
        "processed_columns": list(adopted),
        "skipped_not_high": list(report.skipped),   # '목표 등급 이하라 처리 안 함' (목표 M이면 v23과 같음)
        "no_effect": list(report.no_effect),
        "protection_degree": dict(report.protection_degree),
        "utility_loss": {k: float(v) for k, v in report.utility_loss.items()},
        "trials": trial_df.where(pd.notna(trial_df), None).to_dict("records"),
        "candidates": cand_df.where(pd.notna(cand_df), None).to_dict("records"),
    }
    json_path = out_dir / "strength_search_log.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"trial_csv": str(trial_csv), "candidate_csv": str(cand_csv), "json": str(json_path),
            "n_trials": len(trial_rows), "n_errors": int(trial_df["has_error"].sum()) if len(trial_df) else 0}
