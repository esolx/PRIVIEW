# -*- coding: utf-8 -*-
"""재가명처리 강도 탐색 실험 실행기 (목적 외 사용 게이트 없이 위험평가만).

흐름
  1. 처리 전 평가   식별 · 집계 · 노출 → 통합 등급          → <out>/baseline/
  2. 재가명처리     목표 등급보다 높은 속성마다 기법 후보 × 강도 사다리 탐색 → <out>/repseudonymization/
                    시도마다 기존 위험평가 코드를 다시 호출하고 결과를 단계별로 저장
  3. 처리 후 평가   라운드마다 처리본 전체를 1단계와 같은 기준으로 재평가하고, 통합 등급이
                    목표보다 높으면 다음 라운드 (최대 3) → <out>/repseudonymization/[round_k/]round_assessment/
  4. 전후 비교      마지막 라운드 재평가 + TV 유용성(표시용) → <out>/risk_comparison.json

[v24] target_level="L"(기본): M 속성까지 처리하고 L까지 강도를 올린다. "M"이면 v23과 같은 탐색
      (H만 처리, H→M 첫 달성에서 멈춤)이지만 라운드 루프는 그대로 적용된다. v23과 완전히 같게
      돌리려면 target_level="M", max_rounds=1.

대시보드(app.py)의 평가 함수를 그대로 불러 쓰므로, 위험평가 코드는 대시보드와 같은 버전이다.

사용 예 (코랩 셀)
    import sys; sys.path.insert(0, "/content/PRIVIEW")
    from modules.pseudonymization_pipeline import set_base_seed
    import experiment

    set_base_seed(42)          # 기본 시드 (None이면 시드 고정 안 함)
    experiment.run_experiment("sample.csv", target_col="disease", out_dir="runs/seed_42")

시드 여러 개로 반복할 때는 처리 전 평가를 한 번만 하고 재사용한다.
    baseline = experiment.run_baseline("sample.csv", "disease", "runs/baseline")
    for s in [42, 0, 1]:
        set_base_seed(s)
        experiment.run_repseudonymization(baseline, f"runs/seed_{s}")
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import app
from modules.integration.final_risk import compute_integrated_risk_from_modules
from modules.pseudonymization_pipeline import get_base_seed

# 다른 모듈의 고정 시드 (기록용 — 값은 각 모듈에서 정해져 있다)
MODULE_SEEDS = {"aggregation_seed": 42, "exposure_bootstrap_seed": 7}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(app._json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_baseline(data_path: Union[str, Path], target_col: str, out_dir: Union[str, Path],
                 id_max_combo: int = 3, agg_min_combo: int = 2, agg_max_combo: int = 3) -> Path:
    """처리 전 평가(식별·집계·노출·통합)를 실행하고 저장한다. 저장 폴더를 돌려준다."""
    data_path, out_dir = Path(data_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = app._load_dataframe(data_path)
    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target_col}'이 데이터에 없습니다.")
    df.to_csv(out_dir / "dataset.csv", index=False, encoding="utf-8-sig")

    config = {"target_col": target_col, "id_max_combo": id_max_combo,
              "agg_min_combo": agg_min_combo, "agg_max_combo": agg_max_combo}
    result = {"dataset": {"filename": data_path.name, "rows": int(len(df)),
                          "columns": int(len(df.columns)),
                          "column_names": list(map(str, df.columns))},
              "analysis_config": config, "modules": {}, "warnings": [],
              "gate": None, "purpose_gate": "사용 안 함 (실험 모드)"}

    _log("처리 전 평가: 식별 위험")
    try:
        result["modules"]["identification"] = app._run_identification(df.copy(), id_max_combo, 10)
    except Exception as exc:
        result["modules"]["identification"] = {"status": "error", "message": str(exc),
                                               "trace": traceback.format_exc(limit=6)}
        result["warnings"].append(f"식별 위험 평가 실패: {type(exc).__name__}: {exc}")

    _log("처리 전 평가: 집계 위험")
    try:
        agg_csv = out_dir / "dataset_for_aggregation.csv"
        df.to_csv(agg_csv, index=False)
        result["modules"]["aggregation"] = app._run_aggregation(
            agg_csv, target_col, agg_min_combo, agg_max_combo, out_dir / "aggregation")
    except Exception as exc:
        result["modules"]["aggregation"] = {"status": "error", "message": str(exc),
                                            "trace": traceback.format_exc(limit=6)}
        result["warnings"].append(f"집계 위험 평가 실패: {type(exc).__name__}: {exc}")

    _log("처리 전 평가: 노출 위험")
    try:
        result["modules"]["exposure"], _ = app._run_exposure(df.copy(), out_dir / "exposure")
    except Exception as exc:
        result["modules"]["exposure"] = {"status": "error", "message": str(exc),
                                         "trace": traceback.format_exc(limit=6)}
        result["warnings"].append(f"노출 위험 평가 실패: {type(exc).__name__}: {exc}")

    result["integrated"] = compute_integrated_risk_from_modules(result["modules"], df)
    _write_json(out_dir / "integrated_result.json", result)
    for w in result["warnings"]:
        _log("경고: " + w)
    grade = (result["integrated"] or {}).get("grade")
    _log(f"처리 전 통합 등급: {grade}  → {out_dir / 'integrated_result.json'}")
    return out_dir


def run_repseudonymization(baseline_dir: Union[str, Path], out_dir: Union[str, Path],
                           columns: Optional[list] = None, final_reassessment: bool = True,
                           save_trial_data: bool = True,
                           target_level: str = app.DEFAULT_TARGET_LEVEL,
                           max_rounds: int = app.DEFAULT_MAX_ROUNDS) -> dict:
    """저장된 처리 전 평가를 읽어 강도 탐색 재가명처리(라운드 루프) → 전후 비교를 실행한다.

    columns: 재처리 후보로 넘길 컬럼 (None이면 전체. 실제 처리는 목표 등급보다 높은 속성만 된다)
    final_reassessment: v24에서는 라운드마다 전체 재평가를 하므로 항상 마지막 라운드 재평가로
                        전후 비교를 만든다. False면 risk_comparison.json만 쓰지 않는다.
    시드는 실행 전에 set_base_seed()로 정한다.
    """
    baseline_dir, out_dir = Path(baseline_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_result = json.loads((baseline_dir / "integrated_result.json").read_text(encoding="utf-8"))
    df = app._load_dataframe(baseline_dir / "dataset.csv")
    config = source_result.get("analysis_config") or {}
    base_seed = get_base_seed()
    if base_seed is None:
        _log("주의: 기본 시드가 없습니다. PRAM·스와핑·지터링 결과가 실행마다 달라집니다.")

    run_config = {"baseline_dir": str(baseline_dir), "base_seed": base_seed, **MODULE_SEEDS,
                  "columns": columns, "analysis_config": config,
                  "target_level": target_level, "max_rounds": max_rounds,
                  "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write_json(out_dir / "experiment_config.json", run_config)

    _log(f"재가명처리 강도 탐색 시작 (기본 시드 {base_seed}, 목표 {target_level}, 최대 {max_rounds}라운드)")
    repseudo_dir = out_dir / "repseudonymization"
    rounds = app.run_repseudonymization_rounds(
        source_result, set(columns) if columns else None, df, repseudo_dir,
        target_level=target_level, max_rounds=max_rounds, save_trial_data=save_trial_data,
        progress=lambda stage, title, desc: _log(title))
    processed, summary, final = rounds.pop("processed"), rounds.pop("summary"), rounds.pop("final")
    processed.to_csv(out_dir / "repseudonymized.csv", index=False, encoding="utf-8-sig")
    _write_json(repseudo_dir / "summary.json", summary)
    for r in rounds["rounds"]:
        _log(f"라운드 {r['round']}: 처리 {len(r['processed_columns'])}개, "
             f"통합 등급 {r['grade_before']} → {r['grade_after']}, 목표 초과 속성 {r['attributes_above_target']}개")
    _log(f"멈춘 이유: {rounds['stop_reason']}")
    tv = rounds.get("utility_tv") or {}
    if tv.get("status") == "success":
        _log(f"TV 유용성(표시용): 처리 컬럼 {tv['n_columns']}개 중 {tv['n_columns_pass']}개 기준 이내")

    logs = [p for p in [repseudo_dir / "strength_search_log.csv",
                        *sorted(repseudo_dir.glob("round_*/strength_search_log.csv"))]
            if p.exists() and p.stat().st_size > 3]
    trials = pd.concat([pd.read_csv(p).assign(round=1 if p.parent == repseudo_dir else int(p.parent.name.split("_")[1]))
                        for p in logs], ignore_index=True) if logs else pd.DataFrame()
    n_err = int(trials["has_error"].sum()) if "has_error" in trials else 0
    _log(f"처리 컬럼 {len(summary)}개, 시도 {len(trials)}건, 재평가 오류 {n_err}건")
    if n_err:
        _log("재평가 오류가 있습니다. strength_search_log.csv의 *_error 열과 "
             "repseudonymization/reassessment/*/*/error.log를 확인하세요.")

    output = {"out_dir": str(out_dir), "base_seed": base_seed, "summary": summary,
              "strength_log": str(repseudo_dir / "strength_search_log.csv"),
              "n_trials": int(len(trials)), "n_reassessment_errors": n_err,
              "target_level": target_level, "rounds": rounds["rounds"],
              "stop_reason": rounds["stop_reason"], "utility_tv": tv}

    if final_reassessment:
        # 마지막 라운드에서 최종 처리본을 이미 전체 재평가했으므로 그 결과를 그대로 쓴다
        modules, warnings = final["modules"], final["warnings"]
        followup = {"modules": modules, "warnings": warnings,
                    "integrated": compute_integrated_risk_from_modules(modules, processed)}
        followup["comparison"] = app._build_risk_comparison(source_result, followup, summary)
        followup["comparison"].update({"utility_tv": tv, "rounds": rounds["rounds"],
                                       "target_level": target_level,
                                       "stop_reason": rounds["stop_reason"]})
        _write_json(out_dir / "risk_comparison.json", followup)
        for w in warnings:
            _log("경고: " + w)
        cmp = followup["comparison"]
        _log(f"통합 등급 {cmp.get('before_grade')} → {cmp.get('after_grade')}")
        output["before_grade"] = cmp.get("before_grade")
        output["after_grade"] = cmp.get("after_grade")

    run_config["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(out_dir / "experiment_config.json", run_config)
    _log(f"완료 → {out_dir}")
    return output


def run_experiment(data_path: Union[str, Path], target_col: str, out_dir: Union[str, Path],
                   id_max_combo: int = 3, agg_min_combo: int = 2, agg_max_combo: int = 3,
                   columns: Optional[list] = None, final_reassessment: bool = True,
                   save_trial_data: bool = True,
                   target_level: str = app.DEFAULT_TARGET_LEVEL,
                   max_rounds: int = app.DEFAULT_MAX_ROUNDS) -> dict:
    """처리 전 평가부터 전후 비교까지 한 번에 실행한다."""
    out_dir = Path(out_dir)
    baseline = run_baseline(data_path, target_col, out_dir / "baseline",
                            id_max_combo, agg_min_combo, agg_max_combo)
    return run_repseudonymization(baseline, out_dir, columns=columns,
                                  final_reassessment=final_reassessment,
                                  save_trial_data=save_trial_data,
                                  target_level=target_level, max_rounds=max_rounds)
