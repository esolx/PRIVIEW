# -*- coding: utf-8 -*-
"""저장된 재가명처리 결과의 '처리 후 식별 등급'을 조합 등급 기준만 바꿔 다시 매기는 도구.
식별(Julia)을 다시 계산하지 않고, 저장된 등장비율·개인 위험점수로 등급만 다시 정한다 (몇 초).

사용법 (app.py가 있는 폴더에서):
    미리보기:  python tools/regrade_identification.py results\\repseudo-<task_id> --low 0.3 --high 0.8
    반영하기:  python tools/regrade_identification.py results\\repseudo-<task_id> --low 0.3 --high 0.8 --apply

--apply를 붙이면 final_assessment.json을 고칩니다 (원본은 final_assessment.backup.json으로 한 번만 보관).
반영 후 대시보드에서 '위험도 재평가'를 다시 누르면 바뀐 등급으로 전후 비교가 나옵니다.
"""
import argparse
import json
import shutil
from pathlib import Path

ORDER = {"L": 0, "M": 1, "H": 2}
LABEL = {"H": "고위험", "M": "중위험", "L": "저위험"}


def combo_grade(ratio: float, high: float, low: float) -> str:
    if ratio >= high:
        return "H"
    if ratio > low:
        return "M"
    return "L"


def dataset_grade(grades: list) -> str:
    counts = {g: grades.count(g) for g in "HML"}
    best = max(counts.values())
    return max((g for g, c in counts.items() if c == best), key=ORDER.get)   # 동률이면 더 위험한 쪽


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--low", type=float, default=0.3, help="조합 저위험 상한 (등장비율이 이 값 이하면 저위험)")
    ap.add_argument("--high", type=float, default=0.8, help="조합 고위험 하한 (등장비율이 이 값 이상이면 고위험)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = Path(args.task_dir) / "final_assessment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ident = (data.get("modules") or {}).get("identification") or {}
    if ident.get("status") != "success" or not ident.get("attributes"):
        raise SystemExit("저장된 식별 결과가 없습니다 (식별 재평가가 실패했거나 다른 폴더입니다).")

    old_cuts = ident.get("combo_cuts") or {}
    print(f"저장된 기준: 고위험 ≥ {old_cuts.get('combo_high_cut')}  저위험 ≤ {old_cuts.get('combo_low_cut')}")
    print(f"새 기준    : 고위험 ≥ {args.high}  저위험 ≤ {args.low}\n")
    print(f"{'속성':16s} {'개인':4s} {'등장비율':>7s}  {'조합':4s} {'최종(전→후)'}")
    new_grades = []
    for row in ident["attributes"]:
        ratio = float(row.get("top_appearance_ratio") or 0)
        indiv = str(row.get("individual_grade") or "L")
        combo = combo_grade(ratio, args.high, args.low)
        final = max(indiv, combo, key=ORDER.get)
        print(f"{str(row.get('attribute')):16s} {indiv:4s} {ratio:7.1f}  {combo:4s} {row.get('final_grade')} → {final}")
        new_grades.append(final)
        if args.apply:
            row["combo_grade"], row["final_grade"] = combo, final
    ds_old, ds_new = ident.get("dataset_grade"), dataset_grade(new_grades)
    counts = {g: new_grades.count(g) for g in "HML"}
    print(f"\n데이터셋 식별 등급: {ds_old} → {ds_new}   (H {counts['H']} / M {counts['M']} / L {counts['L']})")

    if args.apply:
        backup = path.with_name("final_assessment.backup.json")
        if not backup.exists():
            shutil.copy2(path, backup)
        ident["dataset_grade"] = ds_new
        ident["dataset_grade_label"] = LABEL.get(ds_new, ds_new)
        ident["attribute_grade_counts"] = counts
        ident["combo_cuts"] = {"combo_high_cut": args.high, "combo_low_cut": args.low}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n반영했습니다: {path}\n원본 보관: {backup}")
        print("대시보드에서 '위험도 재평가'를 다시 누르면 바뀐 등급으로 전후 비교가 나옵니다.")


if __name__ == "__main__":
    main()
