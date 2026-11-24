# -*- coding: utf-8 -*-
"""결과 리포트 출력."""
from __future__ import annotations

from .models import AXIS_LABEL, ProcessingReport

def print_report(report: ProcessingReport) -> None:
    print("=" * 70)
    print("🎯 가명처리 전략 추천 및 재적용 완료 리포트")
    print("=" * 70)

    if report.deleted_purpose:
        print(f"🗑️ [목적외사용 → 삭제] {', '.join(report.deleted_purpose)}")
    if report.matching_key_skipped:
        print(f"🔑 [매칭키 → 재처리 제외] {', '.join(report.matching_key_skipped)}")
    if report.skipped:
        print(f"✅ [유지됨] H 위험이 없어 재처리 스킵된 속성: {', '.join(report.skipped)}\n")
    if report.no_effect:
        print(f"⚠️ [처리 불가] H 위험이지만 모든 후보 기법이 전 강도에서 값을 바꾸지 못해 기존 가명값 유지: "
              f"{', '.join(report.no_effect)}\n")

    for s in report.reprocessed:
        print(f"🔸 속성명 : {s.attribute} ({s.attr_type.value})")
        print(f"   - 지배위험 : {AXIS_LABEL.get(s.dominant_axis, s.dominant_axis)}")
        print(f"   - 적용전략 : {s.technique.value}")
        print(f"   - 처리강도 : {s.intensity}")
        if s.detail:
            print(f"   - 참고 : {s.detail}")
        print(f"   - 보호정도 : {report.protection_degree[s.attribute]}")
        print(f"   - 유용성(손실률) : {report.utility_loss[s.attribute]:.4f} (0에 가까울수록 손실 적음)")
        print("-" * 70)

