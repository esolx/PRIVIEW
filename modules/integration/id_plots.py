# -*- coding: utf-8 -*-
"""
식별 위험 실험 시각화 생성.

식별(재식별) 모듈 결과 dict를 받아 대시보드용 실험 그래프를
out_dir에 PNG로 저장한다. Colab 실험 산출물과 동일한 성격의 그림을 통합 대시보드에서
바로 볼 수 있도록 한다.
"""
from __future__ import annotations

from pathlib import Path

from modules.common import mpl_korean

GRADE_COLOR = {"H": "#c94b45", "M": "#d8bb25", "L": "#3f8164"}
BRAND_COLOR = "#2f7d73"
GRID_COLOR = "#e4ebe8"


def _finish_axis(ax):
    """대시보드와 어울리는 절제된 플롯 스타일을 적용한다."""
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cfdad6")
    ax.tick_params(colors="#52635f")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def generate_identification_plots(id_result: dict, out_dir: Path) -> list[dict]:
    """생성된 플롯의 메타데이터 리스트를 반환한다."""
    mpl_korean.setup()
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    plots: list[dict] = []

    # 1) 속성 최종 등급 분포 (막대)
    counts = id_result.get("attribute_grade_counts", {}) or {}
    if counts:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        grades = ["H", "M", "L"]
        vals = [int(counts.get(g, 0)) for g in grades]
        labels = ["고위험", "중위험", "저위험"]
        bars = ax.bar(labels, vals, color=[GRADE_COLOR[g] for g in grades])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v), ha="center", va="bottom", fontsize=11)
        ax.set_title("속성별 최종 식별 등급 분포")
        ax.set_ylabel("속성 수")
        ax.set_ylim(0, max(vals) + 1 if vals else 1)
        _finish_axis(ax)
        plt.tight_layout()
        p = out_dir / "attribute_grade_distribution.png"
        plt.savefig(p, dpi=200); plt.close()
        plots.append({"name": p.name, "title": "속성별 최종 식별 등급 분포",
                      "caption": "개별 위험과 상위 위험 조합에서의 등장 빈도를 함께 반영한 속성별 최종 등급입니다."})

    # 2) 상위 조합 uniqueness / correctness 히트맵
    combos = (id_result.get("top_combinations") or [])[:12]
    if combos:
        rows = [" + ".join(c.get("combination", []))[:40] for c in combos]
        data = np.array([[float(c.get("uniqueness", 0)), float(c.get("correctness", 0))] for c in combos])
        fig, ax = plt.subplots(figsize=(7, max(3.2, len(rows) * 0.42)), dpi=150)
        im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["유일성", "정확 매칭률"])
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
        for i in range(len(rows)):
            for j in range(2):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        color="white" if data[i, j] > 0.55 else "black", fontsize=8)
        ax.set_title("상위 식별 위험 조합 히트맵")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        p = out_dir / "combination_risk_heatmap.png"
        plt.savefig(p, dpi=200); plt.close()
        plots.append({"name": p.name, "title": "상위 식별 위험 조합 히트맵",
                      "caption": "준식별자 조합별 유일성과 정확 매칭률을 색상 강도로 보여줍니다. 값이 클수록 재식별 위험이 높습니다."})

    # 3) 조합 크기 vs uniqueness (막대)
    if combos:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        sizes = [c.get("size", len(c.get("combination", []))) for c in combos]
        uniq = [float(c.get("uniqueness", 0)) for c in combos]
        idx = np.arange(len(combos))
        ax.bar(idx, uniq, color=BRAND_COLOR)
        ax.set_xticks(idx); ax.set_xticklabels([f"{s}개" for s in sizes], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("조합 크기"); ax.set_ylabel("유일성")
        ax.set_title("상위 조합의 크기별 유일성")
        _finish_axis(ax)
        plt.tight_layout()
        p = out_dir / "combo_size_uniqueness.png"
        plt.savefig(p, dpi=200); plt.close()
        plots.append({"name": p.name, "title": "조합 크기별 유일성",
                      "caption": "상위 위험 조합의 크기와 유일성을 함께 비교할 수 있습니다."})

    # 4) 유일성과 정확 매칭률의 관계 (산점도)
    if combos:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        uniqueness = [float(c.get("uniqueness", 0)) for c in combos]
        correctness = [float(c.get("correctness", 0)) for c in combos]
        sizes = [c.get("size", len(c.get("combination", []))) for c in combos]
        point_sizes = [55 + 22 * max(int(s) - 1, 0) for s in sizes]
        scatter = ax.scatter(uniqueness, correctness, s=point_sizes, c=sizes,
                             cmap="BuGn", vmin=1, vmax=max(sizes) if sizes else 1,
                             edgecolor="none", linewidth=0, alpha=0.9)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("유일성"); ax.set_ylabel("정확 매칭률")
        ax.set_title("유일성과 정확 매칭률의 관계")
        _finish_axis(ax)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("조합 크기")
        plt.tight_layout()
        p = out_dir / "uniqueness_correctness_scatter.png"
        plt.savefig(p, dpi=200); plt.close()
        plots.append({"name": p.name, "title": "유일성과 정확 매칭률의 관계",
                      "caption": "오른쪽 상단에 가까울수록 개인을 구분하고 정확하게 연결할 가능성이 모두 높은 조합입니다."})

    return plots
