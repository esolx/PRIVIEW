# -*- coding: utf-8 -*-
"""
matplotlib 한글 폰트 설정.

기존 시각화 모듈들이 별도 폰트 설정 없이 한글 제목/라벨을 그려 깨짐(두부, □□□)이
발생했다. 이 헬퍼를 앱 시작 시 한 번 호출하면 프로세스 전역 rcParams가 설정되어
이후 모든 모듈의 matplotlib 출력에 한글이 정상 렌더링된다.

우선순위:
  1) 프로젝트에 동봉한 NanumGothic-Regular.ttf (OS 무관 보장)
  2) OS에 설치된 한글 폰트(Malgun Gothic / AppleGothic / NanumGothic / Noto Sans CJK KR)
"""
from __future__ import annotations

from pathlib import Path

_DONE = False
_BUNDLED = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "NanumGothic-Regular.ttf"
_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
               "Noto Sans KR", "Apple SD Gothic Neo", "Gulim", "Batang"]


def setup() -> str | None:
    """한글 폰트를 등록/설정하고 사용된 폰트명을 반환한다. (idempotent)"""
    global _DONE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    if _DONE:
        return plt.rcParams.get("font.family", [None])[0]

    chosen = None

    # 1) 동봉 폰트 우선 등록
    if _BUNDLED.exists():
        try:
            fm.fontManager.addfont(str(_BUNDLED))
            chosen = fm.FontProperties(fname=str(_BUNDLED)).get_name()
        except Exception:
            chosen = None

    # 2) 설치된 한글 폰트 탐색
    if not chosen:
        installed = {f.name for f in fm.fontManager.ttflist}
        for name in _CANDIDATES:
            if name in installed:
                chosen = name
                break

    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
    _DONE = True
    return chosen
