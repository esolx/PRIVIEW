# -*- coding: utf-8 -*-
"""공통 유틸리티 — 타임스탬프 문자열, CSV 저장(append 지원)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def write_csv(df: pd.DataFrame, path: Path, mode: str = 'w', header: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, mode=mode, header=header)
