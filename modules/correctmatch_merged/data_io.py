"""데이터셋 로딩 — 확장자에 따라 자동으로 읽어 DataFrame으로 반환."""

from __future__ import annotations

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """확장자에 따라 csv/xlsx/xls/tsv/json 등을 자동으로 읽어 DataFrame으로 반환."""
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""

    if suffix == "csv":
        return pd.read_csv(path)
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(path)
    if suffix == "tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == "json":
        return pd.read_json(path)
    if suffix == "parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"지원하지 않는 파일 형식입니다: .{suffix}. (csv, xlsx, xls, tsv, json, parquet 지원)"
    )
