"""
data_reader.py
데이터셋(CSV)을 읽어서 컬럼 목록과 기본 정보를 파악하는 모듈.
purpose_gate의 document_reader가 PDF를 읽는 것과 같은 역할을 데이터셋에 대해 한다.
"""

import pandas as pd


def read_csv(file_path):
    """
    CSV 파일을 읽어서 데이터프레임으로 반환한다.

    입력:
        file_path (str): CSV 파일 경로
    출력:
        DataFrame: 읽어들인 데이터
    """
    return pd.read_csv(file_path)


def get_columns(df):
    """
    데이터프레임의 컬럼 이름 목록을 반환한다.

    입력:
        df (DataFrame): 데이터
    출력:
        list: 컬럼 이름 리스트
    """
    return list(df.columns)