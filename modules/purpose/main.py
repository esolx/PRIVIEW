"""
main.py
목적 외 사용 모듈의 최종 진입점.
1. 계획서로 목적 적합성을 판정한다. (통과/반려)
2. 통과하면 목적과 상관없는 민감 컬럼을 가명처리 대상으로 골라낸다.
   (실제 제거는 하지 않으며, 2단계 가명처리에서 사용한다)
3. 반려되면 대상 선별 없이 반려 결과만 반환한다.
"""

from .document_reader import extract_text_from_pdf
from .purpose_gate import purpose_gate
from .data_reader import read_csv, get_columns
from .column_check import find_pseudonym_targets


def run(purpose, plan_pdf_path, data_csv_path):
    """
    목적 외 사용 판정과 가명처리 대상 선별을 수행한다.

    입력:
        purpose (str): 사용자가 선택한 목적
        plan_pdf_path (str): 계획서 PDF 경로
        data_csv_path (str): 데이터셋 CSV 경로
    출력:
        dict: 판정 결과 + 가명처리 대상
    """

    # 1. 계획서에서 텍스트 추출
    text = extract_text_from_pdf(plan_pdf_path)

    # 2. 목적 적합성 판정
    판정 = purpose_gate(purpose, text)

    # 3. 반려되면 대상 선별 없이 종료
    if not 판정["통과여부"]:
        return {
            "판정": "반려",
            "반려사유": 판정["반려사유"],
            "미충족항목": 판정["미충족항목"],
            "위험키워드": 판정["위험키워드"],
            "가명처리대상": [],
        }

    # 4. 통과하면 데이터셋에서 가명처리 대상을 골라낸다
    df = read_csv(data_csv_path)
    columns = get_columns(df)
    targets = find_pseudonym_targets(purpose, columns)

    return {
        "판정": "통과",
        "반려사유": [],
        "미충족항목": [],
        "위험키워드": {},
        "가명처리대상": targets,
    }