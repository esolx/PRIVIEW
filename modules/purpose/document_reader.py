"""
document_reader.py
사용자가 업로드한 문서(PDF)에서 텍스트를 추출하는 모듈.
추출만 담당하며, 텍스트 정리와 검증은 preprocess 모듈에서 처리한다.
"""

import pdfplumber


def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 텍스트를 추출한다.

    입력:
        file_path (str): PDF 파일 경로
    출력:
        str: 추출된 원본 텍스트
    """

    text = ""

    # PDF 파일을 열고, 페이지를 하나씩 순회한다
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            # 텍스트가 없는 페이지(이미지 등)는 건너뛴다
            if page_text:
                text = text + page_text + "\n"

    return text