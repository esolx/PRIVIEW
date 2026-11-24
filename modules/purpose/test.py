from .main import run

케이스 = [
    ("test_files/eval_research.pdf", "통과", "신용평가=연구주제"),
    ("test_files/marketing_research.pdf", "통과", "마케팅=연구주제"),
    ("test_files/misuse_plan.pdf", "반려", "실제 마케팅 의도"),
]

for pdf, 기대, 설명 in 케이스:
    결과 = run("과학적연구", pdf, "test_files/sample_data.csv")
    표시 = "O" if 결과["판정"] == 기대 else "X 불일치!"
    print(f"{설명}: {결과['판정']} (기대: {기대}) {표시}")