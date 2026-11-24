# -*- coding: utf-8 -*-
"""
가명정보 처리 적정성 검토서(.docx) 생성.

통합 분석 결과(dict)를 입력받아 정식 검토서 형식의 Word 문서를 만든다. 검토서는
문서 전반의 서술은 "~한다 / ~이다"로 통일한다. 위험 등급은 고위험, 중위험, 저위험으로
표기한다.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

_GRADE_LABEL = {"H": "고위험", "M": "중위험", "L": "저위험"}
_FONT = "NanumGothic"

_INK = RGBColor(0x1B, 0x1D, 0x20)
_MUTED = RGBColor(0x55, 0x57, 0x5B)
_HIGH = RGBColor(0xB0, 0x24, 0x18)
_MID = RGBColor(0x8A, 0x54, 0x0A)
_LOW = RGBColor(0x1E, 0x6B, 0x3A)

# A4 기준 본문 가용 폭(좌우 여백 2.4cm) = 16.2cm
_CONTENT_W = 16.2


def _set_kfont(run):
    run.font.name = _FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), _FONT)


def _p(doc, text="", size=10.5, bold=False, color=_INK, align=None, after=6, before=0):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(after)
    para.paragraph_format.space_before = Pt(before)
    para.paragraph_format.line_spacing = 1.5
    if align is not None:
        para.alignment = align
    if text:
        run = para.add_run(text)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        _set_kfont(run)
    return para


def _heading(doc, num, title):
    """번호 제목. 하단 구분선을 넣지 않는다."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(18)
    para.paragraph_format.space_after = Pt(8)
    run = para.add_run(f"{num}. {title}")
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = _INK
    _set_kfont(run)
    return para


def _shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(qn("w:shd"), {
        qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): hex_color})
    tcPr.append(shd)


def _cell(cell, text, bold=False, size=9.5, color=_INK, align=None):
    cell.text = ""
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.line_spacing = 1.3
    if align is not None:
        para.alignment = align
    run = para.add_run("" if text is None else str(text))
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    _set_kfont(run)


def _fix_widths(table, widths):
    """표를 고정 폭 레이아웃으로 만들고 각 열 너비를 확정한다."""
    from docx.oxml.ns import qn as _qn
    table.autofit = False
    tblPr = table._tbl.tblPr
    # 고정 레이아웃
    layout = tblPr.find(_qn("w:tblLayout"))
    if layout is None:
        layout = tblPr.makeelement(_qn("w:tblLayout"), {})
        tblPr.append(layout)
    layout.set(_qn("w:type"), "fixed")
    # tblGrid 열 폭(dxa) 설정
    grid = table._tbl.find(_qn("w:tblGrid"))
    if grid is not None:
        cols = grid.findall(_qn("w:gridCol"))
        for col, w in zip(cols, widths):
            col.set(_qn("w:w"), str(int(w * 567)))  # 1cm = 567 dxa
    for row in table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)


def _kv_table(doc, rows):
    w0, w1 = 4.4, _CONTENT_W - 4.4
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for k, v in rows:
        cells = table.add_row().cells
        _cell(cells[0], k, bold=True)
        _shade(cells[0], "F3F3F4")
        _cell(cells[1], v)
    _fix_widths(table, [w0, w1])
    return table


def _grid_table(doc, headers, rows, widths, colors=None, aligns=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    _AL = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}
    if aligns is None:
        aligns = ["left" if i == 0 else "center" for i in range(len(headers))]
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        _cell(hdr[i], h, bold=True, size=9.5, color=RGBColor(0xFF, 0xFF, 0xFF),
              align=_AL[aligns[i]])
        _shade(hdr[i], "3F4046")
    hdr_pr = table.rows[0]._tr.get_or_add_trPr()
    repeat = hdr_pr.makeelement(qn("w:tblHeader"), {})
    repeat.set(qn("w:val"), "true")
    hdr_pr.append(repeat)
    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        tr_pr = table.rows[-1]._tr.get_or_add_trPr()
        cant_split = tr_pr.makeelement(qn("w:cantSplit"), {})
        tr_pr.append(cant_split)
        for i, val in enumerate(row):
            col = colors[r_idx][i] if colors else _INK
            _cell(cells[i], val, size=9.5, color=col, align=_AL[aligns[i]])
    _fix_widths(table, widths)
    return table


def _figure_row(doc, specs) -> bool:
    """표 경계 없이 그래프 두 개를 한 문단에 나란히 배치한다."""
    if not specs:
        return False
    for row_start in range(0, len(specs), 2):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(7)
        para.paragraph_format.space_after = Pt(7)
        for idx in range(row_start, min(row_start + 2, len(specs))):
            image_path, _caption, _width = specs[idx]
            if idx > row_start:
                para.add_run("  ")
            para.add_run().add_picture(str(image_path), width=Cm(7.35))
    return True


def _result_figure_specs(out_path: Path):
    """검토서에 넣을 수 있는 대시보드 그래프 목록을 반환한다."""
    base = out_path.parent
    candidates = [
        (base / "identification" / "attribute_grade_distribution.png",
         "그림 1. 속성별 최종 식별 등급 분포", 7.25),
        (base / "identification" / "uniqueness_correctness_scatter.png",
         "그림 2. 상위 조합의 유일성과 정확 매칭률 관계", 7.25),
        (base / "aggregation" / "risk_distribution_pie.png",
         "그림 3. 집계 추론 위험의 최종 등급 분포", 7.25),
        (base / "aggregation" / "top20_high_risk_combinations.png",
         "그림 4. 평균 정확도 기준 상위 집계 위험 조합", 7.25),
        (base / "exposure" / "exposure_heatmap.png",
         "그림 5. 민감속성별 내부 및 외부 노출 지표", 7.25),
        (base / "exposure" / "risk_curve.png",
         "그림 6. 속성별 통합 노출 위험과 데이터셋 등급", 7.25),
    ]
    return [(p, c, w) for p, c, w in candidates if p.exists()]


def _module_figure_specs(out_path: Path, module: str):
    parts = {
        "identification": ("identification",),
        "aggregation": ("aggregation",),
        "exposure": ("exposure",),
    }
    folder = parts[module][0]
    return [spec for spec in _result_figure_specs(out_path) if spec[0].parent.name == folder]


def _grade_label(g):
    return _GRADE_LABEL.get((g or "").upper(), "-")


def _grade_color(g):
    g = (g or "").upper()
    return _HIGH if g == "H" else _MID if g == "M" else _LOW if g == "L" else _INK


def _raw_color(label):
    return _HIGH if label == "고위험" else _MID if label in ("위험", "중간") else _LOW


def generate_adequacy_report(result: dict, out_path: Path) -> Path:
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = _FONT
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), _FONT)

    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    dataset = result.get("dataset", {})
    gate = result.get("gate", {})
    modules = result.get("modules", {})
    ident = modules.get("identification") or {}
    agg = modules.get("aggregation") or {}
    exposure = modules.get("exposure") or {}
    integ = result.get("integrated") or {}
    halted = bool(result.get("pipeline_halted"))

    # 표제
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    trun = title.add_run("가명정보 처리 적정성 검토서")
    trun.font.size = Pt(19)
    trun.font.bold = True
    trun.font.color.rgb = _INK
    _set_kfont(trun)
    title.paragraph_format.space_after = Pt(3)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    srun = sub.add_run("개인정보 보호법 제28조의2에 따른 가명정보 활용 적정성 검토")
    srun.font.size = Pt(10)
    srun.font.color.rgb = _MUTED
    _set_kfont(srun)
    sub.paragraph_format.space_after = Pt(16)

    # 1. 검토 개요
    _heading(doc, 1, "검토 개요")
    _kv_table(doc, [
        ("문서명", "가명정보 처리 적정성 검토서"),
        ("검토 대상 데이터셋", dataset.get("filename", "-")),
        ("처리 목적", gate.get("purpose_label", "-")),
        ("검토 기준", "개인정보 보호법 제28조의2"),
        ("작성일", date.today().strftime("%Y년 %m월 %d일")),
    ])
    _p(doc, "본 검토서는 대상 데이터셋의 처리 목적 적합성과 재식별 위험성, 집계 추론 위험성, 노출 위험성을 "
            "평가하고 이를 통합한 위험도를 근거로 가명정보 활용의 적정성에 대한 의견과 권고를 "
            "제시한다.", size=10, before=6)

    # 2. 대상 데이터 개요
    _heading(doc, 2, "대상 데이터 개요")
    _kv_table(doc, [
        ("파일명", dataset.get("filename", "-")),
        ("레코드 수", f"{dataset.get('rows', 0):,} 건"),
        ("컬럼 수", f"{dataset.get('columns', 0)} 개"),
        ("가명처리 대상", ", ".join(gate.get("pseudonym_targets", []) or []) or "-"),
    ])

    # 3. 처리 목적 적합성 검토
    _heading(doc, 3, "처리 목적 적합성 검토")
    _p(doc, "가명정보는 통계작성과 과학적 연구, 공익적 기록보존 목적의 범위에서만 처리한다. "
            "본 단계는 활용 계획서와 데이터셋을 대조하여 목적 적합 여부를 판정하는 사전 관문이다. "
            "통과한 경우에만 재식별, 집계 및 노출 위험 분석을 실행한다. 목적 적합성 판정은 활용 가능 "
            "여부를 가르는 판단이므로 통합 위험도 점수에 합산하지 않는다.", size=10)
    _kv_table(doc, [
        ("처리 목적", gate.get("purpose_label", "-")),
        ("판정 결과", gate.get("decision", "-")),
    ])
    reasons = gate.get("rejection_reasons") or []
    unmet = gate.get("unmet_items") or []
    if gate.get("passed"):
        _p(doc, "명시된 처리 목적과 활용 계획서의 적합성을 확인한다. 목적 외 사용에 해당하지 않는 "
                "것으로 판정하고 후속 위험 분석을 진행한다.", size=10, before=4)
    else:
        _p(doc, "아래 사유로 목적 적합성을 충족하지 않아 반려로 판정한다. 반려 시 후속 재식별, "
                "집계 및 노출 위험 분석과 통합 위험도 산출을 실행하지 않는다.",
           size=10, color=_HIGH, before=4)
        if reasons:
            _grid_table(doc, ["No.", "반려 사유"],
                        [[str(i + 1), r] for i, r in enumerate(reasons)],
                        widths=[1.6, _CONTENT_W - 1.6], aligns=["center", "left"])
        if unmet:
            _p(doc, "미충족 항목은 " + ", ".join(map(str, unmet)) + "이다.", size=9.5, color=_MUTED, before=4)

    if halted:
        _heading(doc, 4, "종합 의견")
        _p(doc, "본 데이터셋은 처리 목적 적합성 판정에서 반려한다. 현재 계획서 기준으로는 가명정보 "
                "활용이 적정하지 않은 것으로 판단한다. 반려 사유를 해소하도록 활용 계획서를 보완한 "
                "뒤 재검토를 신청한다.", size=10)
        doc.save(str(out_path))
        return out_path

    # 4. 재식별 위험성 평가
    _heading(doc, 4, "재식별 위험성 평가")
    _p(doc, "준식별자 속성과 속성 조합의 유일성, 정확 매칭률을 기준으로 재식별 가능성을 평가한다. "
            "속성별 최종 등급은 개별 위험 등급과 상위 조합 등장 빈도 기반 등급 중 더 높은 값으로 "
            "정한다. 데이터셋 등급은 속성 등급의 최빈값으로 확정한다.", size=10)
    gc = ident.get("attribute_grade_counts", {}) or {}
    _kv_table(doc, [
        ("데이터셋 재식별 등급", _grade_label(ident.get("dataset_grade"))),
        ("전체 유일성", f"{float(ident.get('uniqueness', 0)) * 100:.1f}%"),
        ("전체 정확 매칭률", f"{float(ident.get('correctness', 0)) * 100:.1f}%"),
        ("속성 등급 분포", f"고위험 {gc.get('H', 0)}개, 중위험 {gc.get('M', 0)}개, 저위험 {gc.get('L', 0)}개"),
    ])
    top_combos = (ident.get("top_combinations") or [])[:5]
    if top_combos:
        _p(doc, "상위 재식별 위험 조합은 다음과 같다.", size=10, before=6, after=4)
        rows = [[" + ".join(c.get("combination", [])),
                 str(c.get("size", len(c.get("combination", [])))),
                 f"{float(c.get('uniqueness', 0)):.3f}",
                 f"{float(c.get('correctness', 0)):.3f}"] for c in top_combos]
        _grid_table(doc, ["준식별자 조합", "크기", "유일성", "정확 매칭률"],
                    rows, widths=[8.4, 1.8, 3.0, 3.0],
                    aligns=["left", "center", "center", "center"])
    _figure_row(doc, _module_figure_specs(out_path, "identification"))

    # 5. 집계 추론 위험성 평가
    _heading(doc, 5, "집계 추론 위험성 평가")
    _p(doc, "특성 조합으로부터 민감정보를 추론할 수 있는 위험을 평가한다. 10종 분류기와 변곡점과 변화율의 두 기준으로 "
            "조합별 위험 라벨을 산출한다. 최빈값으로 네 단계 위험 라벨을 정한 뒤 고위험, 중위험, "
            "저위험의 세 등급으로 재집계한다. 데이터셋 등급은 최다 비율 등급으로 "
            "확정한다.", size=10)
    hml = agg.get("hml_distribution", {}) or {}
    _kv_table(doc, [
        ("민감정보 타깃", agg.get("target_col", "-")),
        ("데이터셋 집계 등급", _grade_label(agg.get("dataset_grade"))),
        ("분석 조합 수", f"{agg.get('total_combinations', 0):,} 개"),
        ("등급 분포", "고위험 {h}%, 중위험 {m}%, 저위험 {l}%".format(
            h=_pct(hml.get("H")), m=_pct(hml.get("M")), l=_pct(hml.get("L")))),
        ("사용 모델", f"{len(agg.get('models', []) or [])}종, 조합 크기 {agg.get('combo_range', '-')}개"),
    ])
    agg_top = (agg.get("top_combinations") or [])[:5]
    if agg_top:
        _p(doc, "상위 집계 추론 위험 조합은 다음과 같다.", size=10, before=6, after=4)
        rows, colors = [], []
        for c in agg_top:
            risk = str(c.get("Final_Risk", ""))
            risk_display = {"고위험": "고위험", "위험": "고위험", "중간": "중위험", "안심": "저위험"}.get(risk, risk)
            combo_disp = str(c.get("Feature_Combination", "")).replace("+", " + ")
            rows.append([combo_disp, str(c.get("Num_Features", "")),
                         f"{float(c.get('Mean_Accuracy', 0)):.3f}", risk_display])
            colors.append([_INK, _INK, _INK, _raw_color(risk)])
        _grid_table(doc, ["특성 조합", "특성 수", "평균 정확도", "최종 위험"],
                    rows, widths=[9.0, 1.8, 2.7, 2.7], colors=colors,
                    aligns=["left", "center", "center", "center"])
    _figure_row(doc, _module_figure_specs(out_path, "aggregation"))

    # 6. 노출 위험성 평가
    _heading(doc, 6, "노출 위험성 평가")
    _p(doc, "민감정보를 탐지하고 속성별 내부 노출과 외부 연결 가능성을 평가한다. 속성별 통합 노출 위험이 "
            "기준값을 초과하는 비율에 따라 데이터셋을 고위험, 중위험, 저위험으로 판정한다. 기준을 초과한 "
            "속성은 위험값이 높은 순서로 재가명처리 우선순위를 정한다.", size=10)
    _kv_table(doc, [
        ("데이터셋 노출 등급", _grade_label(exposure.get("dataset_grade"))),
        ("탐지된 민감속성", f"{len(exposure.get('sensitive_columns', []) or [])} 개"),
        ("위험 임계 초과", f"{exposure.get('n_exceed', 0)} 개"),
        ("임계 초과 비율", f"{float(exposure.get('rho', 0)) * 100:.1f}%"),
        ("검수 필요 컬럼", f"{len(exposure.get('review_columns', []) or [])} 개"),
    ])
    exposure_top = (exposure.get("top_actions") or [])[:6]
    if exposure_top:
        _p(doc, "재가명처리 우선 검토 컬럼은 다음과 같다.", size=10, before=6, after=4)
        _grid_table(doc, ["순서", "컬럼", "노출 위험"],
                    [[str(item.get("rank", "")), str(item.get("attribute", "")),
                      f"{float(item.get('risk', 0)) * 100:.1f}%"] for item in exposure_top],
                    widths=[1.8, 10.8, 3.6], aligns=["center", "left", "center"])
    _figure_row(doc, _module_figure_specs(out_path, "exposure"))

    # 7. 통합 위험도 산출
    _heading(doc, 7, "통합 위험도 산출")
    _kv_table(doc, [
        ("재식별 위험 등급", _grade_label(ident.get("dataset_grade"))),
        ("집계 추론 위험 등급", _grade_label(agg.get("dataset_grade"))),
        ("노출 위험 등급", _grade_label(exposure.get("dataset_grade"))),
        ("통합 위험도", _grade_label(integ.get("grade"))),
    ])

    # 8. 종합 의견 및 권고
    _heading(doc, 8, "종합 의견 및 권고")
    for para in _opinion_paragraphs(integ):
        _p(doc, para, size=10, after=6)
    recs = _recommendations(ident, agg, exposure)
    if recs:
        _p(doc, "권고 사항은 다음과 같다.", size=10, bold=False, before=4, after=4)
        _grid_table(doc, ["No.", "권고 사항"],
                    [[str(i + 1), r] for i, r in enumerate(recs)],
                    widths=[1.6, _CONTENT_W - 1.6], aligns=["center", "left"])

    doc.save(str(out_path))
    return out_path


def _pct(entry):
    if isinstance(entry, dict):
        return f"{float(entry.get('pct', 0)):.1f}"
    return "0.0"


def _opinion_paragraphs(integ):
    grade = (integ.get("grade") or "").upper()
    out = []
    if grade == "H":
        out.append("통합 위험도가 고위험이다. 현재 상태로는 재식별 또는 민감정보 추론 가능성이 높다. "
                   "아래 권고에 따른 추가 가명처리와 관리적 보호조치를 적용하기 전에는 활용 적정성을 "
                   "확보하기 어렵다.")
    elif grade == "M":
        out.append("통합 위험도가 중위험이다. 재식별, 집계 및 노출 상세에서 확인한 상위 위험 속성과 조합에 "
                   "한해 보완 가명처리를 적용하면 활용 적정성을 확보한다.")
    elif grade == "L":
        out.append("통합 위험도가 저위험이다. 명시된 처리 목적 범위 내에서 활용하는 것이 적정하다. "
                   "활용 종료 후 가명정보 파기 절차를 준수한다.")
    else:
        out.append("일부 위험 분석이 완료되지 않아 통합 위험도를 판정할 수 없다. "
                   "누락된 분석을 재실행한 뒤 적정성을 다시 검토한다.")
    out.append("본 검토는 제출된 데이터셋과 활용 계획서를 기준으로 한 판단이다. 데이터 또는 활용 "
               "범위가 변경되는 경우 재검토한다.")
    return out


def _recommendations(ident, agg, exposure):
    recs = []
    high_attrs = [a.get("attribute") for a in (ident.get("attributes") or [])
                  if str(a.get("final_grade", "")).upper() == "H"]
    if high_attrs:
        recs.append(f"재식별 고위험 속성({', '.join(map(str, high_attrs[:6]))})에 대해 일반화와 "
                    "범주화 또는 마스킹 수준 상향을 검토한다.")
    agg_top = [str(c.get("Feature_Combination", "")).replace("+", " + ") for c in (agg.get("top_combinations") or [])[:3]]
    tcol = agg.get("target_col")
    if agg_top and tcol:
        recs.append(f"민감정보({tcol}) 추론 고위험 조합({' / '.join(agg_top)})의 동시 제공 여부를 "
                    "재검토한다.")
    exposure_top = [str(item.get("attribute", "")) for item in (exposure.get("top_actions") or [])[:6]]
    if exposure_top:
        recs.append(f"노출 위험 임계를 초과한 컬럼({', '.join(exposure_top)})부터 재가명처리하고 위험도를 재평가한다.")
    recs.append("가명처리 대상 컬럼의 처리 이력과 재식별 금지 서약 등 관리적 보호조치를 문서화한다.")
    recs.append("활용 종료 후 가명정보 파기 절차와 결합 시 추가 재검토 요건을 계획서에 반영한다.")
    return recs
