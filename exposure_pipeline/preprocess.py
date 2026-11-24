# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (Dict, List, Literal, Optional, Sequence, Set, Tuple)


"""preprocess.py — 형태소/토큰화 · NER · keep/ignore 전처리"""

from .config import CATEGORIES, NEGATIVE_CODE

# ===  preprocess.py

"""
전처리: 토큰화 → 불용어 제거 → 어간 추출 → 개체명 인식  (한국어 + 영어)
(Jaikumar et al., 2023 §III "Data Pre-processing" 구성)

핵심 설계: policy = "keep" | "ignore"
=====================================================================
NER이 찾아낸 보호 구간(protected span)을 어떻게 다룰지에 대한 정책이다.
두 모드는 **완전히 동일한 파이프라인**을 타고 이 한 가지만 다르다.

  keep   : 보호 구간을 하나의 원자 토큰으로 보존.
           불용어 제거·어간 추출을 적용하지 않는다.
  ignore : 보호 구간을 특별 취급하지 않는다.
           일반 텍스트와 똑같이 분해·정규화된다.

공정성 보장 3원칙
  (1) NER은 **두 모드 모두에서 항상 실행**된다. 탐지 결과·탐지 비용 동일.
  (2) 형태소 분석기·NER 사전·불용어 목록은 두 모드가 **같은 객체를 공유**한다.
      (make_baselines()가 인스턴스를 공유시켜 준다)
  (3) 다운스트림(TF-IDF 설정, RF 하이퍼파라미터, 임계값 튜닝 방식)도 동일.
  ⇒ 성능 차이는 온전히 보호 정책 효과로 귀속된다. ignore가 대조군.
"""


Policy = Literal["keep", "ignore"]

# ---------------------------------------------------------------------------
# 불용어
# ---------------------------------------------------------------------------
KO_STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "및", "또는", "그리고", "하지만",
    "에서", "에게", "으로", "로서", "로써", "이다", "있다", "없다", "하다",
    "되다", "관한", "대한", "위한", "통해", "따라", "경우", "때문", "여부",
    "정보", "자료", "항목", "내용", "구분", "값", "기타", "해당",
}
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "as", "is", "are", "was", "were", "be", "been", "being", "with", "from",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "which", "who",
    "what", "when", "where", "how", "not", "no", "but", "if", "than", "then",
    "so", "such", "can", "could", "may", "might", "will", "would", "shall",
    "should", "do", "does", "did", "has", "have", "had", "about", "into",
    "over", "under", "per", "via", "col", "column", "field", "value", "data",
    "info", "information", "record", "table", "item",
}
STOPWORDS = KO_STOPWORDS | EN_STOPWORDS

_HANGUL_JOSA = re.compile(
    r"(은|는|이|가|을|를|의|에|에서|에게|으로|로|와|과|도|만|까지|부터|"
    r"이나|나|든지|라도|처럼|보다|마다|조차|밖에)$")
_HANGUL_EOMI = re.compile(
    r"(하였|했|하는|한다|합니다|였다|이었|되었|된다|되는|하고|하며|"
    r"하여|해서|이며|이고|입니다|이다|다)$")
_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    """'ko' | 'en' | 'mixed'"""
    ko = len(_HANGUL_RE.findall(text))
    en = len(_LATIN_RE.findall(text))
    if ko and en:
        return "mixed"
    return "ko" if ko else "en"


# ---------------------------------------------------------------------------
# 영어 어간 추출
# ---------------------------------------------------------------------------
class EnglishStemmer:
    """nltk SnowballStemmer 우선, 없으면 경량 접미사 절단."""

    _SUFFIXES = ("ational", "tional", "ization", "iveness", "fulness",
                 "ousness", "ization", "ational", "alism", "ities", "ivity",
                 "ement", "ments", "ation", "ingly", "edly", "ness", "ment",
                 "ance", "ence", "able", "ible", "ical", "ing", "ers", "est",
                 "ies", "ive", "ful", "ous", "ity", "ate", "ise", "ize",
                 "ers", "ed", "ly", "es")

    def __init__(self):
        self._impl = None
        try:
            from nltk.stem.snowball import SnowballStemmer  # type: ignore
            self._impl = SnowballStemmer("english")
        except Exception:
            self._impl = None
        self.backend = "snowball" if self._impl else "rule"

    def stem(self, tok: str) -> str:
        if self._impl:
            return self._impl.stem(tok)
        t = tok.lower()
        if len(t) <= 4:
            return t
        for suf in self._SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
        else:
            if t.endswith("sses"):
                t = t[:-2]
            elif t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
        return t


# ---------------------------------------------------------------------------
# 형태소 분석기 (kiwipiepy > konlpy.Okt > 정규식 폴백) + 영어 처리
# ---------------------------------------------------------------------------
class MorphAnalyzer:
    CONTENT_TAGS = {"NNG", "NNP", "NNB", "NR", "SL", "SH", "SN",
                    "VV", "VA", "XR", "Noun", "Verb", "Adjective", "Alpha",
                    "Number", "Foreign", "EN", "NUM"}

    def __init__(self, backend: str = "auto"):
        self.backend = None
        self._impl = None
        self.en = EnglishStemmer()
        for cand in (["kiwi", "okt"] if backend == "auto" else [backend]):
            try:
                if cand == "kiwi":
                    from kiwipiepy import Kiwi  # type: ignore
                    self._impl = Kiwi(); self.backend = "kiwi"; break
                if cand == "okt":
                    from konlpy.tag import Okt  # type: ignore
                    self._impl = Okt(); self.backend = "okt"; break
            except Exception:
                continue
        if self.backend is None:
            self.backend = "regex"

    def analyze(self, text: str, stem: bool = True) -> List[Tuple[str, str]]:
        lang = detect_lang(text)
        if lang == "en":
            return self._analyze_en(text, stem)
        if lang == "mixed":
            out = []
            for chunk in re.findall(r"[가-힣\s\d]+|[A-Za-z\s\d]+", text):
                if not chunk.strip():
                    continue
                out.extend(self.analyze(chunk, stem))
            return out
        return self._analyze_ko(text, stem)

    def _analyze_en(self, text: str, stem: bool) -> List[Tuple[str, str]]:
        out = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            if tok[0].isdigit():
                out.append((tok, "NUM")); continue
            tok = tok.lower()
            out.append((self.en.stem(tok) if stem else tok, "EN"))
        return out

    def _analyze_ko(self, text: str, stem: bool) -> List[Tuple[str, str]]:
        if self.backend == "kiwi":
            out = []
            for tok in self._impl.tokenize(text):
                form = tok.form if stem else text[tok.start: tok.start + tok.len]
                out.append((form, tok.tag))
            return out
        if self.backend == "okt":
            return list(self._impl.pos(text, norm=True, stem=stem))
        out = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            tag = ("Number" if tok[0].isdigit()
                   else "EN" if tok[0].isascii() else "Noun")
            if stem and tag == "Noun":
                tok = self._rule_stem(tok)
            elif tag == "EN":
                tok = self.en.stem(tok.lower()) if stem else tok.lower()
            out.append((tok, tag))
        return out

    @staticmethod
    def _rule_stem(tok: str) -> str:
        prev = None
        while prev != tok and len(tok) > 2:
            prev = tok
            tok = _HANGUL_EOMI.sub("", tok)
            tok = _HANGUL_JOSA.sub("", tok)
        return tok


# ---------------------------------------------------------------------------
# 개체명 인식
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    surface: str
    label: str


class GazetteerNER:
    """카테고리 키워드 사전 기반 경량 NER (한/영 동시).

    영문 용어는 \\b 경계를 강제한다. 그렇지 않으면 'sex'가 'sexual' 안에서,
    'id'가 'identity' 안에서 잘못 매칭된다. 한글은 경계 문자가 없으므로 미적용.
    """

    def __init__(self, extra: Optional[Dict[str, str]] = None,
                 lang: str = "both"):
        idx: Dict[str, str] = {}
        for c in CATEGORIES.values():
            if c.code == NEGATIVE_CODE:
                continue
            kws = ((c.keywords_ko if lang in ("ko", "both") else ()) +
                   (c.keywords_en if lang in ("en", "both") else ()))
            for kw in kws:
                idx.setdefault(kw.lower(), c.code)
        if extra:
            idx.update({k.lower(): v for k, v in extra.items()})
        self._label_of = idx
        terms = sorted(idx.keys(), key=len, reverse=True)   # 최장일치
        parts = [(r"\b" + re.escape(t) + r"\b") if t.isascii() else re.escape(t)
                 for t in terms]
        self._pattern = re.compile("|".join(parts), re.IGNORECASE) if parts else None

    def detect(self, text: str) -> List[Span]:
        if not self._pattern:
            return []
        spans, taken = [], []
        for m in self._pattern.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            surf = m.group()
            spans.append(Span(m.start(), m.end(), surf,
                              self._label_of.get(surf.lower(), "MISC")))
            taken.append((m.start(), m.end()))
        return sorted(spans, key=lambda s: s.start)


class HFNer:
    """transformers 기반 NER (선택). 미설치 시 조용히 비활성화."""

    def __init__(self, model: str = "Davlan/xlm-roberta-base-ner-hrl"):
        self._pipe = None
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline("ner", model=model, aggregation_strategy="simple")
        except Exception:
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def detect(self, text: str) -> List[Span]:
        if not self._pipe:
            return []
        try:
            return [Span(int(e["start"]), int(e["end"]), e["word"],
                         e["entity_group"]) for e in self._pipe(text)]
        except Exception:
            return []


class CompositeNER:
    def __init__(self, *recognizers):
        self.recognizers = [r for r in recognizers if r is not None]

    def detect(self, text: str) -> List[Span]:
        spans: List[Span] = []
        for r in self.recognizers:
            spans.extend(r.detect(text))
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        merged: List[Span] = []
        for s in spans:
            if merged and s.start < merged[-1].end:
                continue
            merged.append(s)
        return merged


# ---------------------------------------------------------------------------
# 전처리기
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    policy: Policy = "keep"
    lang: str = "both"             # "ko" | "en" | "both"
    lowercase: bool = True
    remove_stopwords: bool = True
    stemming: bool = True
    use_ner: bool = True
    content_pos_only: bool = True
    min_token_len: int = 2
    protected_prefix: str = ""


@dataclass
class PreprocessResult:
    tokens: List[str]
    protected: List[str] = field(default_factory=list)
    ne_labels: List[str] = field(default_factory=list)
    policy: str = "keep"
    lang: str = "ko"

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


class Preprocessor:
    def __init__(self,
                 config: Optional[PreprocessConfig] = None,
                 analyzer: Optional[MorphAnalyzer] = None,
                 ner: Optional[object] = None,
                 stopwords: Optional[Set[str]] = None):
        self.cfg = config or PreprocessConfig()
        self.analyzer = analyzer or MorphAnalyzer()
        self.ner = ner or GazetteerNER(lang=self.cfg.lang)
        self.stopwords = stopwords if stopwords is not None else STOPWORDS

    def _normalize(self, text: str) -> str:
        t = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))   # camelCase
        t = re.sub(r"[_\-./]+", " ", t)                          # snake/kebab
        t = re.sub(r"\s+", " ", t).strip()
        return t.lower() if self.cfg.lowercase else t

    def _tokenize_free(self, segment: str) -> List[str]:
        pairs = self.analyzer.analyze(segment, stem=self.cfg.stemming)
        out = []
        for tok, tag in pairs:
            if self.cfg.content_pos_only and tag not in MorphAnalyzer.CONTENT_TAGS:
                continue
            tok = tok.lower() if self.cfg.lowercase else tok
            if self.cfg.remove_stopwords and tok in self.stopwords:
                continue
            if len(tok) < self.cfg.min_token_len and not tok.isdigit():
                continue
            out.append(tok)
        return out

    def run(self, text: str) -> PreprocessResult:
        if not text:
            return PreprocessResult([], [], [], self.cfg.policy, "ko")
        norm = self._normalize(text)
        lang = detect_lang(norm)

        # NER은 정책과 무관하게 항상 수행 (공정성 원칙 1)
        spans = self.ner.detect(norm) if self.cfg.use_ner else []
        labels = [s.label for s in spans]

        if self.cfg.policy == "ignore" or not spans:
            return PreprocessResult(self._tokenize_free(norm), [], labels,
                                    self.cfg.policy, lang)

        tokens: List[str] = []
        protected: List[str] = []
        cursor = 0
        for sp in spans:
            if sp.start > cursor:
                tokens.extend(self._tokenize_free(norm[cursor:sp.start]))
            atom = self.cfg.protected_prefix + sp.surface.replace(" ", "_")
            tokens.append(atom)          # 불용어/어간 미적용
            protected.append(atom)
            cursor = sp.end
        if cursor < len(norm):
            tokens.extend(self._tokenize_free(norm[cursor:]))

        return PreprocessResult(tokens, protected, labels, self.cfg.policy, lang)

    def transform(self, texts: Sequence[str]) -> List[str]:
        return [self.run(t).text for t in texts]


def make_baselines(**kwargs) -> Dict[str, Preprocessor]:
    """keep / ignore 두 베이스라인을 동일 자원으로 생성 (공정성 원칙 2).

    analyzer·ner·stopwords 인스턴스를 공유시켜 두 모드의 유일한 차이가
    policy 하나가 되도록 강제한다.
    """
    lang = kwargs.get("lang", "both")
    analyzer = MorphAnalyzer()
    ner = GazetteerNER(lang=lang)
    sw = STOPWORDS
    return {
        p: Preprocessor(PreprocessConfig(policy=p, **kwargs),
                        analyzer=analyzer, ner=ner, stopwords=sw)
        for p in ("keep", "ignore")
    }



