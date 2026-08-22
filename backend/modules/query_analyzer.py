"""Query Analyzer — deterministic query classification and adaptive retrieval parameter selection.

This module implements the QueryAnalyzer pipeline stage that sits between the STT layer and
the retrieval layer. It inspects the incoming query and:

1. Detects the query language (Hindi / Hinglish / English)
2. Classifies the query type (factual / descriptive / comparison / navigational / conversational)
3. Recommends retrieval parameters (alpha, top_k) based on query type
4. Extracts lightweight keywords for sparse retrieval prioritization
5. Estimates query complexity

All logic is deterministic — no external API calls are required.

The QueryAnalysis object is attached to the query response under ``telemetry.query_analysis``,
making it visible in Judge Mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryLanguage(str, Enum):
    """Detected primary language of the query."""
    HINDI = "hi"
    HINGLISH = "hi-en"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """Semantic classification of the query intent."""
    FACTUAL = "factual"           # Short-answer, who/what/when/where
    DESCRIPTIVE = "descriptive"   # How/explain/describe
    COMPARISON = "comparison"     # Compare / versus / difference
    NAVIGATIONAL = "navigational" # Find / list / show / which
    CONVERSATIONAL = "conversational"  # Chat-like, vague


class QueryComplexity(str, Enum):
    SIMPLE = "simple"       # ≤5 tokens
    MODERATE = "moderate"   # 6–15 tokens
    COMPLEX = "complex"     # >15 tokens


@dataclass
class RetrievalStrategy:
    """Recommended retrieval parameters derived from query analysis."""
    alpha: float               # Dense weight [0,1]; 1.0=dense-only, 0.0=sparse-only
    top_k: int                 # Candidate count to retrieve before reranking
    strategy_name: str         # Human-readable label for Judge Mode
    rationale: str             # Why this strategy was chosen


@dataclass
class QueryAnalysis:
    """Full analysis output for a single query."""
    original_query: str
    normalized_query: str
    language: QueryLanguage
    query_type: QueryType
    complexity: QueryComplexity
    keywords: list[str]
    retrieval_strategy: RetrievalStrategy
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for API responses and Judge Mode."""
        return {
            "language": self.language.value,
            "query_type": self.query_type.value,
            "complexity": self.complexity.value,
            "keywords": self.keywords,
            "retrieval_strategy": {
                "alpha": self.retrieval_strategy.alpha,
                "top_k": self.retrieval_strategy.top_k,
                "strategy_name": self.retrieval_strategy.strategy_name,
                "rationale": self.retrieval_strategy.rationale,
            },
            "normalized_query": self.normalized_query,
        }


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

# Unicode block for Devanagari script (Hindi, Marathi, Sanskrit, etc.)
_DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")

# Common Hindi stopwords / particles that appear even in Hinglish
_HINDI_FUNCTION_WORDS = frozenset({
    "kya", "hai", "hain", "kaise", "kyun", "kaun", "kab", "kahan",
    "mein", "ka", "ki", "ke", "ko", "se", "par", "aur", "ya",
    "nahi", "nahin", "bhi", "to", "jo", "yeh", "woh", "iska",
})

# Common English factual question starters
_FACTUAL_EN = frozenset({
    "what", "who", "when", "where", "which", "how many", "how much",
    "is", "are", "was", "were", "does", "did", "has", "have",
})
_DESCRIPTIVE_EN = frozenset({
    "how", "explain", "describe", "tell me about", "what is the difference",
    "elaborate", "summarize", "overview", "detail",
})
_COMPARISON_EN = frozenset({
    "compare", "versus", "vs", "difference between", "better", "worse",
    "contrast", "advantages", "disadvantages", "pros", "cons",
})
_NAVIGATIONAL_EN = frozenset({
    "find", "show", "list", "give me", "get", "fetch", "retrieve",
    "search for", "look for", "display",
})

# Hindi question words
_FACTUAL_HI = frozenset({
    "क्या", "कौन", "कब", "कहाँ", "कहां", "किसने", "कितना", "कितनी",
})
_DESCRIPTIVE_HI = frozenset({
    "कैसे", "कैसा", "कैसी", "बताइए", "समझाइए", "विवरण", "वर्णन",
})
_COMPARISON_HI = frozenset({
    "तुलना", "अंतर", "फर्क", "बेहतर", "बेहतरीन", "बनाम",
})
_NAVIGATIONAL_HI = frozenset({
    "दिखाएं", "दिखाओ", "खोजो", "ढूंढो", "सूची", "बताओ",
})


def _detect_language(query: str) -> QueryLanguage:
    """Detect the primary language of a query using script analysis and lexical heuristics."""
    if not query:
        return QueryLanguage.UNKNOWN

    devanagari_chars = len(_DEVANAGARI_PATTERN.findall(query))
    total_chars = len(query.replace(" ", ""))

    # Pure Hindi: ≥30% Devanagari characters
    if total_chars > 0 and devanagari_chars / total_chars >= 0.30:
        return QueryLanguage.HINDI

    # Hinglish: has some Devanagari OR uses Hindi function words in Latin script.
    # Tokenize on word boundaries (not naive .split()) so trailing punctuation does not
    # hide a function word — e.g. "hai?" must still match "hai". Without this, short
    # Roman-Hindi queries like "ThinkZen kaise kaam karti hai?" fell below the 2-hit
    # threshold and were misdetected as English.
    tokens = set(re.findall(r"[\wऀ-ॿ]+", query.lower()))
    hindi_function_hits = tokens & _HINDI_FUNCTION_WORDS
    if devanagari_chars > 0 or len(hindi_function_hits) >= 2:
        return QueryLanguage.HINGLISH

    return QueryLanguage.ENGLISH


def _classify_query_type(query: str, language: QueryLanguage) -> QueryType:
    """Classify query intent from lexical patterns."""
    q_lower = query.lower().strip()
    tokens = set(q_lower.split())

    if language in (QueryLanguage.HINDI, QueryLanguage.HINGLISH):
        # Check Hindi-specific patterns first
        if _FACTUAL_HI & set(query.split()):
            return QueryType.FACTUAL
        if _DESCRIPTIVE_HI & set(query.split()):
            return QueryType.DESCRIPTIVE
        if _COMPARISON_HI & set(query.split()):
            return QueryType.COMPARISON
        if _NAVIGATIONAL_HI & set(query.split()):
            return QueryType.NAVIGATIONAL

    # English / Hinglish fallback with English keywords
    first_two = " ".join(q_lower.split()[:2])
    first_word = q_lower.split()[0] if q_lower.split() else ""

    # Check multi-word phrases first
    if any(phrase in q_lower for phrase in ("difference between", "compare", "versus", " vs ", "pros and cons")):
        return QueryType.COMPARISON
    if any(phrase in q_lower for phrase in ("how to", "how do", "how does", "explain", "describe", "tell me about", "what is the", "what are")):
        if "difference" not in q_lower and "compare" not in q_lower:
            return QueryType.DESCRIPTIVE
    if first_word in {"find", "show", "list", "give", "fetch", "get", "search", "display"}:
        return QueryType.NAVIGATIONAL
    if first_word in {"what", "who", "when", "where", "which", "is", "are", "was", "were", "does", "did", "has", "have"}:
        return QueryType.FACTUAL
    if first_word == "how":
        return QueryType.DESCRIPTIVE

    # Hinglish: check English tokens in context
    if tokens & _COMPARISON_EN:
        return QueryType.COMPARISON
    if tokens & _NAVIGATIONAL_EN:
        return QueryType.NAVIGATIONAL
    if tokens & _FACTUAL_EN:
        return QueryType.FACTUAL

    return QueryType.CONVERSATIONAL


def _estimate_complexity(query: str) -> QueryComplexity:
    """Estimate query complexity from token count."""
    tokens = [t for t in query.split() if t.strip()]
    n = len(tokens)
    if n <= 5:
        return QueryComplexity.SIMPLE
    if n <= 15:
        return QueryComplexity.MODERATE
    return QueryComplexity.COMPLEX


def _extract_keywords(query: str, language: QueryLanguage, max_keywords: int = 8) -> list[str]:
    """Extract lightweight keywords for sparse retrieval focus.

    Removes common stop words and question particles; keeps content-bearing tokens.
    """
    # Common English stop words (minimal set)
    _EN_STOPS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can",
        "could", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "up", "about", "into", "through", "during",
        "what", "who", "when", "where", "which", "how", "why",
        "this", "that", "these", "those", "it", "its",
    }
    _HI_STOPS = {
        "है", "हैं", "था", "थी", "थे", "की", "का", "के", "को",
        "से", "में", "पर", "और", "या", "तो", "भी", "नहीं",
        "यह", "वह", "इस", "उस", "जो", "जिस",
    }

    stops = _EN_STOPS | _HI_STOPS if language in (QueryLanguage.HINDI, QueryLanguage.HINGLISH) else _EN_STOPS

    tokens = re.findall(r"[\w\u0900-\u097F]+", query.lower())
    keywords = [t for t in tokens if t not in stops and len(t) >= 2]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)
    return deduped[:max_keywords]


def _select_retrieval_strategy(
    query_type: QueryType,
    complexity: QueryComplexity,
    language: QueryLanguage,
) -> RetrievalStrategy:
    """Map query classification to adaptive retrieval parameters.

    Rationale:
    - FACTUAL: dense-heavy (0.7). Exact factual answers are semantically close to the query.
    - DESCRIPTIVE: balanced (0.5). Both semantic similarity and exact term coverage matter.
    - COMPARISON: sparse-heavy (0.35). Term overlap is critical for multi-entity comparison.
    - NAVIGATIONAL: dense-heavy (0.65). Intent is semantic; exact entity is secondary.
    - CONVERSATIONAL: balanced (0.5). Unknown intent; use balanced fusion.
    - COMPLEX queries: raise top_k to gather more candidates before reranking.
    """
    _top_k_base = {
        QueryComplexity.SIMPLE: 3,
        QueryComplexity.MODERATE: 5,
        QueryComplexity.COMPLEX: 8,
    }
    top_k = _top_k_base[complexity]

    if query_type == QueryType.FACTUAL:
        return RetrievalStrategy(
            alpha=0.70,
            top_k=top_k,
            strategy_name="dense-weighted-factual",
            rationale="Factual queries benefit from dense semantic similarity; alpha=0.70 favours embedding-based retrieval.",
        )
    if query_type == QueryType.DESCRIPTIVE:
        return RetrievalStrategy(
            alpha=0.50,
            top_k=max(top_k, 5),
            strategy_name="balanced-descriptive",
            rationale="Descriptive queries require both semantic coverage and term matching; balanced alpha=0.50.",
        )
    if query_type == QueryType.COMPARISON:
        return RetrievalStrategy(
            alpha=0.35,
            top_k=max(top_k, 5),
            strategy_name="sparse-weighted-comparison",
            rationale="Comparison queries need exact entity term coverage; sparse-heavy alpha=0.35.",
        )
    if query_type == QueryType.NAVIGATIONAL:
        return RetrievalStrategy(
            alpha=0.65,
            top_k=max(top_k, 5),
            strategy_name="dense-weighted-navigational",
            rationale="Navigational intent is semantic; alpha=0.65 prioritises embedding-based retrieval.",
        )
    # CONVERSATIONAL fallback
    return RetrievalStrategy(
        alpha=0.50,
        top_k=max(top_k, 5),
        strategy_name="balanced-conversational",
        rationale="Conversational/unclear intent; balanced fusion alpha=0.50 as safe default.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Deterministic query analyzer — no external API calls required.

    Usage::

        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("What is ThinkZen?")
        print(analysis.query_type)          # QueryType.FACTUAL
        print(analysis.retrieval_strategy.alpha)  # 0.70
        print(analysis.to_dict())
    """

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and return a full ``QueryAnalysis`` object.

        Args:
            query: Raw query string from voice transcript or text input.

        Returns:
            QueryAnalysis dataclass with language, type, complexity, keywords,
            and recommended retrieval strategy.
        """
        if not query or not query.strip():
            return QueryAnalysis(
                original_query=query,
                normalized_query="",
                language=QueryLanguage.UNKNOWN,
                query_type=QueryType.CONVERSATIONAL,
                complexity=QueryComplexity.SIMPLE,
                keywords=[],
                retrieval_strategy=RetrievalStrategy(
                    alpha=0.5,
                    top_k=5,
                    strategy_name="fallback-empty",
                    rationale="Empty query; using safe defaults.",
                ),
            )

        normalized = query.strip()
        language = _detect_language(normalized)
        query_type = _classify_query_type(normalized, language)
        complexity = _estimate_complexity(normalized)
        keywords = _extract_keywords(normalized, language)
        strategy = _select_retrieval_strategy(query_type, complexity, language)

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            language=language,
            query_type=query_type,
            complexity=complexity,
            keywords=keywords,
            retrieval_strategy=strategy,
        )


__all__ = [
    "QueryAnalyzer",
    "QueryAnalysis",
    "QueryLanguage",
    "QueryType",
    "QueryComplexity",
    "RetrievalStrategy",
]
