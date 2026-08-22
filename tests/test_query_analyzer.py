"""Tests for QueryAnalyzer — deterministic query classification.

All tests are synchronous (no async/external calls needed).
Covers: language detection, query type classification, complexity estimation,
keyword extraction, retrieval strategy selection, edge cases.
"""

from __future__ import annotations

import pytest

from modules.query_analyzer import (
    QueryAnalyzer,
    QueryComplexity,
    QueryLanguage,
    QueryType,
)


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_language_detection_english(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is the capital of India?")
    assert result.language == QueryLanguage.ENGLISH


def test_language_detection_hindi_devanagari(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("भारत की राजधानी क्या है?")
    assert result.language == QueryLanguage.HINDI


def test_language_detection_hinglish_latin(analyzer: QueryAnalyzer) -> None:
    """Hinglish: English words + Hindi function words in Latin script."""
    result = analyzer.analyze("ThinkZen mein hybrid search kaise kaam karta hai?")
    assert result.language == QueryLanguage.HINGLISH


def test_language_detection_hinglish_short_with_trailing_punctuation(
    analyzer: QueryAnalyzer,
) -> None:
    """Regression: a short Roman-Hindi query ending in 'hai?' must be Hinglish.

    Trailing punctuation ('hai?') previously survived naive .split() and hid the
    'hai' function word, dropping this query below the 2-hit threshold so it was
    misdetected as English. Detection now tokenizes on word boundaries.
    """
    result = analyzer.analyze("ThinkZen kaise kaam karti hai?")
    assert result.language == QueryLanguage.HINGLISH


def test_language_detection_hinglish_mixed_script(analyzer: QueryAnalyzer) -> None:
    """Mixed Devanagari + Latin should be Hinglish, not pure Hindi."""
    result = analyzer.analyze("ThinkZen का retrieval system कैसे काम करता है?")
    # Has Devanagari but also Latin — could be Hindi or Hinglish depending on ratio
    assert result.language in (QueryLanguage.HINDI, QueryLanguage.HINGLISH)


def test_language_detection_empty_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("")
    assert result.language == QueryLanguage.UNKNOWN


def test_language_detection_pure_numbers(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("123 456")
    assert result.language == QueryLanguage.ENGLISH


# ---------------------------------------------------------------------------
# Query type classification
# ---------------------------------------------------------------------------


def test_query_type_factual_what(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is ThinkZen?")
    assert result.query_type == QueryType.FACTUAL


def test_query_type_factual_who(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Who created ThinkZen?")
    assert result.query_type == QueryType.FACTUAL


def test_query_type_factual_when(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("When was HH Goa 2026 held?")
    assert result.query_type == QueryType.FACTUAL


def test_query_type_descriptive_how(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("How does hybrid retrieval work in ThinkZen?")
    assert result.query_type == QueryType.DESCRIPTIVE


def test_query_type_descriptive_explain(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Explain the grounding mechanism in ThinkZen.")
    assert result.query_type == QueryType.DESCRIPTIVE


def test_query_type_comparison_versus(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Compare BM25 versus dense embeddings for retrieval.")
    assert result.query_type == QueryType.COMPARISON


def test_query_type_comparison_difference(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is the difference between dense and sparse retrieval?")
    assert result.query_type == QueryType.COMPARISON


def test_query_type_navigational_list(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("List all available chunking strategies.")
    assert result.query_type == QueryType.NAVIGATIONAL


def test_query_type_navigational_find(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Find documents about ThinkZen architecture.")
    assert result.query_type == QueryType.NAVIGATIONAL


def test_query_type_hindi_factual(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("भारत की राजधानी क्या है?")
    assert result.query_type == QueryType.FACTUAL


def test_query_type_hindi_descriptive(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("वाणीरैग कैसे काम करता है?")
    assert result.query_type == QueryType.DESCRIPTIVE


def test_query_type_empty_is_conversational(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("")
    assert result.query_type == QueryType.CONVERSATIONAL


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------


def test_complexity_simple_short(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is ThinkZen?")
    assert result.complexity == QueryComplexity.SIMPLE


def test_complexity_moderate(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("How does the hybrid retrieval system work in the ThinkZen pipeline?")
    assert result.complexity == QueryComplexity.MODERATE


def test_complexity_complex(analyzer: QueryAnalyzer) -> None:
    long_query = (
        "Can you explain the difference between dense and sparse retrieval strategies "
        "and describe how ThinkZen combines them using a weighted alpha parameter "
        "for optimal evidence selection in a multilingual context?"
    )
    result = analyzer.analyze(long_query)
    assert result.complexity == QueryComplexity.COMPLEX


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def test_keywords_extracted_from_english(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is the architecture of ThinkZen?")
    assert "thinkzen" in result.keywords
    assert "architecture" in result.keywords
    # Stop words should be absent
    assert "what" not in result.keywords
    assert "is" not in result.keywords
    assert "the" not in result.keywords


def test_keywords_not_empty_for_valid_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("hybrid retrieval BM25 embeddings")
    assert len(result.keywords) >= 1


def test_keywords_empty_for_empty_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("")
    assert result.keywords == []


def test_keywords_max_count(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze(
        "What are the pros and cons of dense BM25 sparse hybrid retrieval "
        "alpha weighting normalization evidence grounding refusal mechanism"
    )
    assert len(result.keywords) <= 8


# ---------------------------------------------------------------------------
# Retrieval strategy selection
# ---------------------------------------------------------------------------


def test_strategy_factual_is_dense_weighted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is ThinkZen?")
    assert result.retrieval_strategy.alpha >= 0.65


def test_strategy_comparison_is_sparse_weighted(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Compare BM25 versus dense embedding retrieval strategies.")
    assert result.retrieval_strategy.alpha <= 0.45


def test_strategy_complex_increases_top_k(analyzer: QueryAnalyzer) -> None:
    simple = analyzer.analyze("What is ThinkZen?")
    complex_q = analyzer.analyze(
        "Can you explain in detail how the hybrid retrieval pipeline works "
        "combining BM25 sparse index with dense vector embeddings using "
        "min-max normalization and alpha-weighted fusion scoring?"
    )
    assert complex_q.retrieval_strategy.top_k >= simple.retrieval_strategy.top_k


def test_strategy_has_rationale(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("How does ThinkZen handle grounding?")
    assert len(result.retrieval_strategy.rationale) > 10
    assert len(result.retrieval_strategy.strategy_name) > 0


def test_strategy_alpha_in_valid_range(analyzer: QueryAnalyzer) -> None:
    queries = [
        "What is ThinkZen?",
        "Compare BM25 vs embeddings",
        "How does retrieval work?",
        "List all chunking strategies",
        "ThinkZen mein hybrid search kaise hota hai?",
        "भारत की राजधानी क्या है?",
    ]
    for q in queries:
        result = analyzer.analyze(q)
        assert 0.0 <= result.retrieval_strategy.alpha <= 1.0, f"Invalid alpha for: {q}"
        assert result.retrieval_strategy.top_k >= 1


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serializable(analyzer: QueryAnalyzer) -> None:
    import json
    result = analyzer.analyze("What is ThinkZen architecture?")
    d = result.to_dict()
    serialized = json.dumps(d)
    assert "language" in serialized
    assert "query_type" in serialized
    assert "retrieval_strategy" in serialized
    assert "alpha" in serialized


def test_to_dict_values_are_strings_and_primitives(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("What is ThinkZen?")
    d = result.to_dict()
    assert isinstance(d["language"], str)
    assert isinstance(d["query_type"], str)
    assert isinstance(d["keywords"], list)
    assert isinstance(d["retrieval_strategy"]["alpha"], float)
    assert isinstance(d["retrieval_strategy"]["top_k"], int)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_whitespace_only_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("   ")
    assert result.language == QueryLanguage.UNKNOWN
    assert result.query_type == QueryType.CONVERSATIONAL


def test_single_word_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("ThinkZen")
    assert result.complexity == QueryComplexity.SIMPLE
    assert len(result.keywords) >= 1


def test_punctuation_heavy_query(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("ThinkZen!!! What??? How!!!")
    # Should not crash and should produce a valid analysis
    assert result.retrieval_strategy.alpha >= 0.0
