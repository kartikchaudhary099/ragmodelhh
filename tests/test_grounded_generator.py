"""Unit tests for GroundedGenerator module."""

import pytest
from modules.generation import GeneratedAnswer
from modules.generation.grounded_generator import GroundedGenerator
from modules.retrieval import RetrievedDocument


@pytest.mark.asyncio
async def test_grounded_generator_empty_query():
    generator = GroundedGenerator()
    answer = await generator.generate("", [])
    assert answer.refused is True
    assert "valid query" in answer.text.lower()


@pytest.mark.asyncio
async def test_grounded_generator_no_context_refusal():
    generator = GroundedGenerator()
    answer = await generator.generate("What is ThinkZen?", [])
    assert answer.refused is True
    assert answer.sources == []
    assert answer.refusal_reason is not None


@pytest.mark.asyncio
async def test_grounded_generator_low_score_refusal():
    generator = GroundedGenerator(confidence_threshold=0.5)
    weak_doc = RetrievedDocument(
        chunk_id="chunk_low",
        text="Random irrelevant text snippet.",
        score=0.1,
        method="hybrid",
    )
    answer = await generator.generate("What is quantum computing?", [weak_doc])
    assert answer.refused is True
    assert answer.sources == []


@pytest.mark.asyncio
async def test_grounded_generator_hindi_query_detection():
    generator = GroundedGenerator()
    assert generator.is_hindi_query("वाणीरैग कैसे काम करता है?") is True
    assert generator.is_hindi_query("What is ThinkZen?") is False


@pytest.mark.asyncio
async def test_grounded_generator_grounded_answer_success():
    generator = GroundedGenerator(confidence_threshold=0.1)
    doc = RetrievedDocument(
        chunk_id="chunk_01",
        text="ThinkZen utilizes hybrid dense and sparse retrieval.",
        score=0.8,
        method="hybrid",
        metadata={"title": "ThinkZen Spec"},
    )
    answer = await generator.generate("How does ThinkZen retrieve docs?", [doc])
    assert answer.refused is False
    assert len(answer.sources) == 1
    assert answer.sources[0].chunk_id == "chunk_01"
    assert "ThinkZen" in answer.text
