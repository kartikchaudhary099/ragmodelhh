"""Test that pipeline module interfaces are importable."""

import importlib


MODULES = [
    "modules.stt",
    "modules.chunking",
    "modules.embeddings",
    "modules.retrieval",
    "modules.reranking",
    "modules.generation",
    "modules.evaluation",
]


def test_pipeline_modules_importable() -> None:
    """All pipeline module packages should import without error."""
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
