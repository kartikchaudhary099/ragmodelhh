# Phase 3B: Tiny real-dataset ingestion and chunking experiment

## Scope

This experiment uses the authoritative competition dataset `ai4bharat/MSMARCO-XI` and keeps the sample tiny.

- Authoritative dataset: `ai4bharat/MSMARCO-XI`
- Hindi config target: `"hi"`
- Streaming access: `True`
- Sample size target: 100 real records
- Full dataset download: not allowed
- Full RAG pipeline: not implemented
- Embeddings / Qdrant: not implemented
- Frontend / STT / generation: not implemented

## Dataset configuration

Verified official usage pattern from the dataset card:

```python
from datasets import load_dataset

dataset = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
```

Runtime verification in this environment showed that the `"hi"` builder config is not exposed here, so the implementation keeps the intended `"hi"` configuration but falls back to `load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)` with a `[VERIFY]` marker instead of guessing. This is deliberate and not treated as interchangeable with `ai4bharat/IndicMSMARCO`.

## Verified runtime schema

The real runtime schema for `ai4bharat/MSMARCO-XI` was observed as:

```python
['source_lang', 'target_lang', 'meta', 'Answer', 'query_id', 'query_type', 'passages', 'Eng_Query', 'Eng_Answer', 'query']
```

This is the authoritative field set used for extraction in this project.

## Fields extracted

The sample pipeline normalizes the actual MSMARCO-XI fields into a project-friendly record while preserving both English and translated content where useful:

- query_id
- query
- Answer
- query_type
- passages
- source_lang
- target_lang
- Eng_Query
- Eng_Answer
- metadata

The extraction logic preserves both the translated user-facing content and the English source context so future multilingual retrieval components can choose the representation they need.

## Chunking strategies

This experiment implements four independent chunking directions:

1. Fixed-size baseline
   - configurable chunk size
   - configurable overlap
2. Sentence/semantic boundary chunking
   - natural sentence boundaries preferred
   - configurable target size
3. Parent-child chunking
   - larger parent context
   - smaller child retrieval units
   - parent_id / child_id preserved in metadata
4. Metadata-aware representation
   - provenance metadata retained at every chunk
   - all chunk objects keep language, query type, source/target language, and source metadata

## Experiment assumptions

- Real dataset samples are used, not synthetic records.
- The loader uses streaming access to avoid a full dataset download.
- Chunk sizes are kept as experimental defaults and are not claimed to be optimal.
- The experiment only validates data extraction and chunk generation behavior.
- No embeddings, Qdrant, retrieval, STT, generation, or frontend work is implemented here.

## Known limitations

- The `"hi"` config is documented in the dataset card but was not exposed at runtime in this environment; the loader falls back to the default config and is marked `[VERIFY]`.
- The tiny sample is not a benchmark and is not intended to represent the full corpus.
- This is not a full multilingual retrieval benchmark.
- Streaming support may vary by dataset package version, so the loader is intentionally wrapped for safe verification.
