# Official corpus — ai4bharat/MSMARCO-XI

This directory holds the **official dataset** subset that ThinkZen can ingest, kept
strictly separate from the demo corpus. It is normally **empty in git**: the actual
dataset content is built locally and is git-ignored (only this README and `.gitkeep`
are tracked).

## OFFICIAL vs DEMO — what's the difference?

| | OFFICIAL corpus | DEMO / SAMPLE corpus |
|---|---|---|
| Source | `ai4bharat/MSMARCO-XI`, file `train/hintrain.parquet` (Hindi split) | `data/samples/demo_docs.json` (20 hand-written docs) |
| Location | `data/official/msmarco_xi_sample.json` (+ `provenance.json`) | `data/samples/demo_docs.json` |
| Chunk metadata | `corpus="official"`, `is_official=True`, plus `dataset` / `source_file` / `source_revision` / `extraction_timestamp` | `corpus="demo"`, `is_official=False` |
| Enabled by | `THINKZEN_CORPUS=official` | default (no env var, or `THINKZEN_CORPUS=demo`) |
| Purpose | Satisfy the official HH Goa dataset requirement | Zero-setup demo / offline tests |

Both corpora are chunked with the project's **advanced** `SentenceChunker`
(strategy `sentence-boundary`) and indexed through the **same** hybrid dense+BM25
pipeline. Only the source and provenance metadata differ. The app **never** silently
serves demo data while claiming MSMARCO-XI: if `THINKZEN_CORPUS=official` is set but the
artifact below is missing or fails validation, the API returns HTTP 503 with the reason.

## How to build the subset (reproducible)

Requires network access and the `datasets` package (`pip install datasets`). Run from
the repo root using the project venv:

```bat
.venv\Scripts\python.exe scripts\build_msmarco_xi_sample.py --limit 100
```

Optional: pin a dataset revision for exact reproducibility:

```bat
.venv\Scripts\python.exe scripts\build_msmarco_xi_sample.py --limit 100 --revision <git-sha-or-tag>
```

This writes two files here:

* `msmarco_xi_sample.json` — up to 100 real records (raw MSMARCO-XI row shape), each
  validated against the strict provenance gate in `backend/modules/sample_ingestion.py`.
* `provenance.json` — `{dataset, source_file, source_revision, extraction_timestamp, sample_size, selected_query_ids}`.

## How to enable it

```bat
set THINKZEN_CORPUS=official
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_pipeline.py
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend
```

With `THINKZEN_CORPUS` unset (or `demo`), the app behaves exactly as before and serves
the demo corpus.

## Provenance & licensing

The dataset content is **not committed** to this repository (see `.gitignore`). Obtain it
directly from the official source and comply with its license:
<https://huggingface.co/datasets/ai4bharat/MSMARCO-XI>. The `provenance.json` file records
exactly which records were extracted and when, so any ingestion can be traced back to the
official source.
