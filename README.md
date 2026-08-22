# 🎧 ThinkZen — Voice-Native Multilingual RAG

> A premium, voice-first Retrieval-Augmented Generation (RAG) system with a tactile interface, interactive speaking avatar, custom sound synthesis, and real-time multilingual evidence grounding.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/HH_Goa_AI-2026-FFA500?style=for-the-badge)](https://github.com/Vikaspal505/ThinkZen)

---

## ✨ What's New: Premium UI/UX & Interactive Capabilities

ThinkZen has been completely overhauled with a state-of-the-art interactive layer:

*   **🎨 Tactile Skin-Tone Theme Mode**: The user interface features a sleek warm wood palette with custom CSS variables (`--bg: #0F0A08` deep umber, `--surface: #1C1410` dark chocolate, `--accent: #D4956A` warm copper). Includes full local-storage persistence for switching between warm-dark and warm-light (`#F5EBE0`) modes.
*   **🤖 Interactive SVG Speaking Avatar**: The central voice orb now houses a vector face that blinks automatically, squints/smiles dynamically based on the system state (Listening, Processing, Answering, Error), and syncs its lip/mouth movements to synthesized text-to-speech feedback.
*   **👁️ Pupil Tracking**: The avatar's eyes follow your mouse cursor coordinates across the screen in real-time.
*   **🎵 Web Audio Synth (Asset-Free Sound Effects)**: Dynamic audio feedback generated via client-side oscillator synthesis:
    *   **Clicks**: Subtle tactile micro-clicks on buttons, toggles, and rating inputs.
    *   **Listening Sweep**: A warm rising sweep when microphone input begins.
    *   **Success Chime**: A pleasant double-tone major chord on query completion.
    *   **Error Buzz**: A low-frequency warning alert if an error/refusal occurs.
*   **📝 Markdown RAG Report Exporter**: Download a beautifully structured local `.md` file containing the query, generated response, and full evidence source list details (score, method, and rank).
*   **🛠️ Offline Diagnostics Mode**: Type `"test_export"` in the composer box to test the entire client-side RAG pipeline, success chimes, and markdown exporter offline without a running backend.

---

## 📁 Project Structure

```
ThinkZen/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app setup & static file server
│   │   ├── config.py            # App settings (via env variables)
│   │   ├── exceptions.py        # Custom RAG pipeline error mapping
│   │   ├── logging_config.py    # Structured server logging
│   │   └── api/routes/
│   │       ├── health.py        # /health endpoint
│   │       ├── query.py         # POST /api/v1/query — hybrid RAG pipeline
│   │       └── stt.py           # POST /api/v1/stt — server STT wrapper
│   └── modules/
│       ├── query_analyzer.py    # Language (EN, HI, Hinglish) & query intent classifier
│       ├── evidence.py          # Evidence Intelligence (dedup, diversity, grounding)
│       ├── sparse_retrieval.py  # BM25 sparse indexer
│       ├── data_pipeline.py     # Document loaders and chunking
│       ├── official_corpus.py   # MSMARCO-XI data importer
│       ├── sample_seeder.py     # Demo content loader
│       ├── telemetry.py         # Latency aggregator & percentile tracker
│       ├── embeddings/          # HashingEmbeddingProvider (dependency-free semantic search)
│       ├── generation/          # GroundedGenerator (llm/fallback generator)
│       ├── retrieval/           # OrchestratedHybridRetriever
│       └── reranking/           # FlashRank cross-encoder reranker
├── frontend/
│   └── static/
-------------
│       ├── index.html           # Single-page interface with SVG avatar structure
│       ├── style.css            # Skin-tone theme stylesheets, keyframes, transitions
│       └── app.js               # Sound synthesis, voice controller, eye tracking, exporter
├── data/
│   ├── samples/                 # Local demo docs corpus (demo_docs.json)
│   └── official/                # MSMARCO-XI official dataset files
├── tests/                       # Complete Pytest test suite
├── .env.example                 # Environment configuration template
├── requirements.txt             # Core python packages
└── Dockerfile                   # Docker build definition
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- **Python 3.10+** (Tested on Python 3.14)
- Web browser with microphone access (Chromium-based browser like Chrome or Edge recommended for browser-native STT)

### 1. Clone & Enter Project
```bash
git clone https://github.com/Vikaspal505/ThinkZen.git
cd ThinkZen
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Environment File
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

### 4. Run Development Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access ThinkZen
Open your browser and navigate to:
```
http://localhost:8000
```
*Note: The FastAPI backend serves the frontend assets automatically from `/`.*

---

## 🌐 REST API Specifications

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Fetch application build version and health status |
| `POST` | `/api/v1/query` | Submit query to the multi-stage hybrid RAG pipeline |
| `POST` | `/api/v1/stt` | Server-side Speech-to-Text conversion (requires `SARVAM_API_KEY`) |

### Example Pipeline Request
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Retrieval-Augmented Generation?", "use_adaptive_retrieval": true}'
```

---

## 🔬 Multi-Stage Grounded RAG Architecture

```
                 User Request (Voice / Text)
                              │
                              ▼
                   ┌───────────────────────┐
                   │  1. Query Analyzer    │  ← Intent & auto language classification
                   └──────────┬────────────┘
                              │ Optimal alpha, top_k
                              ▼
                   ┌───────────────────────┐
                   │  2. Hybrid Retrieval  │  ← Dense (hashing) + Sparse (BM25)
                   └──────────┬────────────┘
                              │ Ranked candidate list
                              ▼
                   ┌───────────────────────┐
                   │  3. Reranker          │  ← FlashRank cross-encoder sorting
                   └──────────┬────────────┘
                              │ Reranked evidence items
                              ▼
                   ┌───────────────────────┐
                   │  4. Evidence Intel    │  ← Coherence check, diversity, grounding
                   └──────────┬────────────┘
                              │ Grounded context
                              ▼
                   ┌───────────────────────┐
                   │  5. Generator         │  ← Deterministic quoting or LLM prompt
                   └──────────┬────────────┘
                              │
                              ▼
                     Grounded Output Response
```

*   **Adaptive Alpha Fusion**: The Query Analyzer matches keywords and intent complexity to set retrieval weighting on a scale of `0.0` (sparse-only) to `1.0` (dense-only). Factual intents automatically lean dense (`0.7`), while keyword-centric queries favor BM25 (`0.3`).
*   **Honest Refusal Safeguards**: If the Evidence Intelligence module detects zero high-confidence matches, the system triggers a clean refusal instead of generating plausible hallucinations.

---

## 📊 Evaluation & Testing

Run the full testing suite locally using `pytest`:

```bash
# Install testing dependencies
pip install -r backend/requirements-dev.txt

# Run all unit and integration tests
pytest
```

---

## 🐳 Docker Deployment

To launch ThinkZen inside a self-contained container:

```bash
# Build the container image
docker build -t thinkzen .

# Start the container serving on port 8000
docker run -d -p 8000:8000 --env-file .env thinkzen
```

---

## 📝 License
Built for the **HH Goa 2026 AI Challenge — Task 2** (Voice-native multilingual RAG).
