"""Vercel serverless entrypoint for ThinkZen.

Vercel's ``@vercel/python`` runtime imports this module and serves the ASGI
application it finds exposed as the module-level ``app`` variable. This mirrors
the production run command used elsewhere::

    uvicorn app.main:app --app-dir backend

The FastAPI application lives in the ``app`` package under ``backend/``, so we
put that directory on ``sys.path`` before importing it — exactly what
``--app-dir backend`` does for uvicorn. No application, routing, UI, or backend
logic is defined or modified here; this file only wires the existing app up to
Vercel's runtime.

The repository layout (``backend/`` ``frontend/`` ``data/`` as siblings of the
repo root) is preserved in the deployment bundle via ``includeFiles`` in
``vercel.json``. That matters because ``app.main`` and ``app.config`` resolve
their static-file and data directories with ``Path(__file__).resolve().parents[2]``
(the repo root), so ``/frontend/static`` and ``/data/samples`` must sit where
they do in the source tree.
"""

import sys
from pathlib import Path

# backend/ is a sibling of this api/ directory in the deployed bundle.
_BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Import the already-built FastAPI app: StaticFiles SPA mounted at "/",
# plus /api/v1/query, /api/v1/judge, /api/v1/stt and /health from the routers.
from app.main import app  # noqa: E402  (import after sys.path setup, by design)

# `app` is the ASGI callable Vercel serves. `handler` is an alias kept as a
# safety net for builder versions that look for that name instead.
handler = app

__all__ = ["app", "handler"]
