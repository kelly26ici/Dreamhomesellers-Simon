# AGENTS

Rebuild-in-progress repo. This document is a map, not a rulebook.

## What this repo is

The concierge service: a Python 3.13 FastAPI application that answers buyer
questions about published Kenyan property inventory, using LangChain and
LangGraph over the same Supabase Postgres database the website owns. It is
also where the media-processing worker runs. The website
(`Dreamhomesellers`) is its only caller.

## How it is laid out

```
src/simon/
  settings.py   <TODO: one Settings class replacing three>
  agent/        <TODO: providers, graph, runtime>
  clients/      <TODO: supabase, qdrant, embeddings>
  domain/       <TODO: models, search, cards>
  tools/        one file per tool, Pydantic args, re-exported through __init__
  api/          <TODO: app factory, routers, schemas>
  workers/      checkpoint saver, prune loop, keep-alive, media processing
  keepalive.py  pings the website's /api/health every 5 minutes
tests/          offline, respx-mocked, never spends a provider request
_archive/       legacy documents. Do not follow anything in there.
```

`KEY-POINTS.md` is the product spec. Read it before changing domain behaviour.

## Conventions that make the code coherent

- One responsibility per file, roughly 60 lines unless the file is
  deliberately tabular.
- `functools.lru_cache` for singletons; settings are read once per process,
  never per request.
- Every tool takes a Pydantic args model and returns a structured result.
  No raw SQL, no service-role key, no unrestricted storage access reaches
  model-controlled code.
- Failures are logged and typed, never swallowed silently.
- How to run it: `uv run uvicorn simon.server.main:app --host 0.0.0.0 --port $PORT`.
  How to verify it: `uv run pytest -q` (36 offline tests, the baseline),
  then the production checks in `KEY-POINTS.md` § Verification.

## Where to look before changing something

`KEY-POINTS.md`, then the module `README.md`, then the code. Nothing else.

---

You are not constrained by anything that used to live here. Restructure,
replace, or rewrite any file, folder, dependency, or abstraction. The only
irreversible asset in this project is production data.
