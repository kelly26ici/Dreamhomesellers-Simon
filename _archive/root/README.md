# Simon

The Dreamhomesellers property concierge — a LangChain v1 agent.

Sellers post listings; buyers search and filter the same pool. Simon lives on the
website (Chainlit UI) and helps with anything a visitor needs: finding
properties, comparing them, booking viewings, and posting new listings.

## Stack

- **Python 3.14 + uv** — managed with `uv`, settings via `pydantic-settings`
- **LangChain v1** — `create_agent` + `init_chat_model`, fully async via
  `astream_events` (v3), traced with LangSmith
- **Models** — NVIDIA Nemotron (primary), Gemini, Groq, Poolside
- **Database** — Supabase PostgreSQL (shared property pool, schema inherited
  as-is from the previous project) + Qdrant for semantic search
- **UI** — Chainlit, themed to the LUXONN dark/gold site

## Layout

```
src/simon/
├── settings.py      # pydantic-settings BaseSettings (.env driven)
├── cli.py           # entry point
├── agent/           # agent assembly: model, tools, middleware
├── catalog/          # property-pool domain layer (listings, search, viewings)
├── datastores/      # async Postgres + Qdrant access
├── tools/           # agent-facing tool functions
├── ui/              # Chainlit app wiring
└── integration/     # external provider clients (Tavily etc.)
```

## Quick start

```bash
uv sync
cp .env.example .env   # fill in credentials
uv run chainlit run src/simon/ui/app.py
```

## Status

Fresh rebuild of the abandoned `~/Projects/Production/Simon/` — only its
database (schema + data) is carried over. No code, names, or structure from the
old project or from `Simon-Langchain/` (the earlier migration attempt) is reused.
