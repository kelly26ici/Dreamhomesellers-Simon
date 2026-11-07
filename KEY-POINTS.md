# KEY-POINTS

The product spec. Everything here is true regardless of what the archived
documents said. Where an archived ADR disagrees with this file, this file
wins; where this file is silent, decide and record the decision here.

## What this service is for

Answer buyer questions about **published** Kenyan property inventory, in
natural language, grounded in current Postgres facts. It also hosts the media
processing worker. The website is its only caller.

## Invariants — these are not negotiable

1. **Published-only, structurally.** A buyer answer may only reference a
   revision whose listing is published. The current implementation does this
   with a `!inner` embed plus `lifecycle_state = 'published'` restated at
   every call site — a filter, not a structural guarantee. **The rebuild must
   make it structural**, because a filter can be forgotten at one call site.
2. **Contact privacy is designed out, not filtered.** No contact column is
   ever read, and `assert_no_contact` is a shape guard with UUID-blanking so
   ordinary payloads do not false-positive.
3. **Money is a decimal string with an explicit currency at every boundary.**
   Never a float.
4. **Similarity is not evidence.** A vector score ranks; it never proves a
   property is current, available, or allowed to be shown. Every claim
   hydrates from Postgres by id before it reaches a buyer.
5. **A reconnect never re-runs a billable turn.** SSE `Last-Event-ID` replay.
6. **No raw SQL, service-role key, unrestricted storage, direct moderation
   transition, or arbitrary vector query** may be reachable from
   model-controlled tool input.
7. **Every endpoint is authenticated.** A shared-secret header that only the
   Next.js relay sends. `/api/aux` is not an open LLM proxy.

## Endpoints that must keep working

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/chat` | one concierge turn, JSON envelope |
| `POST` | `/api/chat/stream` | same turn as SSE |
| `POST` | `/api/aux` | one bounded, tool-less model call |
| `GET` | `/api/health` | readiness: resolved provider, memory mode, tool count |
| `POST` | `/threads/{id}/commands` | Agent Streaming Protocol |
| `POST` | `/threads/{id}/stream` | Agent Streaming Protocol |

## Model fallbacks

Today: a hand-rolled provider chain walked by hand, with a real probe invoke
per provider under a 12s cap, a cooldown map, and module-global side channels.
Both call sites rebuild the model **per request**, so every request can burn
up to three paid invocations before answering a single question.

Target: `ModelFallbackMiddleware` + `ModelRetryMiddleware` from
`langchain.agents.middleware`, verified against the installed
`langchain 1.4.4`. One middleware list, declared once in `build_agent()`.

**Live production fact:** the service is currently answering with
`openai/gpt-oss-120b`, the **third** provider in the chain. The first two are
failing right now. This is exactly the failure the middleware migration
replaces with framework behaviour.

## Deployment

- Render free tier, `dreamhomesellers-simon.onrender.com`, auto-deploys on
  push to `main`.
- Keeps the **website** warm by pinging its `/api/health` every 5 minutes from
  inside the deployment (`src/simon/keepalive.py`). Disabled when
  `LUXONN_SITE_URL` is unset. **Verified live 2026-11-07:** both services
  stayed warm through 12 minutes with no traffic from any other source.
- `CORS_ALLOWED_ORIGINS` must be env-driven. The current hardcoded
  `localhost:3000` list is a live deployment blocker.
- **This machine cannot reach the database directly** — see the website's
  `KEY-POINTS.md`. Apply migrations from the Supabase dashboard.

## Tooling

- `uv`, `ruff` (to be added), `pytest` + `pytest-asyncio` + `respx` for
  offline httpx mocking. No test may spend a provider request.
- **36 offline tests is the baseline.** Any slice that reduces that number has
  deleted coverage, not simplified.
- Chainlit is **removed**. No `app.py`, no `chainlit.md`, no `.chainlit/**`,
  no `chainlit` dependency.
- `qdrant-client` becomes an optional extra, not a hard dependency, until the
  semantic-search slice lands.

## Owner decisions already made (do not re-litigate)

| # | Decision |
|---|---|
| 1 | Models fall back through LangChain middleware, not a hand-rolled chain. |
| 2 | Shared-secret header auth on every endpoint. |
| 3 | The viewing state machine lives in SQL, so no client can forge a transition. |
| 4 | Semantic search is visitor-facing *and* available to this service, over one projection. |
| 5 | Media processing (EXIF strip, derivatives, scan, caption) runs here as a background worker. Captions are generated once per image at publish. |
| 6 | No pre-publication review; listings go live when their owner publishes them. |
| 7 | The rebuild **replaces**, it does not repair. No patches, no fast fixes. |
