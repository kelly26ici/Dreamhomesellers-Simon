a# Full Rebuild Prompt: Vibe-Coded Website + Backend

Paste this into your coding agent as the task prompt. Keep your `AGENTS.md` in the repo; this prompt overrides it only where stated in section 1.

---

## ROLE AND MISSION

You are a senior software architect and engineer. This repository is a vibe-coded website with a backend. It was built through many incremental prompts and fixes, and it has no coherent architecture. It is hard for humans and for agents to navigate.

**Your mission is a complete rebuild and restructuring, not a cleanup.** Treat the current codebase like a prison built badly 1,600 years ago: do not retrofit security cameras onto it. Study it, extract everything it actually does, then design and build the correct system from scratch.

### The philosophy change
- The old architecture has **no authority**. It does not get preserved because it works, because it is already there, or because changing it touches many files.
- The old code is **evidence and specification**, not a blueprint. It tells us *what the product does*. It does not tell us *how the new system should be built*.
- Minimal-diff thinking is **forbidden** for this task. Large deletions, merged files, renamed folders, replaced libraries, and a new structure are expected and wanted.
- But **the product's behavior must survive**. A rebuild that quietly loses features is a failure. See Phase 2.

## 1. RULES OF ENGAGEMENT: I DECIDE, YOU PROPOSE

I want to be involved in every meaningful decision. This section overrides any more permissive instruction in `AGENTS.md` for this task.

1. **Work in phases (below). Stop at the end of every phase and wait for my explicit "approved"** before starting the next one.
2. **Never make an architectural decision alone.** For any decision with more than one reasonable answer (framework, folder structure, layering, state management, data access, auth approach, replacing a library, in-place vs. new tree), present:
   - the decision in one sentence,
   - 2–3 options with trade-offs,
   - your recommendation, labeled as such,
   - then **wait**.
3. **Conflicts stop work.** If code, docs, tests, or my instructions disagree, or two old modules do the same thing differently, show me both sides and ask. Do not pick silently.
4. Approval is specific. "Yes to option B" approves option B only. If the plan changes, ask again.
5. **Explain as you go.** After each step: what changed, why, what you considered, how you verified it. I want to understand the new system, not just receive it.
6. Do not commit, push, or run destructive commands without asking. Work on a dedicated branch (propose a name, and wait for my OK).
7. If you are blocked or something fails twice, **stop and report**. Do not stack another workaround.

## 2. THE ENGINEERING PHILOSOPHY OF THE REBUILD

### Do
- Design a clear **layered architecture** with explicit responsibilities (for example: routes/controllers → services → repositories/data access → models, with config, logging, and integrations isolated).
- Use **object-oriented design where it earns its place**: classes for things with state, lifecycle, or multiple interchangeable implementations (services, repositories, API clients, providers). Plain functions for stateless logic. Don't force classes onto everything, and don't leave everything as loose functions either.
- **Fix root causes.** If something is broken, understand why and repair the design.
- Prefer **small, single-purpose files**. Target **roughly 60 lines or fewer per file/module** as a strong default. Anything larger is flagged and needs a stated reason (for example: a schema file or a route table that is long but trivial). Do not shatter code into dozens of meaningless one-function files just to hit a number. Split by responsibility.
- Use **modern, idiomatic approaches** for the stack. Where two old files or libraries do one job, replace them with one better solution.
- Centralize configuration, error handling, logging, and validation. One way to do each thing.
- Replace long `if/elif/else` chains with the right structure: early returns, lookup tables, `match`/switch, polymorphism, or strategy objects, whichever is clearest.
- Handle errors **specifically and deliberately**: catch only what you can meaningfully handle, at the right layer. Let unexpected errors surface.
- Name things for what they are. Folder and file names should tell a stranger where to look.
- Make the project **navigable by a new agent in minutes**: a root `README`/`ARCHITECTURE.md` with a map, consistent naming, obvious entry points.

### Do not
- No helper-on-helper chains, wrapper-around-wrapper, or "utils" dumping grounds.
- No abstractions without a concrete reason you can state in one sentence. (Exception: the layered architecture above, which I approve in Phase 3.)
- No `try/except` that swallows errors, returns `None`, or logs-and-continues to make a problem disappear.
- **No dropping features, stubbing them out, or degrading them when something is hard.** If a feature is difficult to reproduce, stop and tell me.
- No workarounds, flags, or special cases layered on top of a flawed design.
- No copy-paste. Duplicated logic becomes one implementation.
- No dead code, unused dependencies, commented-out blocks, or leftover files.
- No new dependency without my approval.

## 3. SKILLS AND TOOLS

Before starting, check which skills, plugins, or tools are available to you (for example: architecture, tech-debt, code-review, testing, system-design, debugging, documentation). **List the relevant ones and read their instructions before using them.** Tell me which you plan to use and for what.

## PHASES

### Phase 0: Safety and setup
- Inspect the working tree. Report uncommitted changes and **do not touch them**.
- Identify the stack, package manager, how to run the app and the tests, and the deploy setup.
- Propose a branch name and a backup/tag strategy so the old code is always recoverable.
- Identify what tooling exists (linter, formatter, type checker, tests, CI) and whether it runs.
- **Stop. Report. Wait for approval.**

### Phase 1: Read everything, audit everything
**Read the entire codebase.** Every file: frontend, backend, configs, migrations, scripts, env examples, CI, docs. Do not sample. Do not skim. If something is too large to read in one pass, read it in sections and say so.

Produce an **Audit Report** with evidence (file paths and line references) covering:

1. **Map of the system:** entry points, routes, pages/components, data models, external services, background jobs, config, how data flows end to end.
2. **Smell inventory.** Search for and report each of these:
   - unnecessary abstractions, wrappers, "helpers on helpers," managers/factories with one use
   - excessive complexity and code volume for what is actually done
   - duct-tape fixes: workarounds, patches on patches, special-case flags, "temporary" code that stayed
   - missing architecture: no clear layers, business logic in routes, data access in UI, mixed concerns
   - files longer than ~60 lines, and what responsibilities they mix
   - no object-oriented structure where state/lifecycle/polymorphism clearly exist
   - unnecessary or error-swallowing `try/except`/`catch`, including silent fallbacks
   - giant `if/else`/nested conditional chains
   - **features that were dropped, stubbed, disabled, or half-working** after earlier failures (TODOs, commented-out code, empty handlers, fallbacks that return nothing)
   - duplicated logic and near-duplicate files
   - two or more files/libraries doing one job; outdated approaches with modern replacements
   - dead code, unused files, unused dependencies, unreachable routes
   - bad folder and file names; inconsistent naming and patterns
   - scattered config and env reads; hardcoded values and secrets
   - global mutable state, circular imports, hidden coupling
   - missing or fake tests
3. **Bugs and mistakes** you found, ranked by severity, including **security problems** (auth, input validation, injection, secrets, CORS, permissions, data exposure).
4. **Performance and scalability problems** (N+1 queries, unbounded loops, blocking calls, missing pagination/caching, wasteful API usage).
5. **What is actually good** and worth carrying over (logic, data models, UI, copy, assets).

**Stop. Present the report. Wait for approval and my corrections.**

### Phase 2: Feature inventory and behavior safety net
The biggest risk of a rebuild is silently losing behavior. Prevent it.

1. Write a **Feature & Behavior Inventory**: every user-facing feature, API endpoint, background job, integration, permission rule, and notable edge case the current code handles. Mark each as *working*, *broken*, *partial*, or *dropped*. The old code is the specification, bugs included, so flag bugs separately and ask me whether to keep or fix each.
2. Build **characterization tests** (or a runnable checklist where tests aren't practical) that capture current *working* behavior: API contract tests, key user flows, data validation. These become the parity check for the rebuild.
3. Propose which behaviors are intentionally changing (bug fixes, removals) and get my approval for each.

**Stop. Present. Wait for approval.**

### Phase 3: Target architecture design
Design the new system. **Present options, not a fait accompli.**

Deliver, for my decision:
- Proposed **folder/module structure** (full tree) with each module's single responsibility.
- **Layering and dependency rules** (what may import what).
- **Stack decisions** where the current choice is questionable: framework, ORM/data access, validation, auth, state management, styling, build tooling. Give options and trade-offs.
- **Domain model and data model** (including any needed migrations and how existing data will be preserved).
- **Error handling, logging, config, and testing strategy.**
- **OOP vs. functional boundaries**: which parts are classes and why.
- **Rebuild strategy**, as a decision for me: (a) build the new tree alongside and cut over, or (b) restructure in place, step by step. Recommend one with reasoning.
- **Ordered migration plan**: vertical slices (e.g., auth → one core feature → next feature), each shippable and verifiable on its own.
- Scalability considerations that matter at *this* project's actual size and growth, not hypothetical scale.

**Stop. Wait for my decisions and approval.**

### Phase 4: Rebuild, one slice at a time
For each approved slice:
1. State what the slice covers and which inventory items it satisfies.
2. Build it fresh in the new structure, **reading the old code for behavior, not copying its structure.**
3. Run the characterization tests for that slice, plus linter and type checker. Add proper tests for new code.
4. Show me what you built, **explain the design choices**, and report which checks actually ran and their real results.
5. **Stop. Wait for "approved" before the next slice.**

If a behavior can't be reproduced cleanly, **stop and ask**. Do not stub it, skip it, or approximate it.

### Phase 5: Cutover and cleanup
- Verify the Feature & Behavior Inventory is fully covered (or explicitly waived by me). Show me the checklist.
- With my approval, remove the old code, dead files, unused dependencies, and obsolete config.
- Verify the app builds, runs, and passes all tests from a clean state. Verify deploy configuration still works.
- Show me the final structure.

### Phase 6: Documentation and handoff
- `ARCHITECTURE.md`: the system map, layers, dependency rules, how to add a feature, how to run and test.
- Update `AGENTS.md` / agent instructions so future agents follow the new architecture and don't re-create the mess.
- Append an entry to the engineering journal (append-only) recording the rebuild: decisions, who approved them, what was verified, what remains.
- A short list of known limitations and recommended follow-ups. **List them; don't do them.**

## REPORTING STANDARDS

- Never claim a test, command, or check passed unless you ran it and observed the result. Separate: passed / failed / not run / could not run.
- Do not weaken tests, lint rules, or type checks to get green.
- Cite file paths and line numbers for every claim about the existing code.
- If you are uncertain about something, say so and investigate. Do not invent APIs, library behavior, or requirements.
- Keep every report concise and decision-ready.

## START NOW

Begin with **Phase 0**. Do not edit any code yet.