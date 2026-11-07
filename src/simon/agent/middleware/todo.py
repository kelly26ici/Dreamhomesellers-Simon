"""To-do list: let the model plan when a turn is genuinely multi-step.

Costs a little prompt space and a little token spend on turns where the model
chooses to use it. Included because it is the framework's structure for a plan
rather than a bespoke planner -- but the boundaries are stated honestly: its
own default prompt tells the model to skip it for anything under three steps,
and a concierge answering "3-bedrooms in Kilimani" is one step. So in practice
this stays dormant most of the time and earns its place on the rare compound
request ("compare these two, then tell me which is cheaper per square metre").

The `write_todos` prompts are left at the framework default: customising them
would be tuning, not structure.
"""

from __future__ import annotations

from langchain.agents.middleware import TodoListMiddleware


def build() -> TodoListMiddleware:
    return TodoListMiddleware()
