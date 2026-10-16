"""The concierge's tool surface: seven tools, one module each.

This package is the public surface and nothing else — it re-exports the tool
callables and the one ``TOOLS`` list the agent binds. There are no tool
implementations here; each tool owns a file with its Pydantic args model and a
thin handler, and the cross-cutting machinery (result caps, the contact
redaction rule, structured errors, the ``@tool`` adapter) lives in
``simon.tools._base`` so it is identical for every tool.

Tool set, after owner decision D3 removed ``create_listing``:

    search_properties, describe_property, compare_properties,
    search_by_meaning, schedule_viewing, record_lead

``create_listing`` is gone, not disabled. An LLM must not insert an
unattributed, unmoderated, unpublished listing row: the write path is the
owner domain commands, which require an authenticated org membership, and the
lifecycle/publisher columns are trigger-guarded — so there is no legal
anonymous write path to ``listings`` for an agent to be given.
"""

from __future__ import annotations

from typing import Any

from simon.tools.compare_properties import compare_properties
from simon.tools.describe_property import describe_property
from simon.tools.record_lead import record_lead
from simon.tools.schedule_viewing import schedule_viewing
from simon.tools.search_by_meaning import search_by_meaning
from simon.tools.search_properties import search_properties

# Count and order are part of the agent's capability surface: /api/health
# reports len(TOOLS), and the order the model sees the tools in comes from here.
TOOLS: list[Any] = [
    search_properties,
    describe_property,
    compare_properties,
    search_by_meaning,
    schedule_viewing,
    record_lead,
]

__all__ = [
    "TOOLS",
    "search_properties",
    "describe_property",
    "compare_properties",
    "search_by_meaning",
    "schedule_viewing",
    "record_lead",
]
