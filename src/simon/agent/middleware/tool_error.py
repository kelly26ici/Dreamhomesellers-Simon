"""Tool errors, redacted.

This closes a real finding, not a hypothetical one. The audit recorded that
internal error text -- including provider and DSN detail -- reached clients in
SSE frames (server/main.py, server/protocol.py). A tool that raises with a
Postgres error string hands the model that string, and the model hands it to
the buyer.

The rule: the DETAIL is logged server-side, the buyer gets a category. A buyer
cannot act on "connection refused at db.<ref>..." and an attacker can.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import ToolErrorMiddleware

LOGGER = logging.getLogger("simon.tools")


def _tool_name(request) -> str:
    try:
        return request.tool_call["name"]
    except Exception:  # noqa: BLE001 - the message must be produced regardless
        return "unknown"


def on_tool_error(error: BaseException, request) -> str:
    """Log the detail, return only the category."""
    name = _tool_name(request)
    LOGGER.warning("tool %s failed: %s: %s", name, type(error).__name__, error)
    return (
        f"The {name} tool could not complete. "
        "Say so plainly and offer to record a lead for the sales team."
    )


def build() -> ToolErrorMiddleware:
    """The middleware needs an explicit handler; there is no default.

    Raising on construction without one is the framework saying "decide what
    the buyer is told", so this is where that decision lives rather than being
    spread across the call sites.
    """
    return ToolErrorMiddleware(on_error=on_tool_error)
