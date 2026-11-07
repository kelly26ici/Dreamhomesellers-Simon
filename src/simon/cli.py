import asyncio
import sys
import uuid

from simon.agent.graph import build_agent
from simon.agent.runtime import converse


def main():
    agent = asyncio.run(build_agent())
    thread_id = str(uuid.uuid4())
    print("Simon, Dreamhomesellers property concierge. Ask me anything.")
    while True:
        try:
            message = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message.strip():
            continue
        if message.strip().lower() in {"quit", "exit"}:
            return
        try:
            asyncio.run(
                converse(
                    agent,
                    message,
                    thread_id=thread_id,
                    on_token=lambda t: sys.stdout.write(t) or sys.stdout.flush(),
                    on_tool_call=lambda call: print(
                        f"\n[{call.tool_name}]", file=sys.stderr
                    ),
                )
            )
        except Exception as exc:
            print(f"\nSomething broke: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
