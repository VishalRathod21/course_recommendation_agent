"""Terminal runner (backup for the demo if Streamlit misbehaves).

    python cli.py
    python cli.py "I know Python, want to be a data scientist in 6 months, budget 0"
"""
from __future__ import annotations

import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from agent import llm as llm_mod  # noqa: E402
from agent.graph import build_graph, run_turn  # noqa: E402

ICONS = {"node": "[NODE]", "llm": "[LLM ]", "thought": "[THNK]", "tool_call": "[TOOL]", "tool_result": "[RSLT]", "reflection": "[REFL]", "error": "[ERR ]", "state": "[STAT]"}


def main():
    err = llm_mod.check_credentials()
    if err:
        print("Config error:", err)
        sys.exit(1)
    print("LLM:", llm_mod.provider_label())
    graph = build_graph()
    thread = str(uuid.uuid4())
    first = " ".join(sys.argv[1:]).strip()
    while True:
        text = first or input("\nYou > ").strip()
        first = ""
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        print("\n----- TRACE -----")
        final = None
        for node, events, answer in run_turn(graph, text, thread):
            for e in events:
                print(f"{ICONS.get(e['type'], '[    ]')} {e['node']:<9} {e['title']}")
                if e["type"] in ("tool_call", "reflection", "error", "state") and e.get("detail"):
                    detail = e["detail"].replace("\n", "\n            ")
                    print(f"            {detail[:700]}")
            if answer:
                final = answer
        print("----- ANSWER -----\n")
        print(final)


if __name__ == "__main__":
    main()
