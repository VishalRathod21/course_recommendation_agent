"""Sanity checks that need no API key: DB, tools, graph compilation, ranker/critic logic."""
from __future__ import annotations

import json
import os

os.environ.setdefault("GROQ_API_KEY", "offline-placeholder")

from agent import db  # noqa: E402
from agent.graph import build_graph, critic, ranker  # noqa: E402
from agent.tools import run_tool  # noqa: E402

print("1) seeding DB ...", db.seed_if_needed(), "courses")

print("2) search_course_db:")
print("  ", run_tool("search_course_db", {"skill": "machine learning", "level": "any", "max_price_inr": 0, "max_hours": -1}, [], "test")[:200])

print("3) python_sandbox:")
print("  ", run_tool("python_sandbox", {"code": "print(sum([0, 499, 0]))"}, [], "test"))
print("  ", run_tool("python_sandbox", {"code": "import os"}, [], "test"))

print("4) check_prerequisites:")
print("  ", run_tool("check_prerequisites", {"course_id": 14, "known_skills": "python"}, [], "test"))

print("5) web search failure -> fallback:")
os.environ["SIMULATE_WEB_FAILURE"] = "1"
trace: list = []
out = run_tool("web_course_search", {"query": "sql"}, trace, "test")
print("  ", [e["type"] for e in trace])
assert any(e["type"] == "reflection" for e in trace), "fallback reflection missing"
os.environ["SIMULATE_WEB_FAILURE"] = "0"

print("6) ranker + critic deterministic path:")
profile = {"goal": "data scientist", "current_skills": ["python"], "level": "beginner", "time_weeks": 24, "hours_per_week": 10, "budget_inr": 0, "preferred_format": "any"}
plan = [{"order": 1, "skill": "sql", "why": ""}, {"order": 2, "skill": "statistics", "why": ""}, {"order": 3, "skill": "machine learning", "why": ""}]
cands = {p["skill"]: json.loads(run_tool("search_course_db", {"skill": p["skill"], "level": "any", "max_price_inr": 0, "max_hours": -1}, [], "test"))["results"] for p in plan}
state = {"profile": profile, "plan": plan, "candidates": cands, "critique": {}, "retries": 0}
r = ranker(state)
for c in r["path"]:
    print(f"   step {c['step']} {c['for_skill']:<18} -> {c['title']} ({c['hours']}h, {c['price_inr']} INR)")
print("   totals:", r["totals"])
state.update(r)
state["retries"] = 2  # skip LLM critic
cr = critic(state)
print("   critic decision:", cr["critique"]["decision"], cr["critique"]["issues"])

print("7) graph compiles:")
g = build_graph()
print("  ", list(g.get_graph().nodes.keys()))
print("\nALL OFFLINE CHECKS PASSED")
