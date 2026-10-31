"""Tools the agent can call. Each is a LangChain @tool so the LLM can select it dynamically.

Tools:
  search_course_db     - SQLite catalog search (structured, has price / hours / level)
  web_course_search    - live DuckDuckGo search, no API key needed
  check_prerequisites  - does the learner satisfy a course's prerequisites?
  python_sandbox       - run small Python snippets (cost / hours arithmetic)
  save_profile / load_profile - long-term memory on disk
"""
from __future__ import annotations

import io
import json
import os
import contextlib
from pathlib import Path

from langchain_core.tools import tool

from . import db

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


def _compact(c: dict) -> dict:
    """Trim a course record so tool output stays small (free-tier token limits)."""
    return {
        "id": c["id"],
        "title": c["title"],
        "provider": c["provider"],
        "level": c["level"],
        "skills": c["skills"],
        "prerequisites": c["prerequisites"],
        "hours": c["duration_hours"],
        "price_inr": c["price_inr"],
        "rating": c["rating"],
        "format": c["format"],
    }


@tool
def search_course_db(skill: str, level: str = "any", max_price_inr: int = -1, max_hours: int = -1) -> str:
    """Search the local course catalog (SQLite) for courses teaching a skill.
    skill: the skill or topic to search, e.g. "sql", "machine learning", "react".
    level: "beginner", "intermediate", "advanced" or "any".
    max_price_inr: maximum price in INR, -1 for no limit. 0 returns only free courses.
    max_hours: maximum course duration in hours, -1 for no limit.
    Returns a JSON list of matching courses with id, title, provider, level, hours, price, rating."""
    rows = db.search_courses(
        skill,
        level=level,
        max_price_inr=None if max_price_inr < 0 else max_price_inr,
        max_hours=None if max_hours <= 0 else max_hours,
    )
    return json.dumps({"query": skill, "count": len(rows), "results": [_compact(r) for r in rows]})


@tool
def web_course_search(query: str, max_results: int = 5) -> str:
    """Search the live web (DuckDuckGo, free) for online courses. Use when the local catalog
    has too few results for a skill, or to find very new / niche courses.
    Returns a JSON list of {title, url, snippet}."""
    if os.getenv("SIMULATE_WEB_FAILURE", "0") == "1":
        raise ConnectionError("Simulated outage: web search API returned HTTP 503")
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(f"{query} online course", max_results=max_results):
            results.append(
                {
                    "title": r.get("title", "")[:120],
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:200],
                }
            )
    if not results:
        raise ValueError("Web search returned no results")
    return json.dumps({"query": query, "count": len(results), "results": results})


@tool
def check_prerequisites(course_id: int, known_skills: str) -> str:
    """Check whether a learner with the given comma-separated known_skills satisfies the
    prerequisites of the course with this id. Returns JSON {ok, missing, prerequisites}."""
    c = db.get_course(int(course_id))
    if not c:
        return json.dumps({"error": f"course id {course_id} not found"})
    known = [k.strip().lower() for k in known_skills.split(",") if k.strip()]
    missing = [p for p in c["prerequisites"] if not any(p in k or k in p for k in known)]
    return json.dumps({"course": c["title"], "prerequisites": c["prerequisites"], "missing": missing, "ok": not missing})


_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ["abs", "min", "max", "sum", "len", "range", "round", "sorted", "print", "int", "float", "str", "list", "dict", "enumerate", "zip", "map", "filter", "any", "all"]
}


@tool
def python_sandbox(code: str) -> str:
    """Execute a short, pure-Python snippet (arithmetic, loops, sorting) and return its printed
    output. No imports, no file or network access. Use it to compute totals such as cost,
    total hours or required hours per week."""
    if "import" in code or "__" in code or "open(" in code:
        return "ERROR: imports, dunder access and file IO are not allowed in the sandbox."
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": _SAFE_BUILTINS}, {})
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    out = buf.getvalue().strip()
    return out if out else "(no output - remember to print())"


@tool
def save_profile(name: str, profile_json: str) -> str:
    """Persist a learner profile (JSON string) under a name so it can be recalled in a later session."""
    path = MEMORY_DIR / f"{name.strip().lower().replace(' ', '_')}.json"
    path.write_text(profile_json, encoding="utf-8")
    return f"saved profile to {path.name}"


@tool
def load_profile(name: str) -> str:
    """Load a previously saved learner profile by name. Returns the JSON or an error."""
    path = MEMORY_DIR / f"{name.strip().lower().replace(' ', '_')}.json"
    if not path.exists():
        return json.dumps({"error": f"no saved profile named '{name}'"})
    return path.read_text(encoding="utf-8")


ALL_TOOLS = [search_course_db, web_course_search, check_prerequisites, python_sandbox, save_profile, load_profile]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def run_tool(name: str, args: dict, trace: list, node: str) -> str:
    """Execute a tool with error handling + trace logging.

    Self-correction: if web search fails, automatically fall back to the local DB and
    record the reflection in the trace.
    """
    trace.append({"node": node, "type": "tool_call", "title": f"Tool call -> {name}", "detail": json.dumps(args)})
    tool_obj = TOOLS_BY_NAME.get(name)
    if tool_obj is None:
        msg = f"ERROR: unknown tool '{name}'. Available: {list(TOOLS_BY_NAME)}"
        trace.append({"node": node, "type": "error", "title": "Unknown tool", "detail": msg})
        return msg
    try:
        out = tool_obj.invoke(args)
        trace.append({"node": node, "type": "tool_result", "title": f"Tool result <- {name}", "detail": str(out)[:1500]})
        return str(out)
    except Exception as e:
        err = f"ERROR calling {name}: {type(e).__name__}: {e}"
        trace.append({"node": node, "type": "error", "title": f"Tool error in {name}", "detail": err})
        if name == "web_course_search":
            trace.append(
                {
                    "node": node,
                    "type": "reflection",
                    "title": "Reflection: web search unavailable -> falling back to local catalog",
                    "detail": "Retrying the same query against search_course_db.",
                }
            )
            fb_args = {"skill": args.get("query", ""), "level": "any", "max_price_inr": -1, "max_hours": -1}
            trace.append({"node": node, "type": "tool_call", "title": "Tool call -> search_course_db (fallback)", "detail": json.dumps(fb_args)})
            out = search_course_db.invoke(fb_args)
            trace.append({"node": node, "type": "tool_result", "title": "Tool result <- search_course_db (fallback)", "detail": str(out)[:1500]})
            return f"{err}\nFALLBACK local catalog result: {out}"
        return err
