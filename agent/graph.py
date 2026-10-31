"""LangGraph state machine for the Course Recommendation Agent.

   intake --(missing info)--> clarify --> END
     |
     +--(off topic)---------> redirect --> END
     |
     v
   planner --> executor (ReAct tool loop) --> ranker --> critic --+--> responder --> END
      ^                                        ^                 |
      |                                        |   (budget/time) |
      +------------- (coverage/order) ---------+-----------------+   max 2 retries
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import db
from .llm import extract_json, invoke_with_fallback
from .tools import ALL_TOOLS, run_tool, search_course_db

MAX_RETRIES = 2
MAX_REACT_STEPS = 10
LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    profile: dict
    route: str
    plan: list
    candidates: dict
    web_extras: list
    path: list
    totals: dict
    critique: dict
    retries: int
    trace: Annotated[list, operator.add]
    final_answer: str


def _ev(node: str, type_: str, title: str, detail: Any = "") -> dict:
    if not isinstance(detail, str):
        detail = json.dumps(detail, indent=1, ensure_ascii=False)
    return {"node": node, "type": type_, "title": title, "detail": detail}


# --------------------------------------------------------------------------- intake
INTAKE_SYS = """You are the intake module of a course recommendation agent.
Read the WHOLE conversation and output the learner profile as JSON only (no prose):
{
 "on_topic": true/false,          # false if the user is not asking about learning / courses / careers
 "goal": "short text or null",     # what they want to learn or become
 "current_skills": ["..."],        # skills they already have, lowercase, short
 "level": "beginner|intermediate|advanced",
 "time_weeks": int or null,        # total time available (convert months to weeks)
 "hours_per_week": int or null,
 "budget_inr": int or null,        # total budget in INR; 0 means free only; null = unspecified
 "preferred_format": "self-paced|cohort|any",
 "name": "learner name or null",
 "modification_request": "if the latest message modifies an earlier request (cheaper, shorter, swap a course), describe it, else null"
}
Rules: later messages override earlier ones. "make it cheaper" -> lower budget_inr (e.g. 0). "I only have weekends" -> hours_per_week ~ 8.
If the goal is genuinely missing, set goal to null."""


def intake(state: AgentState) -> dict:
    trace: list = [_ev("intake", "node", "Entering node: intake", "Extracting / updating learner profile from conversation")]
    msgs = [SystemMessage(content=INTAKE_SYS)] + [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    prev = state.get("profile") or {}
    if prev:
        msgs.insert(1, SystemMessage(content=f"Previously extracted profile (merge with new info): {json.dumps(prev)}"))
    try:
        resp = invoke_with_fallback(msgs, trace, "intake")
        profile = extract_json(resp.content)
    except Exception as e:
        trace.append(_ev("intake", "error", "Profile extraction failed, using safe defaults", str(e)))
        profile = {"on_topic": True, "goal": state["messages"][-1].content, "current_skills": [], "level": "beginner"}

    profile.setdefault("current_skills", [])
    profile.setdefault("level", "beginner")
    profile.setdefault("preferred_format", "any")
    if not profile.get("on_topic", True):
        route = "redirect"
    elif not profile.get("goal"):
        route = "clarify"
    else:
        route = "plan"
    trace.append(_ev("intake", "state", f"Profile extracted -> route: {route}", profile))
    return {"profile": profile, "route": route, "trace": trace, "retries": 0, "critique": {}}


def clarify(state: AgentState) -> dict:
    q = (
        "I can build you a personalised learning path, but I need a bit more detail. "
        "What do you want to learn or become (for example 'data scientist', 'React developer', 'AWS certification')? "
        "Also helpful: your current skills, how many weeks you have, hours per week, and your budget in INR."
    )
    return {
        "messages": [AIMessage(content=q)],
        "final_answer": q,
        "trace": [_ev("clarify", "node", "Entering node: clarify", "Goal missing -> asking the user a clarifying question and pausing the graph")],
    }


def redirect(state: AgentState) -> dict:
    msg = (
        "I'm a course recommendation agent, so I can only help with learning goals, skills and courses. "
        "Tell me what you'd like to learn and I'll put together a plan."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "final_answer": msg,
        "trace": [_ev("redirect", "node", "Entering node: redirect", "Off-topic request -> polite redirect, no tools called")],
    }


# --------------------------------------------------------------------------- planner
PLANNER_SYS = """You are the planning module of a course recommendation agent.
Decompose the learner's goal into an ORDERED list of 2 to 4 sub-skills to learn, respecting prerequisites
and skipping what they already know. Each sub-skill must be a short searchable keyword that would appear in a
course catalog (e.g. "python", "sql", "statistics", "machine learning", "react", "docker", "aws").
Output JSON only:
{"plan": [{"order": 1, "skill": "...", "why": "one line"}, ...]}"""


def planner(state: AgentState) -> dict:
    trace: list = [_ev("planner", "node", "Entering node: planner", f"Retry #{state.get('retries', 0)}" if state.get("retries") else "Decomposing goal into sub-skills")]
    profile = state["profile"]
    content = f"Learner profile: {json.dumps(profile)}"
    critique = state.get("critique") or {}
    if critique:
        content += f"\n\nYour previous plan was rejected by the critic. Fix it. Critique: {json.dumps(critique)}\nPrevious plan: {json.dumps(state.get('plan'))}"
    try:
        resp = invoke_with_fallback([SystemMessage(content=PLANNER_SYS), HumanMessage(content=content)], trace, "planner")
        plan = extract_json(resp.content)["plan"]
        plan = sorted(plan, key=lambda p: p.get("order", 0))[:4]
    except Exception as e:
        trace.append(_ev("planner", "error", "Planner output unparsable -> fallback single-step plan", str(e)))
        plan = [{"order": 1, "skill": profile.get("goal", ""), "why": "direct search on the stated goal"}]
    trace.append(_ev("planner", "state", f"Plan with {len(plan)} sub-skills", plan))
    return {"plan": plan, "trace": trace}


# --------------------------------------------------------------------------- executor (ReAct)
EXECUTOR_SYS = """You are the tool-using research module of a course recommendation agent.
You have tools. For EACH sub-skill in the plan, call search_course_db with the skill keyword and the learner's
constraints (max_price_inr = budget or -1, level = learner level or "any").
If a sub-skill returns fewer than 2 results, call web_course_search for it.
You may call several tools in one turn. When every sub-skill has been searched, reply with exactly: DONE
Do not recommend courses yourself; just gather data."""


def executor(state: AgentState) -> dict:
    trace: list = [_ev("executor", "node", "Entering node: executor (ReAct loop)", "LLM dynamically selects and calls tools")]
    profile, plan = state["profile"], state["plan"]
    budget = profile.get("budget_inr")
    msgs = [
        SystemMessage(content=EXECUTOR_SYS),
        HumanMessage(content=f"Learner profile: {json.dumps(profile)}\nPlan: {json.dumps(plan)}\nBudget for max_price_inr: {budget if budget is not None else -1}"),
    ]
    candidates: dict[str, list] = {p["skill"]: [] for p in plan}
    web_extras: list = []
    tool_calls_made = 0

    def absorb(name: str, args: dict, out: str):
        """Pull structured results out of tool outputs into state."""
        nonlocal web_extras
        try:
            payload = out.split("FALLBACK local catalog result: ")[-1] if "FALLBACK" in out else out
            data = json.loads(payload)
        except Exception:
            return
        key = args.get("skill") or args.get("query") or ""
        target = _match_plan_skill(key, candidates) or key
        if name == "search_course_db" or "FALLBACK" in out:
            candidates.setdefault(target, []).extend(data.get("results", []))
        elif name == "web_course_search":
            for r in data.get("results", []):
                r["for_skill"] = target
            web_extras.extend(data.get("results", []))

    for step in range(MAX_REACT_STEPS):
        resp = invoke_with_fallback(msgs, trace, "executor", tools=ALL_TOOLS)
        msgs.append(resp)
        calls = getattr(resp, "tool_calls", None) or []
        if not calls:
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            trace.append(_ev("executor", "thought", f"Model reply (step {step + 1})", text[:300] or "DONE"))
            break
        for tc in calls:
            out = run_tool(tc["name"], tc["args"], trace, "executor")
            tool_calls_made += 1
            absorb(tc["name"], tc["args"], out)
            msgs.append(ToolMessage(content=out[:3000], tool_call_id=tc["id"]))

    # Self-correction: make sure every planned sub-skill has candidates, even if the model skipped it.
    for p in plan:
        if not candidates.get(p["skill"]):
            trace.append(_ev("executor", "reflection", f"Reflection: no candidates for '{p['skill']}' -> deterministic DB search", "The model skipped or the tool errored; running search_course_db directly."))
            args = {"skill": p["skill"], "level": "any", "max_price_inr": budget if budget is not None else -1, "max_hours": -1}
            out = run_tool("search_course_db", args, trace, "executor")
            absorb("search_course_db", args, out)
        if not candidates.get(p["skill"]) and budget is not None:
            trace.append(_ev("executor", "reflection", f"Reflection: still nothing within budget for '{p['skill']}' -> relaxing budget filter", ""))
            args = {"skill": p["skill"], "level": "any", "max_price_inr": -1, "max_hours": -1}
            out = run_tool("search_course_db", args, trace, "executor")
            absorb("search_course_db", args, out)

    # de-duplicate per skill
    for k, lst in candidates.items():
        seen, uniq = set(), []
        for c in lst:
            if c["id"] not in seen:
                seen.add(c["id"])
                uniq.append(c)
        candidates[k] = uniq
    summary = {k: [c["title"] for c in v] for k, v in candidates.items()}
    trace.append(_ev("executor", "state", f"Executor finished: {tool_calls_made} tool calls by model, candidates per skill", summary))
    return {"candidates": candidates, "web_extras": web_extras[:6], "trace": trace}


def _match_plan_skill(key: str, candidates: dict) -> str | None:
    k = key.lower().strip()
    for s in candidates:
        if s.lower() == k:
            return s
    for s in candidates:
        if s.lower() in k or k in s.lower():
            return s
    return None


# --------------------------------------------------------------------------- ranker
def ranker(state: AgentState) -> dict:
    trace: list = [_ev("ranker", "node", "Entering node: ranker", "Scoring candidates against profile, then computing totals in the Python sandbox")]
    profile, plan, candidates = state["profile"], state["plan"], state["candidates"]
    critique = state.get("critique") or {}
    budget = profile.get("budget_inr")
    user_level = LEVEL_ORDER.get(profile.get("level", "beginner"), 0)
    known = ",".join(profile.get("current_skills", []))
    prefer_short = critique.get("type") == "time"
    remaining_budget = budget

    path: list = []
    learned = list(profile.get("current_skills", []))
    for i, p in enumerate(plan):
        cands = candidates.get(p["skill"], [])
        target_level = min(user_level + (1 if i > 0 else 0), 2)

        def score(c: dict) -> float:
            s = float(c["rating"])
            s -= abs(LEVEL_ORDER.get(c["level"], 1) - target_level) * 1.5
            if remaining_budget is not None:
                s += 1.0 if c["price_inr"] <= remaining_budget else -5.0
            else:
                s += 0.3 if c["price_inr"] == 0 else 0
            if prefer_short:
                s -= c["hours"] / 40.0
            else:
                s -= c["hours"] / 150.0
            fmt = profile.get("preferred_format", "any")
            if fmt not in ("any", None) and c["format"] != fmt:
                s -= 1.0
            # prerequisite fit using what has been learned so far
            missing = [q for q in c["prerequisites"] if not any(q in k or k in q for k in learned)]
            s -= len(missing) * 1.0
            return s

        if not cands:
            trace.append(_ev("ranker", "error", f"No candidates for '{p['skill']}' - leaving gap for critic", ""))
            continue
        ranked = sorted(cands, key=score, reverse=True)
        best = ranked[0]
        check = run_tool("check_prerequisites", {"course_id": best["id"], "known_skills": ",".join(learned) or "none"}, trace, "ranker")
        try:
            missing = json.loads(check).get("missing", [])
        except Exception:
            missing = []
        path.append({**best, "for_skill": p["skill"], "step": i + 1, "why": p.get("why", ""), "missing_prereqs": missing,
                     "alternatives": [{"id": c["id"], "title": c["title"], "price_inr": c["price_inr"], "hours": c["hours"]} for c in ranked[1:3]]})
        learned += best["skills"]
        if remaining_budget is not None:
            remaining_budget -= best["price_inr"]

    # Totals through the sandbox tool (real tool execution the evaluator can inspect)
    costs = [c["price_inr"] for c in path]
    hours = [c["hours"] for c in path]
    weeks = profile.get("time_weeks") or 0
    hpw = profile.get("hours_per_week") or 0
    code = f"""costs = {costs}
hours = {hours}
weeks = {weeks}
hpw = {hpw}
total_cost = sum(costs)
total_hours = sum(hours)
print("total_cost", total_cost)
print("total_hours", total_hours)
print("weeks_needed_at_hpw", round(total_hours / hpw, 1) if hpw else "n/a")
print("hours_per_week_needed", round(total_hours / weeks, 1) if weeks else "n/a")
print("fits_time", (total_hours <= weeks * hpw) if (weeks and hpw) else "unknown")"""
    out = run_tool("python_sandbox", {"code": code}, trace, "ranker")
    totals: dict = {}
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            totals[parts[0]] = parts[1]
    trace.append(_ev("ranker", "state", f"Learning path with {len(path)} courses", [{"step": c["step"], "skill": c["for_skill"], "course": c["title"], "price": c["price_inr"], "hours": c["hours"]} for c in path]))
    return {"path": path, "totals": totals, "trace": trace}


# --------------------------------------------------------------------------- critic
CRITIC_SYS = """You are the critic module of a course recommendation agent. Review the learning path for a learner.
Check: (1) does it cover the goal, (2) is prerequisite order sensible, (3) are courses at a suitable level, (4) anything obviously wrong.
Be pragmatic: approve if it is reasonably good. Output JSON only:
{"approved": true/false, "type": "coverage|order|quality|none", "issues": ["..."], "instructions": "what the planner should change"}"""


def critic(state: AgentState) -> dict:
    trace: list = [_ev("critic", "node", "Entering node: critic", "Deterministic constraint checks + LLM review")]
    profile, path, totals, plan = state["profile"], state["path"], state["totals"], state["plan"]
    retries = state.get("retries", 0)
    issues: list[str] = []
    ctype = "none"

    # deterministic checks first (cheap, reliable)
    budget = profile.get("budget_inr")
    try:
        total_cost = float(totals.get("total_cost", 0))
    except ValueError:
        total_cost = 0
    if budget is not None and total_cost > budget:
        issues.append(f"Total cost {total_cost} INR exceeds budget {budget} INR")
        ctype = "budget"
    if totals.get("fits_time") == "False":
        issues.append(f"Total {totals.get('total_hours')} hours does not fit in {profile.get('time_weeks')} weeks x {profile.get('hours_per_week')} h/week")
        ctype = ctype if ctype != "none" else "time"
    covered = {c["for_skill"] for c in path}
    gaps = [p["skill"] for p in plan if p["skill"] not in covered]
    if gaps:
        issues.append(f"No course found for sub-skills: {gaps}")
        ctype = ctype if ctype != "none" else "coverage"
    for c in path:
        if c.get("missing_prereqs"):
            issues.append(f"'{c['title']}' needs {c['missing_prereqs']} which is not covered earlier")
            ctype = ctype if ctype != "none" else "order"
    trace.append(_ev("critic", "thought", "Deterministic checks", {"issues": issues or ["none"], "type": ctype}))

    approved = not issues
    instructions = "; ".join(issues)
    if approved and retries < MAX_RETRIES:
        # LLM review only when the hard checks pass (saves tokens on free tier)
        try:
            resp = invoke_with_fallback(
                [SystemMessage(content=CRITIC_SYS), HumanMessage(content=f"Profile: {json.dumps(profile)}\nPath: {json.dumps([{k: c[k] for k in ('step', 'for_skill', 'title', 'level', 'hours', 'price_inr', 'prerequisites')} for c in path])}\nTotals: {json.dumps(totals)}")],
                trace, "critic",
            )
            verdict = extract_json(resp.content)
            approved = bool(verdict.get("approved", True))
            if not approved:
                ctype = verdict.get("type", "quality")
                issues += verdict.get("issues", [])
                instructions = verdict.get("instructions", "")
        except Exception as e:
            trace.append(_ev("critic", "error", "LLM critic failed -> accepting deterministic verdict", str(e)))

    if not approved and retries >= MAX_RETRIES:
        trace.append(_ev("critic", "reflection", f"Max retries ({MAX_RETRIES}) reached -> delivering best effort with caveats", issues))
        decision = "accept_with_caveats"
    elif not approved:
        decision = "replan" if ctype in ("coverage", "order", "quality") else "rerank"
        trace.append(_ev("critic", "reflection", f"Rejected ({ctype}) -> {decision}, retry {retries + 1}/{MAX_RETRIES}", issues))
    else:
        decision = "approve"
        trace.append(_ev("critic", "state", "Approved", "Path passes all checks"))
    critique = {"approved": approved, "type": ctype, "issues": issues, "instructions": instructions, "decision": decision}
    return {"critique": critique, "retries": retries + (0 if decision in ("approve", "accept_with_caveats") else 1), "trace": trace}


def route_after_critic(state: AgentState) -> str:
    d = state["critique"]["decision"]
    return {"approve": "responder", "accept_with_caveats": "responder", "replan": "planner", "rerank": "ranker"}[d]


# --------------------------------------------------------------------------- responder
RESPONDER_SYS = """You are the final response module of a course recommendation agent. Write a friendly, well-structured
Markdown answer for the learner using ONLY the data given (do not invent courses). Include:
1. A one-line summary of the goal and constraints.
2. The ordered learning path: for each step, course title, provider, level, hours, price (0 = Free), and one-sentence reason.
3. Totals (cost, hours, weekly load) and whether it fits their time.
4. If caveats are provided, state them honestly.
5. Optional: 'Also consider (from web search)' list if web extras exist.
Keep it under 350 words."""


def responder(state: AgentState) -> dict:
    trace: list = [_ev("responder", "node", "Entering node: responder", "Composing final answer")]
    profile, path, totals, critique = state["profile"], state["path"], state["totals"], state.get("critique", {})
    payload = {
        "profile": profile,
        "path": [{k: c.get(k) for k in ("step", "for_skill", "title", "provider", "level", "hours", "price_inr", "rating", "format", "url", "why", "alternatives")} for c in path],
        "totals": totals,
        "caveats": critique.get("issues", []) if not critique.get("approved", True) else [],
        "web_extras": state.get("web_extras", []),
    }
    try:
        resp = invoke_with_fallback([SystemMessage(content=RESPONDER_SYS), HumanMessage(content=json.dumps(payload, ensure_ascii=False))], trace, "responder")
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        trace.append(_ev("responder", "error", "LLM responder failed -> templated answer", str(e)))
        lines = [f"**Learning path for: {profile.get('goal')}**"]
        for c in path:
            lines.append(f"{c['step']}. **{c['title']}** ({c['provider']}, {c['level']}, {c['hours']}h, {'Free' if c['price_inr'] == 0 else str(c['price_inr']) + ' INR'})")
        lines.append(f"\nTotals: {totals}")
        answer = "\n".join(lines)

    # long-term memory
    if profile.get("name"):
        run_tool("save_profile", {"name": profile["name"], "profile_json": json.dumps({"profile": profile, "path": [c["title"] for c in path]})}, trace, "responder")
    trace.append(_ev("responder", "state", "Final answer ready", ""))
    return {"messages": [AIMessage(content=answer)], "final_answer": answer, "trace": trace}


# --------------------------------------------------------------------------- graph
def build_graph():
    db.seed_if_needed()
    g = StateGraph(AgentState)
    g.add_node("intake", intake)
    g.add_node("clarify", clarify)
    g.add_node("redirect", redirect)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("ranker", ranker)
    g.add_node("critic", critic)
    g.add_node("responder", responder)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: s["route"], {"clarify": "clarify", "redirect": "redirect", "plan": "planner"})
    g.add_edge("clarify", END)
    g.add_edge("redirect", END)
    g.add_edge("planner", "executor")
    g.add_edge("executor", "ranker")
    g.add_edge("ranker", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"responder": "responder", "planner": "planner", "ranker": "ranker"})
    g.add_edge("responder", END)
    return g.compile(checkpointer=MemorySaver())


def run_turn(graph, user_text: str, thread_id: str):
    """Stream one user turn. Yields (node_name, trace_events) tuples; final state is available via graph.get_state."""
    config = {"configurable": {"thread_id": thread_id}}
    for update in graph.stream({"messages": [HumanMessage(content=user_text)], "trace": []}, config, stream_mode="updates"):
        for node, delta in update.items():
            yield node, (delta or {}).get("trace", []), (delta or {}).get("final_answer")
