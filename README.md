# Course Recommendation Agent

An autonomous agentic AI system (LangGraph) that turns a learner's goal into an ordered,
budget- and time-aware learning path. Built for the M.Tech Advanced Multi-Agent Systems live demo.
Uses only **free** LLM APIs (Groq / Gemini free tier / local Ollama) and keyless DuckDuckGo search.

## Architecture

```
intake ──(goal missing)──> clarify ──> END
  │
  ├──(off topic)─────────> redirect ──> END
  │
  ▼
planner ──> executor (ReAct tool loop) ──> ranker ──> critic ──┬──> responder ──> END
  ▲                                         ▲                 │
  └──────── coverage / order / quality ─────┼─── budget/time ─┘   (max 2 retries)
```

| Rubric item | Where it is |
|---|---|
| Agent workflow & planning | `agent/graph.py` – LangGraph `StateGraph`, planner decomposes goal into 2-4 sub-skills, conditional edges |
| Tool calling & execution | `agent/tools.py` – 6 tools: SQLite catalog, live web search, prerequisite checker, Python sandbox, save/load profile. LLM picks them dynamically in the executor |
| State management & self-correction | `MemorySaver` checkpointer keeps profile across turns; critic node rejects and loops back; web-search failure falls back to local DB; LLM rate-limit falls back to a second model; empty results relax filters |
| Live demo | `app.py` (Streamlit, chat + live trace pane) or `cli.py` |

## Setup (5 minutes)

```bash
cd course-recommendation-agent
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
copy .env.example .env          # cp on Mac/Linux
```

Get a free Groq key at <https://console.groq.com/keys>, paste it into `.env` as `GROQ_API_KEY=...`.
(Default models are `openai/gpt-oss-120b` (primary) and `openai/gpt-oss-20b` (fallback); the older Llama 3 models were retired by Groq. Alternative: `LLM_PROVIDER=gemini` with a free key from <https://aistudio.google.com/apikey>, or
`LLM_PROVIDER=ollama` after `ollama pull llama3.1` for a fully offline run.)

## Run

```bash
streamlit run app.py      # UI with live trace
python cli.py             # terminal version
python test_offline.py    # tools + graph sanity check, no API key needed
```

## 10-minute demo script

1. **Normal query** – "I know basic Python and want to become a data scientist in 6 months, 10 hours a week, budget 1000 INR."
   Walk the evaluator through the trace: intake → profile JSON, planner → sub-skills, executor → tool calls, ranker → sandbox totals, critic → verdict.
2. **State retention** – "Make it cheaper, only free courses." The profile updates (budget 0) without restating the goal.
3. **Replanning** – "I only have weekends, about 6 hours a week." Critic flags the time constraint → ranker reruns preferring shorter courses.
4. **Tool failure** – toggle *Simulate web-search outage* in the sidebar and ask for something niche. Trace shows the error, the reflection and the fallback to the local catalog.
5. **Edge cases** – an off-topic question is redirected; an empty query triggers a clarifying question; nonsense skills produce a best-effort answer with caveats.

## Project layout

```
agent/
  llm.py     free-provider factory + model fallback + JSON extraction
  db.py      SQLite catalog (seeded from data/courses.csv, 100 courses)
  tools.py   @tool definitions + run_tool with error handling / fallback
  graph.py   LangGraph nodes, routing, checkpointer
app.py       Streamlit UI
cli.py       terminal UI
test_offline.py  sanity tests without an LLM
data/courses.csv sample catalog (edit freely)
memory/      saved learner profiles
```
