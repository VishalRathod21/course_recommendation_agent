"""Screenshot Generator for Course Recommendation Agent
Generates 5 portfolio-grade screenshot views using exact app layout and styles.
"""
from __future__ import annotations

import os
import subprocess
from html import escape

from styles import APP_CSS
import app

os.makedirs("screenshots", exist_ok=True)


def build_full_html(body_content: str, width: int = 1366, is_mobile: bool = False, height: int = 0, extra_script: str = "") -> str:
    """Wraps body content in full Streamlit-like HTML page with APP_CSS."""
    mobile_viewport = '<meta name="viewport" content="width=device-width, initial-scale=1.0">' if is_mobile else ""
    height_style = f"min-height: {height}px;" if height > 0 else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  {mobile_viewport}
  <title>Course Recommendation Agent · Mission Control</title>
  {APP_CSS}
  <style>
    body {{
      width: {width}px;
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      background-color: #070B14 !important;
      {height_style}
    }}
    .block-container {{
      max-width: {width - 60}px !important;
      margin: 0 auto !important;
    }}
  </style>
</head>
<body>
  <div class="stApp">
    <div class="block-container">
      {body_content}
    </div>
  </div>
  {extra_script}
</body>
</html>
"""


def capture_view(name: str, html_content: str, width: int = 1366, height: int = 900):
    html_path = f"screenshots/{name}.html"
    png_path = f"screenshots/{name}.png"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    cmd = [
        "wkhtmltoimage",
        "--width", str(width),
        "--format", "png",
        "--quality", "95",
        "--load-error-handling", "ignore",
    ]
    if height > 0:
        cmd.extend(["--height", str(height)])

    cmd.extend([html_path, png_path])

    print(f"Generating {png_path}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
        print(f"  ✓ Saved {png_path} ({os.path.getsize(png_path)} bytes)")
    else:
        print(f"  ✗ Failed {name}: {res.stderr}")


def get_topbar_html(provider: str = "groq/gpt-oss-120b", outage: bool = False) -> str:
    return f"""
    <div class="topbar-wrap">
      <div class="brand-group">
        <div class="brand-mark">◈</div>
        <div class="brand-title">Course Agent</div>
        <span class="brand-badge">FLIGHT COMPUTER</span>
      </div>
      <div class="topbar-right">
        <span class="status-pill online">
          <span class="indicator"></span>
          <span>{provider} online</span>
        </span>
        <span class="status-pill {'live-web' if not outage else 'outage-web'}">
          <span class="indicator"></span>
          <span>Web search: {'LIVE' if not outage else 'OUTAGE'}</span>
        </span>
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:999px; padding:0.4rem 0.8rem; font-size:0.85rem; color:#8E9DB2; display:flex; align-items:center; gap:0.3rem;">
          <span>⚙</span>
        </div>
      </div>
    </div>
    """


# --------------------------------------------------------------------------- View 1: Empty Hero State
def make_view_1_hero():
    topbar = get_topbar_html()
    presets_html = []
    for p in app.MISSION_PRESETS:
        presets_html.append(f"""
        <div class="mission-card-btn">
          <div class="mission-icon">{p["icon"]}</div>
          <div class="mission-title">{escape(p["title"])}</div>
          <div class="mission-detail">{escape(p["detail"])}</div>
        </div>
        """)

    body = f"""
    {topbar}
    <div class="hero-container">
      <h1 class="hero-headline">Plot your route to a new skill.</h1>
      <p class="hero-sub">Mission control for your learning plan: map goals, time, budget, and the next best courses in a single flight trajectory.</p>
      
      <!-- Center glowing composer -->
      <div style="max-width: 720px; margin: 0 auto 2.5rem; position: relative;">
        <div data-testid="stChatInput" style="display: flex; align-items: center; justify-content: space-between; border-radius: 18px; border: 1px solid rgba(91,108,255,0.35); background: rgba(10,16,28,0.92); padding: 0.75rem 1rem; box-shadow: 0 0 25px rgba(91,108,255,0.18);">
          <span style="color: #546277; font-size: 0.95rem;">Input career coordinates, weekly bandwidth, budget…</span>
          <button style="background: linear-gradient(135deg, #5B6CFF, #19D3C5); border-radius: 12px; border: none; width: 34px; height: 34px; color: #FFF; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center;">➤</button>
        </div>
      </div>

      <div class="mission-grid">
        {''.join(presets_html)}
      </div>
    </div>
    """
    html = build_full_html(body, width=1366, height=800)
    capture_view("1_empty_hero", html, width=1366, height=800)


# --------------------------------------------------------------------------- View 2: Mid-Run Pipeline & Console Active
def make_view_2_mid_run():
    topbar = get_topbar_html()
    user_prompt = "I know basic Python and want to become a data scientist in 6 months, 10 hours a week, budget 1000 INR."

    # Events mid-run (executor is currently active)
    active_node = "executor"
    completed = ["intake", "planner"]
    events = [
        {"timestamp": "12:04:08", "node": "intake", "type": "node", "title": "Entering node: intake", "detail": "Profile extraction"},
        {"timestamp": "12:04:09", "node": "intake", "type": "state", "title": "Profile extracted -> route: plan", "detail": '{"goal": "Python Data Scientist", "budget_inr": 1000}'},
        {"timestamp": "12:04:09", "node": "planner", "type": "node", "title": "Entering node: planner", "detail": "Decomposing goal into sub-skills"},
        {"timestamp": "12:04:10", "node": "planner", "type": "state", "title": "Plan with 3 sub-skills generated", "detail": '[{"order": 1, "skill": "sql & databases"}, {"order": 2, "skill": "data analysis"}, {"order": 3, "skill": "machine learning"}]'},
        {"timestamp": "12:04:11", "node": "executor", "type": "node", "title": "Entering node: executor (ReAct loop)", "detail": "LLM selecting tools"},
        {"timestamp": "12:04:11", "node": "executor", "type": "tool_call", "title": "Calling tool: search_course_db", "detail": '{"skill": "sql & databases", "level": "beginner", "max_price_inr": 1000}'},
        {"timestamp": "12:04:12", "node": "executor", "type": "tool_call", "title": "Calling tool: web_course_search", "detail": '{"query": "data analysis python top courses"}'},
    ]

    pipe_svg = app.render_pipeline_svg(active_node, completed, has_retries=False)
    console_html = app.render_console_html(events, "ALL")
    vitals_html = app.render_vitals_html({"totals": {"total_cost": 0, "total_hours": 30}, "profile": {"budget_inr": 1000, "hours_per_week": 10}, "path": [{}]})

    body = f"""
    {topbar}
    <div style="display: grid; grid-template-columns: 1.55fr 1fr; gap: 1.8rem; align-items: start;">
      <!-- LEFT: Chat stream -->
      <div class="chat-thread-zone">
        <div data-testid="stChatMessage" data-message-role="user">
          <div data-testid="stMarkdownContainer">
            {escape(user_prompt)}
          </div>
        </div>

        <div data-testid="stChatMessage" data-message-role="assistant">
          <div class="shimmer-placeholder">
            <div class="shimmer-pulse-core"></div>
            <div class="shimmer-line"></div>
            <div class="shimmer-text">FLIGHT COMPUTER COMPUTING TRAJECTORY (EXECUTOR TOOL DISPATCH)...</div>
          </div>
        </div>

        <!-- Docked bottom composer -->
        <div style="margin-top: 1.5rem;">
          <div data-testid="stChatInput" style="display: flex; align-items: center; justify-content: space-between; border-radius: 18px; border: 1px solid rgba(255,255,255,0.12); background: rgba(10,16,28,0.88); padding: 0.6rem 0.9rem;">
            <span style="color: #546277; font-size: 0.92rem;">Input career coordinates, weekly bandwidth, budget…</span>
            <button style="background: linear-gradient(135deg, #5B6CFF, #19D3C5); border-radius: 10px; border: none; width: 30px; height: 30px; color: #FFF; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center;">➤</button>
          </div>
        </div>
      </div>

      <!-- RIGHT: Sticky Flight Deck -->
      <div>
        <div class="flight-deck-sticky">
          <div class="deck-section-title">
            <span>Mission Pipeline</span>
            <span class="live-dot"></span>
          </div>
          <div class="pipeline-shell">
            {pipe_svg}
          </div>

          <div class="deck-section-title">
            <span>Flight Console</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#8E9DB2;">LIVE TELEMETRY</span>
          </div>
          <div class="console-filter-row">
            <div class="filter-btn-active"><button style="background:rgba(91,108,255,0.2); border:1px solid #5B6CFF; color:#FFF; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ALL</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">TOOLS</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ERRORS</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">FIXES</button></div>
          </div>
          <div class="console-shell">
            <div class="console-body">{console_html}</div>
          </div>

          <div class="deck-section-title">
            <span>Mission Vitals</span>
          </div>
          {vitals_html}
        </div>
      </div>
    </div>
    """
    html = build_full_html(body, width=1366, height=920)
    capture_view("2_mid_run_active", html, width=1366, height=920)


# --------------------------------------------------------------------------- View 3: Finished Route Map
def make_view_3_route_map():
    topbar = get_topbar_html()
    user_prompt = "I know basic Python and want to become a data scientist in 6 months, 10 hours a week, budget 1000 INR."

    # Realistic mock state
    state = {
        "profile": {"goal": "Python Data Scientist", "budget_inr": 1000, "hours_per_week": 10, "time_weeks": 24},
        "totals": {"total_cost": 599, "total_hours": 94, "weeks_needed_at_hpw": 9.4, "fits_time": "True"},
        "path": [
            {
                "step": 1,
                "title": "SQL for Data Science",
                "provider": "Coursera (UC Davis)",
                "level": "beginner",
                "hours": 20,
                "price_inr": 0,
                "rating": 4.6,
                "url": "https://www.coursera.org/learn/sql-for-data-science",
                "why": "Core relational database querying, filtering, subqueries, and table aggregations.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Intro to SQL", "price_inr": 0, "hours": 4},
                    {"title": "The Complete SQL Bootcamp", "price_inr": 499, "hours": 9},
                ],
            },
            {
                "step": 2,
                "title": "Data Analysis with Python",
                "provider": "freeCodeCamp",
                "level": "beginner",
                "hours": 30,
                "price_inr": 0,
                "rating": 4.5,
                "url": "https://www.freecodecamp.org",
                "why": "Industry-standard data manipulation with NumPy, Pandas, and Matplotlib visualizations.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Python Data Science Toolbox", "price_inr": 0, "hours": 8},
                ],
            },
            {
                "step": 3,
                "title": "Machine Learning Specialization",
                "provider": "Coursera (DeepLearning.AI)",
                "level": "intermediate",
                "hours": 44,
                "price_inr": 599,
                "rating": 4.9,
                "url": "https://www.coursera.org",
                "why": "Andrew Ng's flagship curriculum covering supervised learning, regressions, and trees.",
                "missing_prereqs": ["multivariable calculus basics"],
                "alternatives": [
                    {"title": "Intro to Machine Learning", "price_inr": 0, "hours": 6},
                    {"title": "Machine Learning A-Z", "price_inr": 499, "hours": 40},
                ],
            },
        ],
    }

    route_map_markup = app.render_route_map_html(state)
    radar_markup = app.render_skill_radar_svg(state)

    events = [
        {"timestamp": "12:04:08", "node": "intake", "type": "state", "title": "Profile extracted -> route: plan", "detail": '{"goal": "Python Data Scientist"}'},
        {"timestamp": "12:04:09", "node": "planner", "type": "state", "title": "Plan with 3 sub-skills generated", "detail": 'SQL, Pandas, Machine Learning'},
        {"timestamp": "12:04:11", "node": "executor", "type": "tool_call", "title": "Calling tool: search_course_db", "detail": "Queried local course database"},
        {"timestamp": "12:04:12", "node": "ranker", "type": "tool_call", "title": "Calling tool: python_sandbox", "detail": "total_cost = 599; fits_time = True"},
        {"timestamp": "12:04:13", "node": "critic", "type": "state", "title": "Approved", "detail": "All constraint checks passed"},
        {"timestamp": "12:04:14", "node": "responder", "type": "state", "title": "Final answer ready", "detail": "Delivered interactive route map"},
    ]

    pipe_svg = app.render_pipeline_svg(None, app.PIPELINE, has_retries=False)
    console_html = app.render_console_html(events, "ALL")
    vitals_html = app.render_vitals_html(state)

    body = f"""
    {topbar}
    <div style="display: grid; grid-template-columns: 1.55fr 1fr; gap: 1.8rem; align-items: start;">
      <!-- LEFT: Chat stream & Route Map -->
      <div class="chat-thread-zone">
        <div data-testid="stChatMessage" data-message-role="user">
          <div data-testid="stMarkdownContainer">
            {escape(user_prompt)}
          </div>
        </div>

        <div data-testid="stChatMessage" data-message-role="assistant">
          <div class="assistant-prose">
            <h3 style="margin-top:0;">Mission Route Confirmed: Python Data Scientist</h3>
            <p>Your personalized flight trajectory spans <strong>3 key milestones</strong> tailored to your weekly bandwidth of <strong>10h/week</strong> and total budget of <strong>₹1,000</strong>.</p>
          </div>
          {route_map_markup}
          {radar_markup}
        </div>

        <!-- Docked bottom composer -->
        <div style="margin-top: 1.5rem;">
          <div data-testid="stChatInput" style="display: flex; align-items: center; justify-content: space-between; border-radius: 18px; border: 1px solid rgba(255,255,255,0.12); background: rgba(10,16,28,0.88); padding: 0.6rem 0.9rem;">
            <span style="color: #546277; font-size: 0.92rem;">Input career coordinates, weekly bandwidth, budget…</span>
            <button style="background: linear-gradient(135deg, #5B6CFF, #19D3C5); border-radius: 10px; border: none; width: 30px; height: 30px; color: #FFF; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center;">➤</button>
          </div>
        </div>
      </div>

      <!-- RIGHT: Sticky Flight Deck -->
      <div>
        <div class="flight-deck-sticky">
          <div class="deck-section-title">
            <span>Mission Pipeline</span>
            <span class="live-dot"></span>
          </div>
          <div class="pipeline-shell">
            {pipe_svg}
          </div>

          <div class="deck-section-title">
            <span>Flight Console</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#8E9DB2;">LIVE TELEMETRY</span>
          </div>
          <div class="console-filter-row">
            <div class="filter-btn-active"><button style="background:rgba(91,108,255,0.2); border:1px solid #5B6CFF; color:#FFF; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ALL</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">TOOLS</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ERRORS</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">FIXES</button></div>
          </div>
          <div class="console-shell">
            <div class="console-body">{console_html}</div>
          </div>

          <div class="deck-section-title">
            <span>Mission Vitals</span>
          </div>
          {vitals_html}
        </div>
      </div>
    </div>
    """
    html = build_full_html(body, width=1366, height=1350)
    capture_view("3_finished_route_map", html, width=1366, height=1350)


# --------------------------------------------------------------------------- View 4: Error & Fallback Case
def make_view_4_error_fallback():
    topbar = get_topbar_html(outage=True)
    user_prompt = "I am a fresher, want to get into cloud and DevOps, budget zero."

    # Events illustrating ERROR + SELF-CORRECT reflection
    events = [
        {"timestamp": "12:04:08", "node": "intake", "type": "state", "title": "Profile extracted -> route: plan", "detail": '{"goal": "Cloud & DevOps", "budget_inr": 0}'},
        {"timestamp": "12:04:09", "node": "planner", "type": "state", "title": "Plan with 3 sub-skills generated", "detail": 'Linux, Docker, AWS'},
        {"timestamp": "12:04:10", "node": "executor", "type": "tool_call", "title": "Calling tool: search_course_db", "detail": '{"skill": "linux", "max_price_inr": 0}'},
        {"timestamp": "12:04:11", "node": "executor", "type": "tool_call", "title": "Calling tool: web_course_search", "detail": '{"query": "docker free courses"}'},
        {"timestamp": "12:04:11", "node": "executor", "type": "error", "title": "Tool error: web_course_search", "detail": '503 Service Unavailable: Simulated web search outage'},
        {"timestamp": "12:04:12", "node": "executor", "type": "reflection", "title": "SELF-CORRECT -> fallback: local_catalog", "detail": 'Web search failed. Falling back to local catalog course repository.'},
        {"timestamp": "12:04:13", "node": "ranker", "type": "tool_call", "title": "Calling tool: check_prerequisites", "detail": "Checking prerequisite graph"},
        {"timestamp": "12:04:14", "node": "critic", "type": "reflection", "title": "Critic retry 1/2: re-evaluating workload", "detail": 'Workload adjustment required'},
    ]

    # Show retry loop in pipeline
    pipe_svg = app.render_pipeline_svg("critic", ["intake", "planner", "executor", "ranker"], has_retries=True)
    console_html = app.render_console_html(events, "ALL")
    vitals_html = app.render_vitals_html({"totals": {"total_cost": 0, "total_hours": 66}, "profile": {"budget_inr": 0, "hours_per_week": 8}, "path": [{}, {}]})

    # Simulated Toast in top right
    toast_html = """
    <div style="position: fixed; top: 1.5rem; right: 1.5rem; z-index: 9999; background: rgba(14,21,37,0.95); border: 1px solid rgba(255,181,71,0.5); box-shadow: 0 0 25px rgba(255,181,71,0.25), 0 20px 40px rgba(0,0,0,0.6); border-radius: 14px; padding: 0.85rem 1.2rem; display: flex; align-items: center; gap: 0.7rem; color: #FFE7C2; font-size: 0.88rem; animation: floatFadeIn 0.3s ease-out both;">
      <span style="font-size: 1.2rem;">⚠️</span>
      <div>
        <div style="font-weight: 600; color: #FFB547; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase;">Self-Correction Triggered</div>
        <div>Fell back to local catalog due to web search outage.</div>
      </div>
    </div>
    """

    body = f"""
    {toast_html}
    {topbar}
    <div style="display: grid; grid-template-columns: 1.55fr 1fr; gap: 1.8rem; align-items: start;">
      <!-- LEFT: Chat stream -->
      <div class="chat-thread-zone">
        <div data-testid="stChatMessage" data-message-role="user">
          <div data-testid="stMarkdownContainer">
            {escape(user_prompt)}
          </div>
        </div>

        <div data-testid="stChatMessage" data-message-role="assistant">
          <div class="assistant-prose">
            <h3 style="margin-top:0;">Mission Route Confirmed: Cloud & DevOps Engineer</h3>
            <p>External web search encountered a connection anomaly. Agent self-corrected seamlessly using cached offline course telemetry.</p>
          </div>
        </div>

        <!-- Docked bottom composer -->
        <div style="margin-top: 1.5rem;">
          <div data-testid="stChatInput" style="display: flex; align-items: center; justify-content: space-between; border-radius: 18px; border: 1px solid rgba(255,255,255,0.12); background: rgba(10,16,28,0.88); padding: 0.6rem 0.9rem;">
            <span style="color: #546277; font-size: 0.92rem;">Input career coordinates, weekly bandwidth, budget…</span>
            <button style="background: linear-gradient(135deg, #5B6CFF, #19D3C5); border-radius: 10px; border: none; width: 30px; height: 30px; color: #FFF; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center;">➤</button>
          </div>
        </div>
      </div>

      <!-- RIGHT: Sticky Flight Deck -->
      <div>
        <div class="flight-deck-sticky">
          <div class="deck-section-title">
            <span>Mission Pipeline</span>
            <span class="live-dot" style="background:#FFB547; box-shadow:0 0 8px #FFB547;"></span>
          </div>
          <div class="pipeline-shell">
            {pipe_svg}
          </div>

          <div class="deck-section-title">
            <span>Flight Console</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#FFB547;">OUTAGE HANDLED</span>
          </div>
          <div class="console-filter-row">
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ALL</button></div>
            <div><button style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#8E9DB2; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">TOOLS</button></div>
            <div class="filter-btn-active"><button style="background:rgba(255,92,122,0.2); border:1px solid #FF5C7A; color:#FFF; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">ERRORS</button></div>
            <div><button style="background:rgba(255,181,71,0.2); border:1px solid #FFB547; color:#FFF; border-radius:8px; padding:0.25rem 0.6rem; font-size:0.7rem;">FIXES</button></div>
          </div>
          <div class="console-shell">
            <div class="console-body">{console_html}</div>
          </div>

          <div class="deck-section-title">
            <span>Mission Vitals</span>
          </div>
          {vitals_html}
        </div>
      </div>
    </div>
    """
    html = build_full_html(body, width=1366, height=920)
    capture_view("4_error_fallback", html, width=1366, height=920)


# --------------------------------------------------------------------------- View 5: Mobile Width (400px)
def make_view_5_mobile():
    topbar = f"""
    <div class="topbar-wrap" style="flex-direction: column; align-items: flex-start; gap: 0.6rem; padding-bottom: 0.8rem;">
      <div class="brand-group">
        <div class="brand-mark">◈</div>
        <div class="brand-title" style="font-size:0.95rem;">Course Agent</div>
        <span class="brand-badge" style="font-size:0.55rem;">FLIGHT COMPUTER</span>
      </div>
      <div class="topbar-right" style="width: 100%; justify-content: space-between;">
        <span class="status-pill online" style="font-size:0.65rem; padding:0.3rem 0.6rem;">
          <span class="indicator"></span>
          <span>groq online</span>
        </span>
        <span class="status-pill live-web" style="font-size:0.65rem; padding:0.3rem 0.6rem;">
          <span class="indicator"></span>
          <span>Web: LIVE</span>
        </span>
        <span style="font-size: 0.85rem; color:#8E9DB2;">⚙</span>
      </div>
    </div>
    """

    state = {
        "profile": {"goal": "Cloud & DevOps", "budget_inr": 0, "hours_per_week": 8, "time_weeks": 16},
        "totals": {"total_cost": 0, "total_hours": 66, "weeks_needed_at_hpw": 8.2, "fits_time": "True"},
        "path": [
            {
                "step": 1,
                "title": "CS50 Computer Science",
                "provider": "edX (Harvard)",
                "level": "beginner",
                "hours": 36,
                "price_inr": 0,
                "rating": 4.9,
                "url": "https://cs50.harvard.edu",
                "why": "Linux, C, and algorithmic systems.",
                "missing_prereqs": [],
            },
            {
                "step": 2,
                "title": "Data Engineering Zoomcamp",
                "provider": "DataTalks.Club",
                "level": "intermediate",
                "hours": 30,
                "price_inr": 0,
                "rating": 4.7,
                "url": "https://github.com",
                "why": "Containerization & Docker compositions.",
                "missing_prereqs": ["bash basics"],
            },
        ],
    }

    route_map_markup = app.render_route_map_html(state)
    pipe_svg = app.render_pipeline_svg("responder", ["intake", "planner", "executor", "ranker", "critic"], has_retries=False)
    events = [
        {"timestamp": "12:04:10", "node": "intake", "type": "state", "title": "Profile ready"},
        {"timestamp": "12:04:12", "node": "executor", "type": "tool_call", "title": "search_course_db"},
        {"timestamp": "12:04:14", "node": "responder", "type": "state", "title": "Route plotted"},
    ]
    console_html = app.render_console_html(events, "ALL")
    vitals_html = app.render_vitals_html(state)

    body = f"""
    {topbar}
    <div style="display: flex; flex-direction: column; gap: 1.2rem;">
      <!-- Chat message -->
      <div data-testid="stChatMessage" data-message-role="user">
        <div data-testid="stMarkdownContainer" style="font-size: 0.85rem; padding: 0.7rem 0.9rem;">
          Cloud & DevOps on a zero budget
        </div>
      </div>

      <div data-testid="stChatMessage" data-message-role="assistant">
        <div class="assistant-prose" style="font-size: 0.88rem;">
          <h4 style="margin: 0 0 0.4rem;">Route Plotted: Cloud & DevOps</h4>
          <p>Flight trajectory confirmed for zero budget with 2 core stations.</p>
        </div>
        {route_map_markup}
      </div>

      <!-- Flight Deck Collapsed Underneath on Mobile -->
      <div class="flight-deck-sticky" style="margin-top: 0.5rem; padding: 0.9rem;">
        <div class="deck-section-title" style="font-size: 0.7rem;">
          <span>Mission Pipeline</span>
          <span class="live-dot"></span>
        </div>
        <div class="pipeline-shell">
          {pipe_svg}
        </div>

        <div class="deck-section-title" style="font-size: 0.7rem; margin-top: 0.8rem;">
          <span>Live Console</span>
        </div>
        <div class="console-shell">
          <div class="console-body" style="font-size: 0.65rem; max-height: 160px;">{console_html}</div>
        </div>

        <div class="deck-section-title" style="font-size: 0.7rem; margin-top: 0.8rem;">
          <span>Vitals</span>
        </div>
        {vitals_html}
      </div>

      <!-- Docked Input -->
      <div data-testid="stChatInput" style="display: flex; align-items: center; justify-content: space-between; border-radius: 16px; border: 1px solid rgba(255,255,255,0.15); background: rgba(10,16,28,0.95); padding: 0.5rem 0.7rem; margin-top: 0.5rem;">
        <span style="color: #546277; font-size: 0.82rem;">Input coordinates…</span>
        <button style="background: linear-gradient(135deg, #5B6CFF, #19D3C5); border-radius: 8px; border: none; width: 28px; height: 28px; color: #FFF; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center;">➤</button>
      </div>
    </div>
    """
    html = build_full_html(body, width=400, is_mobile=True, height=1300)
    capture_view("5_mobile_width", html, width=400, height=1300)


if __name__ == "__main__":
    print("Capturing 5 required portfolio-grade screenshots...")
    make_view_1_hero()
    make_view_2_mid_run()
    make_view_3_route_map()
    make_view_4_error_fallback()
    make_view_5_mobile()
    print("All screenshots generated successfully!")
