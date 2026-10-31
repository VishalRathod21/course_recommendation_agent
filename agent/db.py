"""SQLite course catalog. Seeded from data/courses.csv on first use."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "courses.csv"
DB_PATH = ROOT / "data" / "courses.db"

LEVELS = {"beginner": 0, "intermediate": 1, "advanced": 2}


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def seed_if_needed() -> int:
    """Create / refresh the DB from the CSV. Returns row count."""
    needs = not DB_PATH.exists() or os.path.getmtime(CSV_PATH) > os.path.getmtime(DB_PATH)
    if needs:
        df = pd.read_csv(CSV_PATH).fillna("")
        with connect() as con:
            con.execute("DROP TABLE IF EXISTS courses")
            df.to_sql("courses", con, index=False)
            con.execute("CREATE INDEX IF NOT EXISTS idx_domain ON courses(domain)")
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM courses").fetchone()[0]


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["skills"] = [s.strip() for s in str(d["skills"]).split(",") if s.strip()]
    d["prerequisites"] = [s.strip() for s in str(d["prerequisites"]).split(",") if s.strip()]
    return d


def search_courses(
    skill: str,
    level: str = "any",
    max_price_inr: int | None = None,
    max_hours: int | None = None,
    limit: int = 6,
) -> list[dict]:
    """Keyword search over title / skills / domain with simple relevance scoring."""
    seed_if_needed()
    tokens = [t for t in skill.lower().replace("/", " ").replace("-", " ").split() if len(t) > 1]
    if not tokens:
        return []
    where, params = [], []
    # any token matches title, skills or domain
    tok_clauses = []
    for t in tokens:
        tok_clauses.append("(LOWER(title) LIKE ? OR LOWER(skills) LIKE ? OR LOWER(domain) LIKE ?)")
        params += [f"%{t}%"] * 3
    where.append("(" + " OR ".join(tok_clauses) + ")")
    if level and level != "any" and level in LEVELS:
        where.append("level = ?")
        params.append(level)
    if max_price_inr is not None and max_price_inr >= 0:
        where.append("price_inr <= ?")
        params.append(max_price_inr)
    if max_hours is not None and max_hours > 0:
        where.append("duration_hours <= ?")
        params.append(max_hours)
    sql = "SELECT * FROM courses WHERE " + " AND ".join(where)
    with connect() as con:
        rows = [_row_to_dict(r) for r in con.execute(sql, params).fetchall()]

    full = skill.lower().strip()

    def relevance(c: dict) -> float:
        hay_skills = " ".join(c["skills"])
        score = 0.0
        if full in hay_skills:
            score += 3
        if full in c["title"].lower():
            score += 2
        if full == c["domain"].lower():
            score += 2
        for t in tokens:
            if any(t in s for s in c["skills"]):
                score += 1
            if t in c["title"].lower():
                score += 0.5
            if t in c["domain"].lower():
                score += 0.5
        return score + float(c["rating"]) / 10

    rows.sort(key=relevance, reverse=True)
    return rows[:limit]


def get_course(course_id: int) -> dict | None:
    seed_if_needed()
    with connect() as con:
        r = con.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
    return _row_to_dict(r) if r else None
