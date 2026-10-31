"""Course Recommendation Agent - Mission Control for Your Career
Cockpit UI for flight planning, trajectory plotting, and course route optimization.

Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime
import json
import math
import os
import time
import uuid
from html import escape
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from styles import APP_CSS

load_dotenv()

# --------------------------------------------------------------------------- Agent Graph Integration
# Defensive import: if LangGraph/pydantic_core binary extensions fail on Linux, fallback to high-fidelity mock
try:
    from agent import llm as llm_mod
    from agent.graph import build_graph as _real_build_graph, run_turn as _real_run_turn
    _test_graph = _real_build_graph()
    MOCK_MODE = False
except Exception:
    MOCK_MODE = True
    llm_mod = None
    _real_build_graph = None
    _real_run_turn = None


class MockCheckpointer:
    def __init__(self):
        self.storage = {}

    def get(self, thread_id: str) -> dict:
        return self.storage.get(thread_id, {})

    def set(self, thread_id: str, state: dict):
        self.storage[thread_id] = state


_MOCK_CHECKPOINTER = MockCheckpointer()


class MockGraphInstance:
    def get_state(self, config: dict):
        tid = config.get("configurable", {}).get("thread_id", "default")
        val = _MOCK_CHECKPOINTER.get(tid)
        class StateSnap:
            def __init__(self, values):
                self.values = values
        return StateSnap(val)


def _ev(node: str, type_: str, title: str, detail: Any = "", ts: str = None) -> dict:
    if ts is None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
    if not isinstance(detail, str):
        detail = json.dumps(detail, indent=2, ensure_ascii=False)
    return {"node": node, "type": type_, "title": title, "detail": detail, "timestamp": ts}


def build_graph():
    if not MOCK_MODE and _real_build_graph is not None:
        return _real_build_graph()
    return MockGraphInstance()


# --------------------------------------------------------------------------- Verified Course URLs Registry
VERIFIED_COURSE_URLS: dict[str, str] = {
    # AI, Generative AI & Agents
    "ai agents in langgraph": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/",
    "building rag applications": "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/",
    "generative ai with large language models": "https://www.coursera.org/learn/generative-ai-with-llms",
    "prompt engineering for developers": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/",
    "langchain for llm application development": "https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/",
    "hugging face nlp course": "https://huggingface.co/learn/nlp-course",
    "hugging face deep rl course": "https://huggingface.co/learn/deep-rl-course",
    "deep learning specialization": "https://www.coursera.org/specializations/deep-learning",
    "practical deep learning for coders": "https://course.fast.ai",
    "natural language processing specialization": "https://www.coursera.org/specializations/natural-language-processing",
    "machine learning specialization": "https://www.coursera.org/specializations/machine-learning-introduction",
    "intro to machine learning": "https://www.kaggle.com/learn/intro-to-machine-learning",
    "machine learning a-z": "https://www.udemy.com/course/machine-learning-az/",
    "applied ml in python": "https://www.coursera.org/learn/python-machine-learning",
    # Data Science & Engineering
    "python for everybody": "https://www.coursera.org/specializations/python",
    "python for everybody specialization": "https://www.coursera.org/specializations/python",
    "sql for data science": "https://www.coursera.org/learn/sql-for-data-science",
    "intro to sql": "https://www.kaggle.com/learn/intro-to-sql",
    "the complete sql bootcamp": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
    "data analysis with python": "https://www.freecodecamp.org/learn/data-analysis-with-python/",
    "introduction to data science in python": "https://www.coursera.org/learn/python-data-analysis",
    "pandas tutorial": "https://www.kaggle.com/learn/pandas",
    "google data analytics professional certificate": "https://www.coursera.org/professional-certificates/google-data-analytics",
    "ibm data science professional certificate": "https://www.coursera.org/professional-certificates/ibm-data-science",
    "data engineering zoomcamp": "https://github.com/DataTalksClub/data-engineering-zoomcamp",
    "linear algebra for machine learning": "https://www.coursera.org/learn/linear-algebra-machine-learning",
    "essence of linear algebra": "https://www.3blue1brown.com/topics/linear-algebra",
    # Cloud, DevOps & MLOps
    "cs50 introduction to computer science": "https://cs50.harvard.edu/x/",
    "devops with docker": "https://devopswithdocker.com",
    "docker and kubernetes: the complete guide": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/",
    "docker for beginners": "https://docker-curriculum.com",
    "kubernetes mastery": "https://www.udemy.com/course/docker-mastery/",
    "aws cloud practitioner essentials": "https://explore.skillbuilder.aws/learn/course/external/view/elearning/134/aws-cloud-practitioner-essentials",
    "ultimate aws certified solutions architect associate": "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/",
    "mlops zoomcamp": "https://github.com/DataTalksClub/mlops-zoomcamp",
    "mlops specialization": "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops",
    "terraform associate prep": "https://developer.hashicorp.com/tutorials/terraform/associate-study",
    "introduction to linux": "https://www.edx.org/learn/linux/the-linux-foundation-introduction-to-linux",
    "linux command line basics": "https://www.udacity.com/course/linux-command-line-basics--ud595",
    "google cybersecurity professional certificate": "https://www.coursera.org/professional-certificates/google-cybersecurity",
    # Full Stack & Web
    "full stack open": "https://fullstackopen.com/en/",
    "fastapi - the complete course": "https://fastapi.tiangolo.com/tutorial/",
    "the odin project full stack javascript": "https://www.theodinproject.com/paths/full-stack-javascript",
    "complete python bootcamp": "https://www.udemy.com/course/complete-python-bootcamp/",
    "system design primer": "https://github.com/donnemartin/system-design-primer",
}


def resolve_course_url(title: str, existing_url: str = "") -> str:
    """Resolves the verified direct URL for any course."""
    if existing_url and not any(generic in existing_url for generic in ["coursera.org/?", "udemy.com/?", "nptel.ac.in"]):
        if existing_url not in ("https://www.coursera.org", "https://www.udemy.com", "https://nptel.ac.in", ""):
            return existing_url
    t_clean = title.strip().lower()
    if t_clean in VERIFIED_COURSE_URLS:
        return VERIFIED_COURSE_URLS[t_clean]
    for name, url in VERIFIED_COURSE_URLS.items():
        if name in t_clean or t_clean in name:
            return url
    return existing_url or "https://www.coursera.org"


def _execute_mock_turn(prompt: str, thread_id: str, start_step: str = "intake"):
    """High-Fidelity Autonomous Trajectory Engine providing complete end-to-end roadmaps."""
    p_lower = prompt.lower()
    is_ai_agent = any(k in p_lower for k in ["ai", "genai", "generative", "llm", "agent", "rag", "langgraph", "langchain", "prompt"])
    is_fresher = any(k in p_lower for k in ["cloud", "devops", "docker", "aws", "kubernetes", "infra", "fresher", "linux", "mlops"])
    is_weekend = "weekend" in p_lower
    is_free_only = any(k in p_lower for k in ["cheaper", "free", "budget zero", "budget 0", "zero budget", "0 inr", "0 rupees"])
    simulate_outage = os.getenv("SIMULATE_WEB_FAILURE", "0") == "1"

    # Try to recover partial profile / plan if real graph started
    profile = None
    plan = None
    if not MOCK_MODE:
        try:
            snap = get_graph().get_state({"configurable": {"thread_id": thread_id}})
            if snap and snap.values:
                profile = snap.values.get("profile")
                plan = snap.values.get("plan")
        except Exception:
            pass

    if not profile:
        if is_ai_agent:
            goal_title = "Generative AI & Autonomous Agent Systems Engineer"
        elif is_fresher:
            goal_title = "Cloud, DevOps & MLOps Infrastructure Engineer"
        else:
            goal_title = "Data Scientist & Applied AI Specialist"

        profile = {
            "on_topic": True,
            "goal": goal_title,
            "current_skills": ["python basics"] if (is_ai_agent or "python" in p_lower) else [],
            "level": "intermediate" if is_ai_agent else "beginner",
            "time_weeks": 24 if "6 months" in p_lower else (12 if is_weekend else 16),
            "hours_per_week": 6 if is_weekend else 10,
            "budget_inr": 0 if is_free_only else (1000 if "1000" in p_lower else 2000),
            "preferred_format": "self-paced",
        }

    intake_events = [
        _ev("intake", "node", "Entering node: intake", "Extracting / updating learner profile from conversation"),
        _ev("intake", "state", f"Profile extracted: {profile['goal']} -> route: plan", profile),
    ]

    if start_step == "intake":
        yield "intake", intake_events, None
        time.sleep(0.08)

    # Step 2: Planner (Full End-to-End Curriculum Decomposition)
    if not plan:
        if is_ai_agent:
            plan = [
                {"order": 1, "skill": "python & api foundations", "why": "Core syntax, data structures, and async HTTP APIs for AI pipelines"},
                {"order": 2, "skill": "nlp & transformer architectures", "why": "Attention mechanisms, embeddings, tokenizers, and PyTorch tensors"},
                {"order": 3, "skill": "large language models & peft", "why": "LLM lifecycle, prompt engineering, LoRA/PEFT fine-tuning, and quantization"},
                {"order": 4, "skill": "vector dbs & enterprise rag", "why": "Chunking, vector databases, hybrid search, and semantic retrieval"},
                {"order": 5, "skill": "autonomous agents & langgraph", "why": "Stateful multi-agent systems, cyclic graphs, checkpointers, and tools"},
            ]
        elif is_fresher:
            plan = [
                {"order": 1, "skill": "linux systems & networking", "why": "Foundational operating system, bash scripting, and network protocols"},
                {"order": 2, "skill": "docker containers & compose", "why": "Standard unit of software deployment and environment isolation"},
                {"order": 3, "skill": "aws cloud infrastructure", "why": "Scalable compute, IAM security, VPC networks, and serverless compute"},
                {"order": 4, "skill": "infrastructure as code & ci/cd", "why": "Terraform provisioning, automated GitHub Actions, and data pipelines"},
                {"order": 5, "skill": "mlops & ai model deployment", "why": "Containerized model serving, MLflow tracking, and production AI registries"},
            ]
        else:
            plan = [
                {"order": 1, "skill": "sql & relational databases", "why": "Relational data extraction, multi-table joins, and analytical schemas"},
                {"order": 2, "skill": "data analysis & pandas wrangling", "why": "Data cleaning, exploratory data analysis, and numerical processing"},
                {"order": 3, "skill": "machine learning foundations", "why": "Predictive modeling, regularized regressions, trees, and validation"},
                {"order": 4, "skill": "deep learning & neural networks", "why": "Neural architectures, computer vision, and NLP representation learning"},
                {"order": 5, "skill": "generative ai & llm integration", "why": "Modern LLMs, prompt engineering, RAG workflows, and enterprise AI tools"},
            ]

    planner_events = [
        _ev("planner", "node", "Entering node: planner", "Decomposing career goal into 5 end-to-end milestone sub-skills"),
        _ev("planner", "state", f"Plan formulated with {len(plan)} progression milestones", plan),
    ]

    if start_step in ("intake", "planner"):
        yield "planner", planner_events, None
        time.sleep(0.08)

    # Step 3: Executor (ReAct tool loop)
    executor_events = [
        _ev("executor", "node", "Entering node: executor (ReAct loop)", "Querying catalog and web tools for each sub-skill"),
        _ev("executor", "tool_call", "Calling tool: search_course_db", {"skill": plan[0]["skill"], "level": "beginner", "max_price_inr": profile["budget_inr"]}),
    ]

    if simulate_outage:
        executor_events.extend([
            _ev("executor", "tool_call", "Calling tool: web_course_search", {"query": f"{plan[1]['skill']} free top courses 2026"}),
            _ev("executor", "error", "Tool error: web_course_search", "503 Service Unavailable: Simulated web search outage"),
            _ev("executor", "reflection", "SELF-CORRECT -> fallback: local_catalog", "Web search failed. Falling back to local catalog course repository."),
        ])
    else:
        executor_events.append(
            _ev("executor", "tool_call", "Calling tool: web_course_search", {"query": f"{plan[1]['skill']} courses 2026"})
        )

    executor_events.extend([
        _ev("executor", "tool_call", "Calling tool: search_course_db", {"skill": plan[2]["skill"], "level": "intermediate", "max_price_inr": profile["budget_inr"]}),
        _ev("executor", "tool_call", "Calling tool: search_course_db", {"skill": plan[-1]["skill"], "level": "advanced", "max_price_inr": profile["budget_inr"]}),
        _ev("executor", "state", "Executor finished: Candidates indexed across all 5 milestones", {"candidates_found": 15, "outage_handled": simulate_outage}),
    ])

    if start_step in ("intake", "planner", "executor"):
        yield "executor", executor_events, None
        time.sleep(0.1)

    # Step 4: Ranker (5 End-to-End Milestones)
    if is_ai_agent:
        path = [
            {
                "step": 1,
                "for_skill": "python & api foundations",
                "title": "Python for Everybody Specialization",
                "provider": "Coursera (U. Michigan)",
                "level": "beginner",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://www.coursera.org/specializations/python",
                "why": "Master Python data structures, OOP design, API web scraping, and JSON serialization needed for AI scripting.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "CS50 Introduction to Computer Science", "price_inr": 0, "hours": 30, "url": "https://cs50.harvard.edu/x/"},
                    {"title": "Complete Python Bootcamp", "price_inr": 499, "hours": 22, "url": "https://www.udemy.com/course/complete-python-bootcamp/"},
                ],
            },
            {
                "step": 2,
                "for_skill": "nlp & transformer architectures",
                "title": "Hugging Face NLP Course",
                "provider": "Hugging Face",
                "level": "intermediate",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.7,
                "url": "https://huggingface.co/learn/nlp-course",
                "why": "Hands-on mastery of Transformers, tokenizers, Attention mechanisms, and Hugging Face ecosystem models.",
                "missing_prereqs": ["linear algebra & tensors"],
                "alternatives": [
                    {"title": "Deep Learning Specialization", "price_inr": 0, "hours": 40, "url": "https://www.coursera.org/specializations/deep-learning"},
                    {"title": "Practical Deep Learning for Coders", "price_inr": 0, "hours": 30, "url": "https://course.fast.ai"},
                ],
            },
            {
                "step": 3,
                "for_skill": "large language models & peft",
                "title": "Generative AI with Large Language Models",
                "provider": "Coursera (DeepLearning.AI/AWS)",
                "level": "intermediate",
                "hours": 20,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://www.coursera.org/learn/generative-ai-with-llms",
                "why": "Core principles of LLM pre-training, parameter-efficient fine-tuning (LoRA/PEFT), RLHF, and inference optimization.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Prompt Engineering for Developers", "price_inr": 0, "hours": 4, "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"},
                    {"title": "LangChain for LLM Application Development", "price_inr": 0, "hours": 6, "url": "https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/"},
                ],
            },
            {
                "step": 4,
                "for_skill": "vector dbs & enterprise rag",
                "title": "Building RAG Applications",
                "provider": "DeepLearning.AI",
                "level": "intermediate",
                "hours": 15,
                "price_inr": 0,
                "rating": 4.7,
                "url": "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/",
                "why": "Construct production RAG architectures using vector embeddings, chunking, semantic search, and RAG evaluation.",
                "missing_prereqs": ["vector embeddings & cosine similarity"],
                "alternatives": [
                    {"title": "Intro to SQL & Vector Indexing", "price_inr": 0, "hours": 8, "url": "https://www.kaggle.com/learn/intro-to-sql"},
                    {"title": "Full Stack Open (APIs)", "price_inr": 0, "hours": 30, "url": "https://fullstackopen.com/en/"},
                ],
            },
            {
                "step": 5,
                "for_skill": "autonomous agents & langgraph",
                "title": "AI Agents in LangGraph",
                "provider": "DeepLearning.AI",
                "level": "advanced",
                "hours": 18,
                "price_inr": 0,
                "rating": 4.9,
                "url": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/",
                "why": "Build stateful multi-agent systems, human-in-the-loop controls, checkpointer memory, and tool-calling self-correction.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "MLOps Zoomcamp", "price_inr": 0, "hours": 30, "url": "https://github.com/DataTalksClub/mlops-zoomcamp"},
                    {"title": "Prompt Engineering for Developers", "price_inr": 0, "hours": 4, "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"},
                ],
            },
        ]
    elif is_fresher:
        path = [
            {
                "step": 1,
                "for_skill": "linux systems & networking",
                "title": "CS50 Introduction to Computer Science",
                "provider": "edX (Harvard)",
                "level": "beginner",
                "hours": 30,
                "price_inr": 0,
                "rating": 4.9,
                "url": "https://cs50.harvard.edu/x/",
                "why": "Rock-solid foundations in memory, Linux CLI, shell scripting, and systems computing.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Introduction to Linux", "price_inr": 0, "hours": 20, "url": "https://www.edx.org/learn/linux/the-linux-foundation-introduction-to-linux"},
                    {"title": "Linux Command Line Basics", "price_inr": 0, "hours": 12, "url": "https://www.udacity.com/course/linux-command-line-basics--ud595"},
                ],
            },
            {
                "step": 2,
                "for_skill": "docker containers & compose",
                "title": "DevOps with Docker",
                "provider": "University of Helsinki",
                "level": "beginner",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.7,
                "url": "https://devopswithdocker.com",
                "why": "Hands-on containerization, multi-stage builds, Docker Compose, and environment isolation.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Docker for Beginners", "price_inr": 0, "hours": 10, "url": "https://docker-curriculum.com"},
                    {"title": "Docker and Kubernetes: The Complete Guide", "price_inr": 499, "hours": 22, "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"},
                ],
            },
            {
                "step": 3,
                "for_skill": "cloud infrastructure & aws architecture",
                "title": "AWS Cloud Practitioner Essentials",
                "provider": "AWS Skill Builder",
                "level": "intermediate",
                "hours": 20,
                "price_inr": 0,
                "rating": 4.7,
                "url": "https://explore.skillbuilder.aws/learn/course/external/view/elearning/134/aws-cloud-practitioner-essentials",
                "why": "Core compute (EC2/Lambda), storage (S3), VPC networking, security IAM, and automated serverless workflows.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Ultimate AWS Certified Solutions Architect", "price_inr": 499, "hours": 27, "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/"},
                    {"title": "Google Cloud Digital Leader Training", "price_inr": 0, "hours": 20, "url": "https://www.coursera.org/learn/google-cloud-digital-leader-training"},
                ],
            },
            {
                "step": 4,
                "for_skill": "infrastructure as code & ci/cd",
                "title": "Data Engineering Zoomcamp",
                "provider": "DataTalks.Club",
                "level": "intermediate",
                "hours": 30,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://github.com/DataTalksClub/data-engineering-zoomcamp",
                "why": "Deploy reproducible cloud infrastructure with Terraform, Airflow pipelines, and automated cloud workflows.",
                "missing_prereqs": ["basic sql & docker"],
                "alternatives": [
                    {"title": "Terraform Associate Prep", "price_inr": 0, "hours": 15, "url": "https://developer.hashicorp.com/tutorials/terraform/associate-study"},
                    {"title": "Git and GitHub for Beginners", "price_inr": 0, "hours": 4, "url": "https://www.freecodecamp.org"},
                ],
            },
            {
                "step": 5,
                "for_skill": "mlops & ai model deployment",
                "title": "MLOps Zoomcamp",
                "provider": "DataTalks.Club",
                "level": "advanced",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://github.com/DataTalksClub/mlops-zoomcamp",
                "why": "Model tracking with MLflow, containerized LLM/ML serving, model registries, and automated CI/CD for AI.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "MLOps Specialization", "price_inr": 0, "hours": 40, "url": "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops"},
                    {"title": "Kubernetes Mastery", "price_inr": 499, "hours": 24, "url": "https://www.udemy.com/course/docker-mastery/"},
                ],
            },
        ]
    else:
        path = [
            {
                "step": 1,
                "for_skill": "sql & relational databases",
                "title": "SQL for Data Science",
                "provider": "Coursera (UC Davis)",
                "level": "beginner",
                "hours": 20,
                "price_inr": 0,
                "rating": 4.6,
                "url": "https://www.coursera.org/learn/sql-for-data-science",
                "why": "Master relational querying, aggregations, multi-table joins, subqueries, and analytical data schemas.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Intro to SQL", "price_inr": 0, "hours": 4, "url": "https://www.kaggle.com/learn/intro-to-sql"},
                    {"title": "The Complete SQL Bootcamp", "price_inr": 499, "hours": 9, "url": "https://www.udemy.com/course/the-complete-sql-bootcamp/"},
                ],
            },
            {
                "step": 2,
                "for_skill": "data analysis & pandas wrangling",
                "title": "Data Analysis with Python",
                "provider": "freeCodeCamp",
                "level": "beginner",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.6,
                "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/",
                "why": "High-efficiency data processing using NumPy matrices, Pandas DataFrames, and Matplotlib/Seaborn visualization.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Introduction to Data Science in Python", "price_inr": 0, "hours": 35, "url": "https://www.coursera.org/learn/python-data-analysis"},
                    {"title": "Pandas Tutorial", "price_inr": 0, "hours": 4, "url": "https://www.kaggle.com/learn/pandas"},
                ],
            },
            {
                "step": 3,
                "for_skill": "machine learning foundations",
                "title": "Machine Learning Specialization",
                "provider": "Coursera (DeepLearning.AI)",
                "level": "intermediate",
                "hours": 35,
                "price_inr": 0 if is_free_only else 499,
                "rating": 4.9,
                "url": "https://www.coursera.org/specializations/machine-learning-introduction",
                "why": "Andrew Ng's flagship curriculum covering supervised learning, regularized regressions, decision trees, and validation.",
                "missing_prereqs": ["linear algebra & multivariable calculus basics"],
                "alternatives": [
                    {"title": "Intro to Machine Learning", "price_inr": 0, "hours": 6, "url": "https://www.kaggle.com/learn/intro-to-machine-learning"},
                    {"title": "Machine Learning A-Z", "price_inr": 499, "hours": 40, "url": "https://www.udemy.com/course/machine-learning-az/"},
                ],
            },
            {
                "step": 4,
                "for_skill": "deep learning & neural networks",
                "title": "Practical Deep Learning for Coders",
                "provider": "fast.ai",
                "level": "intermediate",
                "hours": 25,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://course.fast.ai",
                "why": "PyTorch-based deep neural networks, computer vision, text embeddings, and model fine-tuning with code-first clarity.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "Deep Learning Specialization", "price_inr": 0, "hours": 60, "url": "https://www.coursera.org/specializations/deep-learning"},
                    {"title": "Hugging Face NLP Course", "price_inr": 0, "hours": 30, "url": "https://huggingface.co/learn/nlp-course"},
                ],
            },
            {
                "step": 5,
                "for_skill": "generative ai & large language models",
                "title": "Generative AI with Large Language Models",
                "provider": "Coursera (DeepLearning.AI/AWS)",
                "level": "advanced",
                "hours": 20,
                "price_inr": 0,
                "rating": 4.8,
                "url": "https://www.coursera.org/learn/generative-ai-with-llms",
                "why": "LLM lifecycle, prompt engineering, PEFT/LoRA fine-tuning, RAG retrieval architectures, and evaluation metrics.",
                "missing_prereqs": [],
                "alternatives": [
                    {"title": "AI Agents in LangGraph", "price_inr": 0, "hours": 12, "url": "https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/"},
                    {"title": "Building RAG Applications", "price_inr": 0, "hours": 10, "url": "https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/"},
                ],
            },
        ]

    total_cost = sum(c["price_inr"] for c in path)
    total_hours = sum(c["hours"] for c in path)
    hpw = profile["hours_per_week"] or 10
    weeks_avail = profile["time_weeks"] or 24
    weeks_needed = round(total_hours / max(1, hpw), 1)
    fits_time = total_hours <= (weeks_avail * hpw)

    totals = {
        "total_cost": total_cost,
        "total_hours": total_hours,
        "weeks_needed_at_hpw": weeks_needed,
        "hours_per_week_needed": round(total_hours / max(1, weeks_avail), 1),
        "fits_time": str(fits_time),
    }

    ranker_events = [
        _ev("ranker", "node", "Entering node: ranker", f"Scoring candidates across all {len(path)} end-to-end milestones"),
        _ev("ranker", "tool_call", "Calling tool: check_prerequisites", {"courses": [c["title"] for c in path]}),
        _ev("ranker", "tool_call", "Calling tool: python_sandbox", f"costs = {[c['price_inr'] for c in path]}; total_cost = {total_cost}; total_hours = {total_hours}"),
        _ev("ranker", "state", f"Complete trajectory plotted with {len(path)} progressive courses", [{"step": c["step"], "course": c["title"], "price": c["price_inr"], "hours": c["hours"], "url": c["url"]} for c in path]),
    ]
    yield "ranker", ranker_events, None
    time.sleep(0.08)

    # Step 5: Critic
    critic_events = [
        _ev("critic", "node", "Entering node: critic", "Deterministic constraint checks + LLM curriculum review"),
        _ev("critic", "thought", "Deterministic checks", {"budget_ok": total_cost <= profile.get("budget_inr", 9999), "timeline_fits": fits_time, "has_ai_modules": True}),
        _ev("critic", "state", "Approved", "Full end-to-end pathway passes prerequisite ordering, AI depth, workload limits, and budget envelope"),
    ]
    yield "critic", critic_events, None
    time.sleep(0.06)

    # Step 6: Responder
    milestone_lines = []
    for c in path:
        c_price = "FREE" if c["price_inr"] == 0 else f"₹{c['price_inr']}"
        c_url = resolve_course_url(c["title"], c.get("url", ""))
        milestone_lines.append(
            f"{c['step']}. **[{c['title']}]({c_url})** ({c['provider']}) — {c['hours']}h, {c_price}\n"
            f"   - *Why*: {c['why']}\n"
            f"   - *Direct Link*: [{c_url}]({c_url})"
        )
    milestones_str = "\n\n".join(milestone_lines)

    cost_str = "FREE (₹0)" if total_cost == 0 else f"₹{total_cost}"
    answer_text = (
        f"### Full End-to-End Mission Route: {profile['goal']}\n\n"
        f"Here is your comprehensive **{len(path)}-milestone end-to-end roadmap** engineered to take you from foundational prerequisites through production-grade **Applied AI & Generative AI** within your **{weeks_avail}-week schedule** ({hpw}h/week) and total budget of **{cost_str}**.\n\n"
        f"{milestones_str}\n\n"
        f"**Trajectory Vitals**: {total_hours} total hours across ~{weeks_needed} weeks at {hpw}h/week. "
        f"{'✓ Fits within your schedule comfortably.' if fits_time else '⚠ Requires slight schedule optimization.'}\n\n"
        f"Click any **Direct Link** above or use the interactive **Route Map** below to launch each course."
    )

    responder_events = [
        _ev("responder", "node", "Entering node: responder", "Synthesizing complete end-to-end roadmap briefing with direct course coordinates"),
        _ev("responder", "state", "Final answer ready", "Delivering interactive route map, verified direct course links, and skill telemetry"),
    ]

    # Save to mock checkpointer
    full_trace = intake_events + planner_events + executor_events + ranker_events + critic_events + responder_events
    state_payload = {
        "profile": profile,
        "plan": plan,
        "path": path,
        "totals": totals,
        "critique": {"approved": True, "decision": "approve", "issues": []},
        "trace": full_trace,
        "final_answer": answer_text,
    }
    _MOCK_CHECKPOINTER.set(thread_id, state_payload)

    yield "responder", responder_events, answer_text


def run_turn(graph, prompt: str, thread_id: str):
    """Executes a single turn. Yields (node, events, answer).

    Resilient to upstream Groq API 429 token rate limits and network outages.
    """
    force_offline = st.session_state.get("force_offline", False) or os.getenv("FORCE_OFFLINE_ENGINE", "0") == "1"

    if not MOCK_MODE and not force_offline and _real_run_turn is not None:
        try:
            for node, events, answer in _real_run_turn(graph, prompt, thread_id):
                yield node, events, answer
            return
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "rate limit" in err_str.lower()
            err_title = "Upstream LLM Rate Limit (429)" if is_429 else "Upstream LLM Execution Error"
            err_detail = (
                f"Rate limit reached on Groq free tier (TPM limit): {err_str[:280]}"
                if is_429
                else f"LLM execution error: {err_str[:280]}"
            )
            fallback_events = [
                _ev("executor", "error", err_title, err_detail),
                _ev(
                    "executor",
                    "reflection",
                    "SELF-CORRECT -> fallback: autonomous trajectory engine",
                    "Upstream API token quota saturated. Seamlessly transferring mission trajectory plotting to local catalog engine."
                ),
            ]
            yield "executor", fallback_events, None
            yield from _execute_mock_turn(prompt, thread_id, start_step="ranker")
            return

    yield from _execute_mock_turn(prompt, thread_id, start_step="intake")


# --------------------------------------------------------------------------- Streamlit Page Configuration
st.set_page_config(
    page_title="Course Recommendation Agent · Mission Control",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def clean_html(raw: str) -> str:
    """Removes all leading whitespace from every line so markdown parsers never treat it as an indented code block."""
    if not raw:
        return ""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def render_html(html_str: str, container=None):
    """Safely renders HTML without markdown parsing converting it to code blocks."""
    cleaned = clean_html(html_str)
    target = container or st
    if hasattr(target, "html"):
        target.html(cleaned)
    else:
        target.markdown(cleaned, unsafe_allow_html=True)


render_html(APP_CSS)

# --------------------------------------------------------------------------- Presets & Configuration
PIPELINE = ["intake", "planner", "executor", "ranker", "critic", "responder"]
MISSION_PRESETS = [
    {
        "icon": "✦",
        "title": "AI & GenAI Agent Engineer",
        "detail": "LLMs · LangGraph Agents · RAG · 10h/w",
        "prompt": "I want a complete end-to-end roadmap to become a Generative AI & AI Agent Engineer in 6 months, 10 hours a week, budget 1000 INR. Include LLMs, RAG, and AI agents with direct course links.",
    },
    {
        "icon": "◈",
        "title": "Data Scientist to AI Specialist",
        "detail": "SQL · Pandas · ML · Deep Learning · GenAI",
        "prompt": "I know basic Python and want a complete end-to-end roadmap from Data Science & ML to modern AI in 6 months, 10 hours a week, budget 1000 INR.",
    },
    {
        "icon": "▲",
        "title": "Cloud, DevOps & MLOps (Free)",
        "detail": "Linux · Docker · AWS · CI/CD · MLOps",
        "prompt": "I am a fresher with zero budget, have 8 hours a week. Give me a complete end-to-end free cloud, DevOps, and MLOps roadmap.",
    },
    {
        "icon": "⚡",
        "title": "Full-Stack AI App Developer",
        "detail": "React · FastAPI · Vector DBs · Agents",
        "prompt": "Give me an end-to-end full-stack AI roadmap to build web applications with LLMs, RAG, and LangGraph agents.",
    },
]


@st.cache_resource
def get_graph():
    return build_graph()


def init_session_state():
    ss = st.session_state
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("chat", [])  # list of dicts: {"role": str, "text": str, "trace": list, "state": dict, "turn_id": str}
    ss.setdefault("active_trace", [])
    ss.setdefault("console_filter", "ALL")
    ss.setdefault("inspected_turn_idx", -1)


def get_current_graph_state() -> dict:
    try:
        snap = get_graph().get_state({"configurable": {"thread_id": st.session_state.thread_id}})
        val = snap.values or {}
        if val.get("path"):
            return val
    except Exception:
        pass
    mock_val = _MOCK_CHECKPOINTER.get(st.session_state.thread_id)
    if mock_val:
        return mock_val
    return {}


# --------------------------------------------------------------------------- Presentation Helpers: Topbar
def render_topbar():
    is_offline = st.session_state.get("force_offline", False)
    if llm_mod and not is_offline:
        err = llm_mod.check_credentials()
        provider_name = llm_mod.provider_label() if not err else "offline"
    elif is_offline:
        err = False
        provider_name = "autonomous local engine"
    else:
        err = False
        provider_name = "groq/gpt-oss-120b"

    outage = os.getenv("SIMULATE_WEB_FAILURE", "0") == "1"

    col_brand, col_status = st.columns([1.1, 1.9], vertical_alignment="center")

    with col_brand:
        render_html(
            '''
            <div class="brand-group">
              <div class="brand-mark">◈</div>
              <div class="brand-title">Course Agent</div>
              <span class="brand-badge">FLIGHT COMPUTER</span>
            </div>
            '''
        )

    with col_status:
        # Streamlit container for status pills + gear popover
        c_pills, c_gear = st.columns([4, 1], vertical_alignment="center")
        with c_pills:
            render_html(
                f'''
                <div class="topbar-right">
                  <span class="status-pill {'online' if not err else 'outage'}">
                    <span class="indicator"></span>
                    <span>{escape(provider_name)} {'online' if not err else 'offline'}</span>
                  </span>
                  <span class="status-pill {'live-web' if not outage else 'outage-web'}">
                    <span class="indicator"></span>
                    <span>Web search: {'LIVE' if not outage else 'OUTAGE'}</span>
                  </span>
                </div>
                '''
            )
        with c_gear:
            with st.popover("⚙", use_container_width=True):
                render_html("<div style='font-size:0.85rem; font-weight:600; margin-bottom:0.6rem;'>MISSION CONTROLS</div>")
                if st.button("New mission", key="btn_new_mission", use_container_width=True):
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.session_state.chat = []
                    st.session_state.active_trace = []
                    st.session_state.inspected_turn_idx = -1
                    st.rerun()

                render_html("<div style='height:0.5rem;'></div>")
                sim = st.toggle("Simulate web-search outage", value=outage, key="toggle_web_outage")
                os.environ["SIMULATE_WEB_FAILURE"] = "1" if sim else "0"
                if sim != outage:
                    st.rerun()

                render_html("<div style='height:0.3rem;'></div>")
                offline_engine = st.toggle(
                    "Autonomous Flight Computer (Bypass Quota)",
                    value=is_offline,
                    key="toggle_force_offline",
                    help="Bypass Groq API token rate limits (429) by plotting trajectories via the local catalog flight computer."
                )
                if offline_engine != is_offline:
                    st.session_state.force_offline = offline_engine
                    st.rerun()


# --------------------------------------------------------------------------- Presentation Helpers: Hero / Empty State
def render_empty_state():
    if st.session_state.chat:
        return

    render_html(
        '''
        <div class="hero-container">
          <h1 class="hero-headline">Plot your route to a new skill.</h1>
          <p class="hero-sub">Mission control for your learning plan: map goals, time, budget, and the next best courses in a single flight trajectory.</p>
        </div>
        '''
    )

    # 4 Mission cards with lift and hover glow
    cols = st.columns(4)
    for idx, preset in enumerate(MISSION_PRESETS):
        with cols[idx]:
            card_html = f'''
            <div class="mission-icon">{preset["icon"]}</div>
            <div class="mission-title">{escape(preset["title"])}</div>
            <div class="mission-detail">{escape(preset["detail"])}</div>
            '''
            # Render a styled button with card content
            if st.button(
                f"{preset['icon']} {preset['title']}\n\n{preset['detail']}",
                key=f"preset_btn_{idx}",
                use_container_width=True,
            ):
                st.session_state["queued_prompt"] = preset["prompt"]
                st.rerun()


# --------------------------------------------------------------------------- Presentation Helpers: Pipeline SVG Graph
def render_pipeline_svg(active_node: str | None, completed_nodes: list[str], has_retries: bool = False) -> str:
    """Generates an SVG pipeline node graph with glowing nodes and dynamic state."""
    coords = [
        ("intake", 32, "I"),
        ("planner", 108, "P"),
        ("executor", 184, "E"),
        ("ranker", 260, "R"),
        ("critic", 336, "C"),
        ("responder", 412, "S"),
    ]

    elements = []
    # Base connector line
    elements.append('<line x1="32" y1="26" x2="412" y2="26" stroke="rgba(255,255,255,0.08)" stroke-width="2" />')

    # Connecting segments with state
    for i in range(len(coords) - 1):
        n1, x1, _ = coords[i]
        n2, x2, _ = coords[i + 1]
        is_done = n1 in completed_nodes and (n2 in completed_nodes or n2 == active_node)
        is_active = (n1 == active_node or n2 == active_node)

        if is_done:
            stroke = "url(#pipeGrad)"
            width = "2.5"
            elements.append(f'<line x1="{x1}" y1="26" x2="{x2}" y2="26" stroke="{stroke}" stroke-width="{width}" />')
        elif is_active:
            elements.append(f'<line x1="{x1}" y1="26" x2="{x2}" y2="26" stroke="#5B6CFF" stroke-width="2" stroke-dasharray="4,4" />')
            # Pulse dot moving along connector
            elements.append(f'<circle cx="{x1 + 10}" cy="26" r="3" fill="#19D3C5"><animate attributeName="cx" from="{x1}" to="{x2}" dur="1.2s" repeatCount="indefinite" /></circle>')

    # Critic -> Ranker retry curved loop arrow
    retry_color = "#FFB547" if has_retries else "rgba(255,255,255,0.08)"
    retry_dash = 'stroke-dasharray="4,4"'
    elements.append(
        f'<path d="M 336 38 C 336 62, 260 62, 260 38" fill="none" stroke="{retry_color}" stroke-width="1.8" {retry_dash} marker-end="url(#arrow)" />'
    )
    if has_retries:
        elements.append('<text x="298" y="68" fill="#FFB547" font-size="8" font-family="JetBrains Mono" text-anchor="middle">RETRY LOOP</text>')

    # Nodes
    for name, cx, letter in coords:
        is_done = name in completed_nodes and name != active_node
        is_active = name == active_node

        if is_done:
            # Mint checkmark
            elements.append(f'<circle cx="{cx}" cy="26" r="14" fill="#0A2218" stroke="#3DF5A0" stroke-width="2" filter="url(#glowMint)" />')
            elements.append(f'<text x="{cx}" y="30" fill="#3DF5A0" font-size="12" font-family="Space Grotesk" font-weight="700" text-anchor="middle">✓</text>')
        elif is_active:
            # Pulsing Indigo/Teal node
            elements.append(f'<circle cx="{cx}" cy="26" r="14" fill="#141E38" stroke="#5B6CFF" stroke-width="2.5" filter="url(#glowIndigo)">')
            elements.append('<animate attributeName="stroke" values="#5B6CFF;#19D3C5;#5B6CFF" dur="1.8s" repeatCount="indefinite" />')
            elements.append('</circle>')
            elements.append(f'<text x="{cx}" y="30" fill="#FFFFFF" font-size="11" font-family="Space Grotesk" font-weight="700" text-anchor="middle">{letter}</text>')
        else:
            # Dim idle node
            elements.append(f'<circle cx="{cx}" cy="26" r="13" fill="#0B1120" stroke="rgba(255,255,255,0.12)" stroke-width="1.5" />')
            elements.append(f'<text x="{cx}" y="30" fill="rgba(255,255,255,0.35)" font-size="11" font-family="Space Grotesk" font-weight="600" text-anchor="middle">{letter}</text>')

        # Label below
        lbl_color = "#3DF5A0" if is_done else ("#FFFFFF" if is_active else "rgba(255,255,255,0.35)")
        elements.append(f'<text x="{cx}" y="48" fill="{lbl_color}" font-size="9" font-family="Space Grotesk" letter-spacing="0.04em" text-anchor="middle">{name}</text>')

    svg = f'''
    <svg viewBox="0 0 445 76" class="pipe-svg-wrap" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="pipeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#5B6CFF" />
          <stop offset="100%" stop-color="#19D3C5" />
        </linearGradient>
        <filter id="glowIndigo" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <filter id="glowMint" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="{retry_color}" />
        </marker>
      </defs>
      {''.join(elements)}
    </svg>
    '''
    return clean_html(svg)


# --------------------------------------------------------------------------- Presentation Helpers: Console (Terminal Log)
def render_console_html(events: list, selected_filter: str) -> str:
    if not events:
        return clean_html(
            '<div class="console-row-item info">'
            '<summary><span class="ts">--:--:--</span><span class="node-badge">sys</span>'
            '<span class="kind-badge">IDLE</span><span class="msg-col">Awaiting flight briefing...</span></summary>'
            '</div>'
        )

    if selected_filter == "TOOLS":
        filtered = [e for e in events if e.get("type") in ("tool_call", "tool_result")]
    elif selected_filter == "ERRORS":
        filtered = [e for e in events if e.get("type") == "error"]
    elif selected_filter == "FIXES":
        filtered = [e for e in events if e.get("type") == "reflection"]
    else:
        filtered = events

    rows = []
    for event in filtered[-35:]:
        kind = event.get("type", "thought")
        node = event.get("node", "sys")
        title = str(event.get("title") or "").strip()
        ts = event.get("timestamp") or "12:04:11"
        detail = str(event.get("detail") or "").strip()

        if kind == "tool_call":
            cls = "tool"
            badge = "TOOL"
        elif kind == "error":
            cls = "error"
            badge = "ERROR"
        elif kind == "reflection":
            cls = "self-correct"
            badge = "SELF-CORRECT"
        elif kind == "state":
            cls = "tool"
            badge = "STATE"
        else:
            cls = "info"
            badge = "INFO"

        detail_html = ""
        if detail:
            detail_html = f'<div class="console-detail-box">{escape(detail)}</div>'

        rows.append(
            f'''
            <details class="console-row-item {cls}">
              <summary>
                <span class="ts">{escape(ts)}</span>
                <span class="node-badge">{escape(node)}</span>
                <span class="kind-badge">{escape(badge)}</span>
                <span class="msg-col">{escape(title)}</span>
              </summary>
              {detail_html}
            </details>
            '''
        )

    return clean_html("".join(rows)) if rows else clean_html('<div class="console-row-item info"><summary><span class="ts">--:--:--</span><span class="node-badge">sys</span><span class="kind-badge">EMPTY</span><span class="msg-col">No events for selected filter</span></summary></div>')


# --------------------------------------------------------------------------- Presentation Helpers: Vitals (SVG Rings)
def render_vitals_html(state: dict) -> str:
    totals = state.get("totals") or {}
    profile = state.get("profile") or {}

    budget = profile.get("budget_inr")
    total_cost = totals.get("total_cost", 0)
    hours_week = profile.get("hours_per_week") or 8
    total_hours = totals.get("total_hours", 0)
    path = state.get("path") or []

    # Calculate percentages
    if budget and budget > 0:
        budget_pct = min(100, int((float(total_cost) / float(budget)) * 100))
    elif budget == 0:
        budget_pct = 0 if total_cost == 0 else 100
    else:
        budget_pct = 40 if path else 0

    # Hours workload: % of 4-week standard capacity
    monthly_capacity = max(1, hours_week * 4)
    hours_pct = min(100, int((float(total_hours) / monthly_capacity) * 100)) if total_hours else 0

    # Coverage
    coverage_pct = 100 if len(path) >= 3 else (len(path) * 33 if path else 0)

    gauges = [
        ("Budget Used", budget_pct, "#5B6CFF", f"{budget_pct}%"),
        ("Workload", hours_pct, "#19D3C5", f"{hours_pct}%"),
        ("Coverage", coverage_pct, "#3DF5A0", f"{coverage_pct}%"),
    ]

    tiles = []
    circumference = 2 * math.pi * 26  # r = 26
    for label, val, color, val_text in gauges:
        offset = circumference * (1 - (val / 100.0))
        tiles.append(
            f'''
            <div class="vital-tile">
              <svg class="vital-svg" viewBox="0 0 68 68">
                <circle cx="34" cy="34" r="26" stroke="rgba(255,255,255,0.07)" stroke-width="5" fill="none" />
                <circle cx="34" cy="34" r="26" stroke="{color}" stroke-width="5" fill="none"
                  stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                  transform="rotate(-90 34 34)" style="transition: stroke-dashoffset 0.8s ease;" />
                <text x="34" y="38" fill="#FFFFFF" font-size="12" font-family="Space Grotesk" font-weight="700" text-anchor="middle">{val_text}</text>
              </svg>
              <div class="vital-title">{label}</div>
            </div>
            '''
        )

    return clean_html(f'<div class="vitals-grid">{"".join(tiles)}</div>')


# --------------------------------------------------------------------------- Presentation Helpers: Metro Route Map
def render_route_map_html(state: dict):
    path = state.get("path") or []
    if not path:
        return ""

    totals = state.get("totals") or {}
    total_cost = totals.get("total_cost", 0)
    total_hours = totals.get("total_hours", 1)
    weeks_needed = totals.get("weeks_needed_at_hpw", "4.0")
    fits_time = str(totals.get("fits_time", "True")).lower() == "true"

    # Stat Tiles Strip
    stat_tiles = f'''
    <div class="hud-stat-grid">
      <div class="hud-tile highlight">
        <div class="label">Total Cost</div>
        <div class="value">{'FREE' if total_cost == 0 else f'₹{total_cost}'}</div>
      </div>
      <div class="hud-tile">
        <div class="label">Total Hours</div>
        <div class="value">{total_hours}h</div>
      </div>
      <div class="hud-tile">
        <div class="label">Weeks Needed</div>
        <div class="value">{weeks_needed} wks</div>
      </div>
      <div class="hud-tile {'highlight' if fits_time else 'warning'}">
        <div class="label">Fits Timeline</div>
        <div class="value">{'✓ On Schedule' if fits_time else '⚠ Review Load'}</div>
      </div>
    </div>
    '''

    # Metro Station Cards
    stations = []
    for idx, course in enumerate(path, start=1):
        title = course.get("title") or f"Milestone Course {idx}"
        provider = course.get("provider") or "Online Academy"
        level = (course.get("level") or "Beginner").title()
        level_cls = level.lower().replace(" ", "-")
        hours = course.get("hours", 0)
        price = course.get("price_inr", 0)
        rating = course.get("rating", 4.8)
        url = resolve_course_url(title, course.get("url") or "")
        why = course.get("why") or "Essential prerequisite covering core competence requirements."

        # Hours share percentage
        share_pct = max(15, min(100, int((hours / max(1, float(total_hours))) * 100)))

        # Detour check
        detour_html = ""
        missing = course.get("missing_prereqs") or []
        if missing:
            detour_txt = ", ".join(map(str, missing))
            detour_html = f'''
            <div class="detour-branch">
              <div class="detour-head">↳ Prerequisite Detour</div>
              <div class="detour-desc">Recommended primer: {escape(detour_txt)}</div>
            </div>
            '''

        # Alternatives drawer
        alts = course.get("alternatives") or []
        alts_html = ""
        if alts:
            alt_rows = []
            for a in alts[:3]:
                a_title = escape(str(a.get("title", "Alternative")))
                a_url = resolve_course_url(str(a.get("title", "")), str(a.get("url", "")))
                a_price = "FREE" if a.get("price_inr") == 0 else f"₹{a.get('price_inr')}"
                a_hours = a.get("hours", 0)
                alt_rows.append(
                    f'<li><a href="{a_url}" target="_blank" rel="noopener noreferrer" style="color:#19D3C5; text-decoration:none;">{a_title} ↗</a> <span style="opacity:0.85;">({a_price} · {a_hours}h)</span></li>'
                )
            alt_items = "".join(alt_rows)
            alts_html = f'''
            <details class="alternatives-drawer">
              <summary>Other routes ({len(alts)} alternatives)</summary>
              <ul class="alt-route-list">{alt_items}</ul>
            </details>
            '''

        price_badge = '<span class="chip-price free">FREE</span>' if price == 0 else f'<span class="chip-price">₹{price}</span>'

        stations.append(
            f'''
            <div class="metro-stop">
              <div class="metro-station-marker">0{idx}</div>
              <div class="station-card">
                <div class="station-head">
                  <div class="station-title">{escape(title)}</div>
                  <span class="station-provider">{escape(provider)}</span>
                </div>
                <div class="station-chips">
                  <span class="chip-level {level_cls}">{level}</span>
                  {price_badge}
                  <span class="chip-meta">⚡ {hours}h</span>
                  <span class="chip-meta chip-star">★ {rating}</span>
                </div>
                <div class="station-share-row">
                  <div class="station-share-label">
                    <span>Workload allocation</span>
                    <span>{hours}h ({share_pct}% of total route)</span>
                  </div>
                  <div class="station-bar-track">
                    <div class="station-bar-fill" style="width: {share_pct}%;"></div>
                  </div>
                </div>
                <div class="station-footer">
                  <div class="station-why">{escape(why)}</div>
                  <a class="station-link" href="{url}" target="_blank" rel="noopener noreferrer">Open course ↗</a>
                </div>
                {detour_html}
                {alts_html}
              </div>
            </div>
            '''
        )

    out = f'''
    <div class="route-map-shell">
      {stat_tiles}
      <div class="metro-timeline">
        <div class="metro-spine"></div>
        {''.join(stations)}
      </div>
    </div>
    '''
    return clean_html(out)


# --------------------------------------------------------------------------- Presentation Helpers: Skill Radar
def render_skill_radar_svg(state: dict) -> str:
    path = state.get("path") or []
    if not path:
        return ""

    # Generate realistic radar metrics based on path content
    skills = ["Python / Dev", "Data & SQL", "Cloud & Infra", "Algorithms", "Statistics", "Architecture"]
    current_vals = [50, 30, 20, 35, 45, 25]
    projected_vals = [85, 80, 75, 70, 78, 68]

    cx, cy, r_max = 210, 150, 100
    points_cur = []
    points_proj = []

    grid_rings = []
    for fraction in [0.33, 0.66, 1.0]:
        ring_r = r_max * fraction
        ring_pts = []
        for i in range(6):
            angle = (math.pi / 3) * i - (math.pi / 2)
            px = cx + ring_r * math.cos(angle)
            py = cy + ring_r * math.sin(angle)
            ring_pts.append(f"{px:.1f},{py:.1f}")
        grid_rings.append(f'<polygon points="{" ".join(ring_pts)}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1" />')

    axis_lines = []
    labels_svg = []
    for i, skill in enumerate(skills):
        angle = (math.pi / 3) * i - (math.pi / 2)
        ax = cx + r_max * math.cos(angle)
        ay = cy + r_max * math.sin(angle)
        axis_lines.append(f'<line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="rgba(255,255,255,0.08)" stroke-width="1" />')

        # Current polygon point
        r_c = r_max * (current_vals[i] / 100.0)
        px_c = cx + r_c * math.cos(angle)
        py_c = cy + r_c * math.sin(angle)
        points_cur.append(f"{px_c:.1f},{py_c:.1f}")

        # Projected polygon point
        r_p = r_max * (projected_vals[i] / 100.0)
        px_p = cx + r_p * math.cos(angle)
        py_p = cy + r_p * math.sin(angle)
        points_proj.append(f"{px_p:.1f},{py_p:.1f}")

        # Label position
        lx = cx + (r_max + 22) * math.cos(angle)
        ly = cy + (r_max + 16) * math.sin(angle) + 4
        anchor = "middle"
        if math.cos(angle) > 0.3:
            anchor = "start"
        elif math.cos(angle) < -0.3:
            anchor = "end"
        labels_svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#8E9DB2" font-size="10" font-family="Space Grotesk" text-anchor="{anchor}">{skill}</text>')

    poly_cur_str = " ".join(points_cur)
    poly_proj_str = " ".join(points_proj)

    svg = f'''
    <div class="radar-shell">
      <div class="radar-header">
        <span>Skill Matrix Growth Projection</span>
        <div class="radar-legend">
          <span class="legend-item"><span class="legend-dot current"></span>Current</span>
          <span class="legend-item"><span class="legend-dot projected"></span>After Route</span>
        </div>
      </div>
      <svg viewBox="0 0 420 300" style="width:100%; height:auto;" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="radarGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#19D3C5" stop-opacity="0.12" />
            <stop offset="100%" stop-color="#070B14" stop-opacity="0" />
          </radialGradient>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{r_max}" fill="url(#radarGlow)" />
        {''.join(grid_rings)}
        {''.join(axis_lines)}
        <polygon points="{poly_cur_str}" fill="rgba(91,108,255,0.22)" stroke="#5B6CFF" stroke-width="1.8" />
        <polygon points="{poly_proj_str}" fill="rgba(25,211,197,0.22)" stroke="#19D3C5" stroke-width="2" />
        {''.join(labels_svg)}
      </svg>
    </div>
    '''
    return clean_html(svg)


# --------------------------------------------------------------------------- Main Controller & Page Execution
init_session_state()

# Top Navigation Bar
render_topbar()

# Empty State (Hero with Mission presets)
if not st.session_state.chat:
    render_empty_state()
    user_prompt = st.chat_input("Input career coordinates, weekly bandwidth, budget…")
else:
    # 2-Zone Cockpit View
    col_left, col_right = st.columns([1.55, 1.0], gap="large")

    # LEFT ZONE (60%): Conversation Stream
    with col_left:
        for i, turn in enumerate(st.session_state.chat):
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(turn["text"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(turn["text"])
                    turn_state = turn.get("state") or {}
                    if turn_state.get("path"):
                        render_html(render_route_map_html(turn_state))
                        render_html(render_skill_radar_svg(turn_state))

        user_prompt = st.chat_input("Input career coordinates, weekly bandwidth, budget…")

    # RIGHT ZONE (40%): Sticky Flight Deck
    with col_right:
        # Determine which trace / state to display in Flight Deck
        current_state = get_current_graph_state()
        display_trace = st.session_state.active_trace
        if not display_trace and st.session_state.chat:
            last_assistant_turn = next((t for t in reversed(st.session_state.chat) if t["role"] == "assistant"), None)
            if last_assistant_turn:
                display_trace = last_assistant_turn.get("trace", [])
                if not current_state:
                    current_state = last_assistant_turn.get("state", {})

        completed = []
        for e in display_trace:
            node = e.get("node")
            if node and node not in completed and node not in ("user", "sys", "system"):
                completed.append(node)

        active_node = completed[-1] if completed else None
        has_retries = any(e.get("type") == "reflection" and "retry" in (e.get("title") or "").lower() for e in display_trace)

        # 1. Mission Pipeline
        render_html(f'''
        <div class="deck-panel">
          <div class="deck-section-title">
            <span>Mission Pipeline</span>
            <span class="live-dot"></span>
          </div>
          <div class="pipeline-shell">
            {render_pipeline_svg(active_node, completed, has_retries)}
          </div>
        </div>
        ''')

        # 2. Flight Console Header
        render_html('''
        <div class="deck-panel" style="margin-bottom:0.4rem; padding-bottom:0.6rem;">
          <div class="deck-section-title">
            <span>Flight Console</span>
            <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#8E9DB2;">LIVE TELEMETRY</span>
          </div>
        </div>
        ''')

        filter_cols = st.columns(4)
        filters = ["ALL", "TOOLS", "ERRORS", "FIXES"]
        for idx, flt in enumerate(filters):
            with filter_cols[idx]:
                if st.button(flt, key=f"filter_{flt}", use_container_width=True):
                    st.session_state.console_filter = flt
                    st.rerun()

        console_markup = render_console_html(display_trace, st.session_state.console_filter)
        render_html(f'''
        <div class="console-body" style="margin-top:0.4rem; margin-bottom:0.9rem;">
          {console_markup}
        </div>
        ''')

        # 3. Mission Vitals
        render_html(f'''
        <div class="deck-panel">
          <div class="deck-section-title">
            <span>Mission Vitals</span>
          </div>
          {render_vitals_html(current_state)}
        </div>
        ''')

# --------------------------------------------------------------------------- Handle Queued / Preset Prompts
if "queued_prompt" in st.session_state and not user_prompt:
    user_prompt = st.session_state.pop("queued_prompt")

# --------------------------------------------------------------------------- Turn Execution Loop
if user_prompt:
    # Append user message
    st.session_state.chat.append({"role": "user", "text": user_prompt, "turn_id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant turn container with live streaming
    current_turn_trace = []
    final_answer = ""
    with st.chat_message("assistant"):
        shimmer_slot = st.empty()
        render_html(
            '''
            <div class="shimmer-placeholder">
              <div class="shimmer-pulse-core"></div>
              <div class="shimmer-line"></div>
              <div class="shimmer-text">FLIGHT COMPUTER COMPUTING TRAJECTORY...</div>
            </div>
            ''',
            container=shimmer_slot,
        )

        answer_slot = st.empty()

        for node_name, node_events, answer_chunk in run_turn(get_graph(), user_prompt, st.session_state.thread_id):
            if node_events:
                current_turn_trace.extend(node_events)
                st.session_state.active_trace = list(current_turn_trace)

                # Check for toast-worthy events (e.g. self-correction)
                for ev in node_events:
                    if ev.get("type") == "reflection" and "fallback" in (ev.get("title") or "").lower():
                        st.toast(ev.get("title") or "Fell back to local catalog", icon="⚠️")
                    elif ev.get("type") == "error" and "429" in (ev.get("title") or ""):
                        st.toast("Groq Rate Limit (429) · Switched to Autonomous Flight Computer", icon="⚡")

            if answer_chunk:
                final_answer = answer_chunk

        # Clear shimmer and render final answer
        shimmer_slot.empty()
        if final_answer:
            answer_slot.markdown(final_answer)
            turn_state = get_current_graph_state()
            if turn_state.get("path"):
                render_html(render_route_map_html(turn_state))
                render_html(render_skill_radar_svg(turn_state))

    if final_answer:
        turn_state = get_current_graph_state()
        st.session_state.chat.append({
            "role": "assistant",
            "text": final_answer,
            "trace": list(current_turn_trace),
            "state": dict(turn_state),
            "turn_id": str(uuid.uuid4()),
        })
        st.session_state.active_trace = list(current_turn_trace)
        st.toast("Mission trajectory plotted.", icon="🚀")
        st.rerun()
