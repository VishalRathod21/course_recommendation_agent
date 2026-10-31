"""LLM provider factory. Only FREE providers are wired up:

- groq   : console.groq.com   (free tier, tool calling, very fast)   <- default
- gemini : aistudio.google.com (free tier)
- ollama : fully local, no key

Also provides `invoke_with_fallback`, which retries on a smaller/other model when
the primary model errors (rate limit, timeout). That fallback is logged into the
agent trace so the evaluator can see the self-correction.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def provider_label() -> str:
    if PROVIDER == "groq":
        return f"Groq / {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"
    if PROVIDER == "gemini":
        return f"Google Gemini / {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}"
    if PROVIDER == "ollama":
        return f"Ollama (local) / {os.getenv('OLLAMA_MODEL', 'llama3.1')}"
    return PROVIDER


def check_credentials() -> str | None:
    """Return an error string if the chosen provider is not configured."""
    if PROVIDER == "groq" and not os.getenv("GROQ_API_KEY"):
        return "GROQ_API_KEY is missing. Get a free key at https://console.groq.com/keys and put it in .env"
    if PROVIDER == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey and put it in .env"
    if PROVIDER not in {"groq", "gemini", "ollama"}:
        return f"Unknown LLM_PROVIDER '{PROVIDER}'. Use groq, gemini or ollama."
    return None


def get_llm(temperature: float = 0.0, fallback: bool = False):
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        model = (
            os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
            if fallback
            else os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        )
        return ChatGroq(model=model, temperature=temperature, max_retries=1, timeout=60)

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = "gemini-2.5-flash-lite" if fallback else os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=1)

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def model_name(llm) -> str:
    for attr in ("model_name", "model"):
        if hasattr(llm, attr):
            return str(getattr(llm, attr))
    return type(llm).__name__


def invoke_with_fallback(messages: list, trace: list, node: str, tools: list | None = None, temperature: float = 0.0):
    """Call the primary model; on failure wait briefly, then switch to the fallback model.

    Every failure and switch is appended to `trace` so it shows up in the UI.
    """
    primary = get_llm(temperature)
    attempts = [("primary", primary), ("fallback", get_llm(temperature, fallback=True))]
    last_err: Exception | None = None
    for label, llm in attempts:
        runnable = llm.bind_tools(tools) if tools else llm
        try:
            t0 = time.time()
            resp = runnable.invoke(messages)
            trace.append(
                {
                    "node": node,
                    "type": "llm",
                    "title": f"LLM call ({label}: {model_name(llm)}) took {time.time() - t0:.1f}s",
                    "detail": (resp.content[:600] if isinstance(resp.content, str) else str(resp.content)[:600]),
                }
            )
            return resp
        except Exception as e:  # rate limit, timeout, auth...
            last_err = e
            trace.append(
                {
                    "node": node,
                    "type": "error",
                    "title": f"LLM error on {label} model {model_name(llm)}: {type(e).__name__}",
                    "detail": str(e)[:500] + ("\n-> Reflecting: switching to fallback model." if label == "primary" else ""),
                }
            )
            if label == "primary":
                time.sleep(2)
    raise RuntimeError(f"All LLM attempts failed: {last_err}")


_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.S)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM reply (tolerates ```json fences and prose)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON found in model output: {text[:200]}")
    return json.loads(m.group(1))
