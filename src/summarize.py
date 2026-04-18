from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

SYSTEM_PROMPT = "You are an elite AI research analyst."

USER_PROMPT_TEMPLATE = """For the following AI research paper, generate a structured summary.

Return STRICT JSON:

{{
  "core_idea": "...",
  "problem": "...",
  "key_innovations": ["...", "..."],
  "impact": "...",
  "limitations": "..."
}}

Rules:
- Core idea: 2-3 lines
- Keep everything concise
- No fluff
- No extra text outside JSON

Paper:
Title: {title}
Abstract: {summary}
"""


def _fallback_summary(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(paper.get("title") or ""),
        "core_idea": str(paper.get("summary") or "")[:200],
        "problem": "",
        "key_innovations": [],
        "impact": "",
        "limitations": "",
        "link": str(paper.get("link") or ""),
    }


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def _parse_summary_payload(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        json_block = _extract_json_object(content)
        if not json_block:
            return None
        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None

    key_innovations = payload.get("key_innovations")
    if isinstance(key_innovations, list):
        innovations = [str(item).strip() for item in key_innovations if str(item).strip()]
    else:
        innovations = []

    return {
        "core_idea": str(payload.get("core_idea") or "").strip(),
        "problem": str(payload.get("problem") or "").strip(),
        "key_innovations": innovations,
        "impact": str(payload.get("impact") or "").strip(),
        "limitations": str(payload.get("limitations") or "").strip(),
    }


def _get_llm() -> ChatGroq | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        return ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
    except Exception:
        return None


def summarize_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not papers:
        return []

    llm = _get_llm()
    results: list[dict[str, Any]] = []

    for paper in papers:
        fallback = _fallback_summary(paper)
        print("Summarizing:", paper.get("title"))

        if llm is None:
            results.append(fallback)
            continue

        prompt = USER_PROMPT_TEMPLATE.format(
            title=str(paper.get("title") or "").strip(),
            summary=str(paper.get("summary") or "").strip(),
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            content = str(getattr(response, "content", "") or "")
        except Exception:
            results.append(fallback)
            continue

        parsed = _parse_summary_payload(content)
        if parsed is None:
            results.append(fallback)
            continue

        results.append(
            {
                "title": str(paper.get("title") or ""),
                "core_idea": parsed["core_idea"],
                "problem": parsed["problem"],
                "key_innovations": parsed["key_innovations"],
                "impact": parsed["impact"],
                "limitations": parsed["limitations"],
                "link": str(paper.get("link") or ""),
            }
        )

    return results