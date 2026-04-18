from __future__ import annotations
import json
import os
from typing import Any
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found")
print("api loaded")

# -----------------------
# MODEL
# -----------------------
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",  # stronger than 8B
    temperature=0.2,
)


MAX_PAPERS = 30
TOP_K = 5
SUMMARY_LIMIT = 500


SYSTEM_PROMPT = "You are an elite AI research analyst."

USER_PROMPT_TEMPLATE = """You are given a list of AI research papers.

Your job is to identify ONLY the most important AI papers from the last 24 hours.

STRICT RULES:

ONLY include papers that are clearly impactful in:
- Large Language Models (LLMs)
- AI agents / autonomous systems
- Multimodal models (vision-language)
- Diffusion / generative models
- Alignment / reasoning

AVOID completely:
- Generic ML optimization
- Finance / domain-specific ML
- Hardware-only work
- Minor incremental papers

Be extremely selective.

If fewer than 5 papers meet criteria, return fewer.
Return ONLY valid JSON. No truncation.
Return STRICT JSON:

[
  {{
    "id": "<paper_id>",
    "reason": "<1 line why important>"
  }}
]

PAPERS:
{papers_block}
"""


def _fallback(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return papers[:TOP_K]


def _truncate_summary(summary: Any, limit: int = SUMMARY_LIMIT) -> str:
    text = str(summary or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_papers_block(papers: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper.get("id") or "").strip()
        title = str(paper.get("title") or "").strip()
        summary = _truncate_summary(paper.get("summary"))

        lines.append(
            f'{index}. [ID: {paper_id}] Title: {title}\n   Summary: {summary}'
        )

    return "\n\n".join(lines)


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
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
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def _parse_ranked_ids(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        json_block = _extract_json_array(content)
        if not json_block:
            return []
        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError:
            return []

    if not isinstance(payload, list):
        return []

    ranked_ids: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id") or "").strip()
        if paper_id:
            ranked_ids.append(paper_id)

    return ranked_ids


def rank_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not papers:
        return []

    candidates = papers[:MAX_PAPERS]
    fallback = _fallback(candidates)


    prompt = USER_PROMPT_TEMPLATE.format(
        papers_block=_build_papers_block(candidates)
    )

    print("CALLING GROQ (ChatGroq)...")

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        content = response.content
        print("RAW:", content[:300])
    except Exception as e:
        print("ERROR:", e)
        return fallback

    ranked_ids = _parse_ranked_ids(content)
    if not ranked_ids:
        return fallback

    paper_by_id = {
        str(paper.get("id") or "").strip(): paper
        for paper in candidates
        if str(paper.get("id") or "").strip()
    }

    ranked_papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for paper_id in ranked_ids:
        if paper_id in seen_ids:
            continue
        paper = paper_by_id.get(paper_id)
        if paper is None:
            continue
        ranked_papers.append(paper)
        seen_ids.add(paper_id)
        if len(ranked_papers) == TOP_K:
            return ranked_papers

    for paper in fallback:
        paper_id = str(paper.get("id") or "").strip()
        if paper_id and paper_id in seen_ids:
            continue
        ranked_papers.append(paper)
        if paper_id:
            seen_ids.add(paper_id)
        if len(ranked_papers) == TOP_K:
            break

    return ranked_papers
