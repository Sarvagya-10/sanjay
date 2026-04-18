from __future__ import annotations

import html
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser


ARXIV_RSS_FEEDS = [
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.CL",
    "https://arxiv.org/rss/stat.ML",
]

TOP_ORGS = [
    "openai",
    "deepmind",
    "google",
    "meta",
    "anthropic",
    "microsoft",
    "nvidia",
    "stanford",
    "mit",
    "berkeley",
    "cmu",
]

TOP_RESEARCHERS = [
    "geoffrey hinton",
    "yann lecun",
    "ilya sutskever",
    "andrej karpathy",
    "dario amodei",
    "demis hassabis",
    "fei-fei li",
]

TOP_VENUES = [
    "neurips",
    "icml",
    "iclr",
    "cvpr",
    "acl",
]

HIGH_IMPACT_KEYWORDS = [
    "llm",
    "large language model",
    "agent",
    "autonomous agent",
    "multimodal",
    "vision-language",
    "diffusion",
    "text-to-image",
    "reasoning",
    "alignment",
    "foundation model",
    "scaling",
]

NEGATIVE_KEYWORDS = [
    "portfolio",
    "finance",
    "medical",
    "survey",
    "review",
    "case study",
]

ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

Paper = dict[str, Any]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_arxiv_id(value: Any) -> str:
    if not value:
        return ""

    match = ARXIV_ID_PATTERN.search(str(value))
    if not match:
        return ""

    arxiv_id = match.group(1)
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    if "v" in arxiv_id:
        base, _, version = arxiv_id.rpartition("v")
        if version.isdigit():
            return base
    return arxiv_id


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def _extract_authors(entry: Any) -> list[str]:
    authors = getattr(entry, "authors", None)
    if authors:
        extracted = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
            else:
                name = getattr(author, "name", None) or str(author)
            cleaned = _clean_text(name)
            if cleaned:
                extracted.append(cleaned)
        if extracted:
            return extracted

    author = _clean_text(getattr(entry, "author", ""))
    if not author:
        return []

    return [part.strip() for part in author.split(",") if part.strip()]


def _extract_tags(entry: Any) -> list[str]:
    tags = getattr(entry, "tags", None) or []
    extracted: list[str] = []

    for tag in tags:
        if isinstance(tag, dict):
            term = tag.get("term")
        else:
            term = getattr(tag, "term", None) or str(tag)
        cleaned = _clean_text(term)
        if cleaned:
            extracted.append(cleaned)

    return extracted


def _to_output_paper(paper: Paper) -> Paper:
    return {
        "id": str(paper.get("id") or "").strip(),
        "title": _clean_text(paper.get("title")),
        "summary": _clean_text(paper.get("summary")),
        "link": str(paper.get("link") or "").strip(),
        "published": paper.get("published"),
        "source": "arxiv",
    }


def normalize_date(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, time.struct_time):
        return datetime(*value[:6], tzinfo=timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            parsed = None

        if parsed is None:
            iso_value = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(iso_value)
            except ValueError:
                return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    return None


def fetch_arxiv_rss() -> list[Paper]:
    papers: list[Paper] = []

    for feed_url in ARXIV_RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in getattr(feed, "entries", []):
            link = getattr(entry, "link", "") or ""
            summary = (
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or ""
            )
            published = normalize_date(
                getattr(entry, "published", None)
                or getattr(entry, "updated", None)
                or getattr(entry, "published_parsed", None)
                or getattr(entry, "updated_parsed", None)
            )

            papers.append(
                {
                    "id": _extract_arxiv_id(link),
                    "title": _clean_text(getattr(entry, "title", "")),
                    "summary": _clean_text(summary),
                    "link": link,
                    "published": published,
                    "source": "arxiv",
                    "authors": _extract_authors(entry),
                    "tags": _extract_tags(entry),
                }
            )

    return papers


def filter_last_24h(papers: list[Paper]) -> list[Paper]:
    now = datetime.now(timezone.utc)
    window = timedelta(hours=24)

    filtered: list[Paper] = []
    for paper in papers:
        published = paper.get("published")
        if not isinstance(published, datetime):
            continue
        age = now - published
        if timedelta(0) <= age <= window:
            filtered.append(paper)

    return filtered


def deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[Paper] = []

    for paper in papers:
        paper_id = str(paper.get("id") or "").strip()
        normalized_title = _normalize_title(str(paper.get("title") or ""))

        if paper_id and paper_id in seen_ids:
            continue
        if normalized_title and normalized_title in seen_titles:
            continue

        if paper_id:
            seen_ids.add(paper_id)
        if normalized_title:
            seen_titles.add(normalized_title)
        deduped.append(paper)

    return deduped


def is_high_signal(paper: Paper) -> bool:
    authors = paper.get("authors") or []
    text = (
        " ".join(str(author) for author in authors) + " " +
        str(paper.get("title") or "") + " " +
        str(paper.get("summary") or "")
    ).lower()

    score = 0

    if any(org in text for org in TOP_ORGS):
        score += 5

    if any(researcher in text for researcher in TOP_RESEARCHERS):
        score += 5

    if any(venue in text for venue in TOP_VENUES):
        score += 4

    keyword_hits = sum(keyword in text for keyword in HIGH_IMPACT_KEYWORDS)
    if keyword_hits == 0:
        return False
    score += 2 * keyword_hits

    if any(keyword in text for keyword in NEGATIVE_KEYWORDS):
        score -= 5

    return score >= 4


def ingest() -> list[Paper]:
    arxiv = fetch_arxiv_rss()
    print(f"Fetched {len(arxiv)} from arxiv")

    papers = filter_last_24h(arxiv)
    papers = deduplicate(papers)

    filtered = [_to_output_paper(paper) for paper in papers if is_high_signal(paper)]
    print(f"After high-signal filter: {len(filtered)}")

    return filtered
