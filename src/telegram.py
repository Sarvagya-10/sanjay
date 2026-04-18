from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# -----------------------
# ENV
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found")

if not CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID not found")


# -----------------------
# FORMAT MESSAGE
# -----------------------
def format_message(summaries: list[dict[str, Any]]) -> str:
    parts = []
    for i, s in enumerate(summaries, 1):
        innovations = (
            "\n- ".join(s["key_innovations"])
            if s["key_innovations"]
            else "N/A"
        )

        block = f"""
*{i}. {s['title']}*

*Core Idea:*  
{s['core_idea']}

*Problem:*  
{s['problem']}

*Key Innovations:*  
- {innovations}

*Impact:*  
{s['impact']}

*Limitations:*  
{s['limitations']}

🔗 {s['link']}
"""
        parts.append(block.strip())

    return "\n\n".join(parts)


# -----------------------
# SEND MESSAGE
# -----------------------
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    res = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )

    if res.status_code != 200:
        print("Telegram error:", res.text)
    else:
        print("Message sent successfully")


# -----------------------
# MAIN ENTRY
# -----------------------
from datetime import datetime


def deliver(summaries: list[dict[str, Any]]):
    if not summaries:
        print("No summaries to send")
        return

    today = datetime.now().strftime("%d %B %Y")
    send_telegram(f"🧠 Top AI Papers — {today}")

    for s in summaries:
        single_message = format_message([s])
        send_telegram(single_message)