# lmms_eval/mmcot/postprocess.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReasoningExtractResult:
    reasoning: str
    leaked: bool



ANSWER_MARKERS = [
    r"\n\s*(final\s*answer|answer)\s*[:]",
    r"(therefore|thus|so)\s+the\s+answer\s+is",
]


def extract_pure_reasoning(
    text: str,
    end_reason_token: str = "<END_REASON>",
    answer_str: Optional[str] = None,
) -> ReasoningExtractResult:
    t = (text or "").strip()


    if end_reason_token in t:
        t = t.split(end_reason_token, 1)[0].strip()


    for pat in ANSWER_MARKERS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            t = t[: m.start()].strip()


    lines = [ln.strip("-•* \t") for ln in t.splitlines() if ln.strip()]
    reasoning = "\n".join(f"- {ln}" for ln in lines)

    leaked = False
    if answer_str and answer_str in reasoning:
        leaked = True

    if len(reasoning) < 10:
        leaked = True

    return ReasoningExtractResult(reasoning=reasoning, leaked=leaked)
