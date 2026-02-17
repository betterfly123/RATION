from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_magic_prompt(
    *,
    question: str,
    triplets: List[Dict[str, Any]],
    confidence: Optional[float] = None,
    include_scores: bool = False,
) -> str:

    lines: List[str] = []
    lines.append("[Commonsense Knowledge]")
    if not triplets:
        lines.append("- (None)")
    else:
        for t in triplets:
            rel = t.get("relevance", "Medium")
            src = t.get("src", "unknown")
            text = t.get("text", "")
            if include_scores:
                score = float(t.get("score", 0.0))
                lines.append(f"- ({rel}, from {src}, score={score:.4f}) {text}")
            else:
                lines.append(f"- ({rel}, from {src}) {text}")

    if confidence is not None:
        lines.append(f"[Commonsense Confidence] {float(confidence):.4f}")

    lines.append("[Question]")
    lines.append(question)

    return "\n".join(lines)


def augment_question(question: str, magic_ctx: Dict[str, Any]) -> str:

    return format_magic_prompt(
        question=question,
        triplets=magic_ctx.get("triplets", []) or [],
        confidence=magic_ctx.get("confidence", None),
    )
