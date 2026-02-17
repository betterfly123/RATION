# lmms_eval/mmcot/prompting.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MMCOTPromptConfig:

    use_question_only: bool = True
    end_reason_token: str = "<END_REASON>"
    stage1_bullets_min: int = 3
    stage1_bullets_max: int = 8

    def build_rationale_prompt(self, question: str) -> str:
        q = (question or "").strip()
        return f"""You must output ONLY the reasoning steps (no final answer).
Requirements:
1) Write {self.stage1_bullets_min}-{self.stage1_bullets_max} bullet points, one sentence per bullet.
2) Do NOT output the final answer.
3) The last line must be exactly: {self.end_reason_token}

Question:
{q}
"""

    def build_answer_prompt(self, question: str, rationale: str) -> str:
        q = (question or "").strip()
        r = (rationale or "").strip()
        return f"""You are given a question and reasoning steps. Decide the required answer format from the question and output ONLY the final answer.

Rules:
1) If the question is YES/NO, output exactly "Yes" or "No".
2) If the question asks to choose an option (e.g., A/B/C/D), output ONLY the option letter.
3) Otherwise, output a short free-form answer (one phrase or one sentence).
Do NOT include any explanation.

Question:
{q}

Reasoning:
{r}

Final Answer:"""
