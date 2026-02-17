# lmms_eval/DDCoT/pipeline.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from .prompts import SPLIT_PROMPT, NEGSPACE_PROMPT, VQA_PROMPT, JOINT_PROMPT
from .parse_utils import extract_json


@dataclass
class DDCoTConfig:
    max_subq: int = 10
    temp_split: float = 0.2
    temp_neg: float = 0.0
    temp_vqa: float = 0.0
    temp_joint: float = 0.2
    max_new_tokens_split: int = 256
    max_new_tokens_neg: int = 512
    max_new_tokens_vqa: int = 128
    max_new_tokens_joint: int = 256


class DDCoTPipeline:

    def __init__(self, generate_fn: Callable[..., str], cfg: Optional[DDCoTConfig] = None):
        self.generate_fn = generate_fn
        self.cfg = cfg or DDCoTConfig()

    def run(self, images: List[Any], question: str) -> Dict[str, Any]:

        split_msgs = [
            {"role": "system", "content": [{"type": "text", "text": SPLIT_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": f"Question: {question}"}]},
        ]
        split_out = self.generate_fn(
            split_msgs,
            max_new_tokens=self.cfg.max_new_tokens_split,
            temperature=self.cfg.temp_split,
        )
        split_json = extract_json(split_out) or {}
        subqs = split_json.get("sub_questions", [])
        if not isinstance(subqs, list) or len(subqs) == 0:
            subqs = [question]
        subqs = subqs[: self.cfg.max_subq]


        neg_msgs = [
            {"role": "system", "content": [{"type": "text", "text": NEGSPACE_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "Sub-questions:\n" + "\n".join([f"- {q}" for q in subqs])}]},
        ]
        neg_out = self.generate_fn(
            neg_msgs,
            max_new_tokens=self.cfg.max_new_tokens_neg,
            temperature=self.cfg.temp_neg,
        )
        neg_json = extract_json(neg_out) or {}
        ans_list = neg_json.get("answers", [])

        qa: Dict[str, str] = {}
        if isinstance(ans_list, list):
            for it in ans_list:
                if isinstance(it, dict) and "q" in it and "a" in it:
                    qa[str(it["q"])] = str(it["a"])

        for q in subqs:
            qa.setdefault(q, "Uncertain")

        for q in list(qa.keys()):
            if qa[q].strip() == "Uncertain":
                vqa_msgs = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": VQA_PROMPT.format(q=q)},
                        ],
                    }
                ]
                vqa_out = self.generate_fn(
                    vqa_msgs,
                    max_new_tokens=self.cfg.max_new_tokens_vqa,
                    temperature=self.cfg.temp_vqa,
                )
                qa[q] = (vqa_out or "").strip()

        evidence = "\n\n".join([f"Q: {q}\nA: {qa[q]}" for q in subqs])
        joint_msgs = [
            {"role": "system", "content": [{"type": "text", "text": JOINT_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Original Question: {question}\n\nEvidence:\n{evidence}"
            }]},
        ]
        joint_out = self.generate_fn(
            joint_msgs,
            max_new_tokens=self.cfg.max_new_tokens_joint,
            temperature=self.cfg.temp_joint,
        )
        joint_json = extract_json(joint_out) or {}
        final_answer = (joint_json.get("final_answer") or "").strip()
        if not final_answer:
            final_answer = (joint_out or "").strip()

        return {
            "final_answer": final_answer,
            "rationale": joint_json.get("rationale", ""),
            "sub_questions": subqs,
            "sub_qa": qa,
            "raw": {"split": split_out, "neg": neg_out, "joint": joint_out},
        }
