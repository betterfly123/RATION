# lmms_eval/mmcot/runner.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
import torch
from PIL import Image
from lmms_eval.mmcot.prompting import MMCOTPromptConfig


def resize_keep_ratio(img: Image.Image, max_long_side: int = 1024) -> Image.Image:

    w, h = img.size
    scale = min(max_long_side / w, max_long_side / h, 1.0)
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


@dataclass
class TwoPassGenConfig:


    prompt_cfg: MMCOTPromptConfig = field(default_factory=MMCOTPromptConfig)

    # Stage 1: rationale generation
    rationale_max_new_tokens: int = 64
    rationale_temperature: float = 0.7
    rationale_top_p: float = 0.9

    # Stage 2: answer generation
    answer_max_new_tokens: int = 12
    answer_temperature: float = 0.0
    answer_top_p: float = 1.0

def _trim_and_decode(processor, inputs, outputs_ids) -> str:

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], outputs_ids)
    ]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text

def _build_qwen_inputs(processor, device: torch.device, images: List[Image.Image], text: str):

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {
                                "type": "image",
                                "image": resize_keep_ratio(img, max_long_side=2048),
                    }
                    for img in images
                ],
                {"type": "text", "text": text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    return inputs

def two_pass_generate(
    model,
    processor,
    images,
    question: str,
    gen_cfg: TwoPassGenConfig,
    device: torch.device,
) -> Dict[str, Any]:


    # ---------- Stage 1: Rationale ----------

    rationale_prompt = gen_cfg.prompt_cfg.build_rationale_prompt(question)
    inputs1 = _build_qwen_inputs(processor, device, images, rationale_prompt)

    gen1_kwargs = {
        "max_new_tokens": gen_cfg.rationale_max_new_tokens,
        "do_sample": False,
        "output_scores": False,
        "return_dict_in_generate": False,
    }
    out1_ids = model.generate(**inputs1, **gen1_kwargs)
    rationale_text = _trim_and_decode(processor, inputs1, out1_ids).strip()


    # ---------- Stage 2: Answer ----------
    answer_prompt = gen_cfg.prompt_cfg.build_answer_prompt(
        question=question,
        rationale=rationale_text,
    )

    inputs_stage2 = _build_qwen_inputs(processor, device, images, answer_prompt)

    gen2_kwargs = {
        "max_new_tokens": gen_cfg.rationale_max_new_tokens,
        "do_sample": False,
        "output_scores": False,
        "return_dict_in_generate": False,
    }
    out2_ids = model.generate(**inputs_stage2, **gen2_kwargs)
    answer_text = _trim_and_decode(processor, inputs_stage2, out2_ids).strip()

    return {
        "rationale": rationale_text,
        "answer": answer_text,
    }
