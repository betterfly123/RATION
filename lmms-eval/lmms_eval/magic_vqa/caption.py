from __future__ import annotations

from typing import Any, List, Optional, Sequence
import torch


class CaptionerBase:


    def caption(self, images: List[Any], *, max_len: int = 64) -> str:
        raise NotImplementedError


class DummyCaptioner(CaptionerBase):


    def caption(self, images: List[Any], *, max_len: int = 64) -> str:
        return ""


class PlaceholderCaptioner(CaptionerBase):


    def __init__(self, backend: str = "qwen"):
        self.backend = backend

    def caption(self, images: List[Any], *, max_len: int = 64) -> str:
        raise RuntimeError(
            f"{self.backend} captioner not implemented yet. "
            "Use DummyCaptioner or implement this class."
        )


class QwenCaptioner(CaptionerBase):


    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: torch.device,
        prompt: str = "Describe the image in one short sentence.",
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.processor = processor
        self.device = device

        self.prompt = prompt
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def caption(self, images: List[Any], *, max_len: int = 64) -> str:
        if not images:
            return ""

        img0 = images[0]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img0},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": int(max_len),
            "do_sample": bool(self.do_sample),
            "temperature": float(self.temperature) if self.do_sample else None,
            "top_p": float(self.top_p) if self.do_sample else None,
            "output_scores": False,
            "return_dict_in_generate": False,
        }

        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        out_ids = self.model.generate(**inputs, **gen_kwargs)


        trimmed = [
            o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


        text = text.split("\n")[0].strip()
        return text
