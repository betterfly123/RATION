# lmms_eval/models/ddcot_qwen_vl.py
import re
import torch
from typing import List, Tuple, Optional, Dict, Any

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

from lmms_eval.DDCoT.pipeline import DDCoTPipeline, DDCoTConfig


def resize_keep_ratio(img: Image.Image, max_long_side: int = 2048) -> Image.Image:
    w, h = img.size
    scale = min(max_long_side / w, max_long_side / h, 1.0)
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _ensure_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(x)}")


def _strip_choices_keep_stem(s: str) -> str:


    s = (s or "").strip()
    s = re.sub(r"\n?\s*(Answer\s*:|答案\s*:)\s*$", "", s, flags=re.I).strip()


    m = re.search(r"(?m)^\s*[A-F][\.\)]\s+", s)
    if m:
        return s[:m.start()].strip()

    m = re.search(r"\b[A-F][\.\)]\s+", s)
    if m and m.start() > 0:
        return s[:m.start()].strip()

    return s


@register_model("ddcot_qwen_vl")
class DDCoT_QwenVL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        # ---- DDCoT configs ----
        ddcot_mode: str = "on",                 
        ddcot_strip_choices: str = "on", 
        ddcot_max_subq: int = 10,
        ddcot_temp_split: float = 0.2,
        ddcot_temp_neg: float = 0.0,
        ddcot_temp_vqa: float = 0.0,
        ddcot_temp_joint: float = 0.2,
        ddcot_tokens_split: int = 256,
        ddcot_tokens_neg: int = 512,
        ddcot_tokens_vqa: int = 128,
        ddcot_tokens_joint: int = 256,
        # ---- image resize ----
        qwen_max_long_side: int = 2048,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(device)
        torch_dtype = getattr(torch, dtype)

        if "Qwen2" in pretrained or "Qwen2.5" in pretrained:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained, torch_dtype=torch_dtype, device_map=None
            ).to(self.device)
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained, torch_dtype=torch_dtype, device_map=None
            ).to(self.device)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.ddcot_mode = ddcot_mode
        self.ddcot_strip_choices = (ddcot_strip_choices == "on")
        self.qwen_max_long_side = int(qwen_max_long_side)

        cfg = DDCoTConfig(
            max_subq=int(ddcot_max_subq),
            temp_split=float(ddcot_temp_split),
            temp_neg=float(ddcot_temp_neg),
            temp_vqa=float(ddcot_temp_vqa),
            temp_joint=float(ddcot_temp_joint),
            max_new_tokens_split=int(ddcot_tokens_split),
            max_new_tokens_neg=int(ddcot_tokens_neg),
            max_new_tokens_vqa=int(ddcot_tokens_vqa),
            max_new_tokens_joint=int(ddcot_tokens_joint),
        )
        self.ddcot = DDCoTPipeline(generate_fn=self._qwen_generate, cfg=cfg)

        self._ddcot_cache: Dict[str, Dict[str, Any]] = {}

    @torch.no_grad()
    def _qwen_generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(temperature > 0),
            "output_scores": False,
            "return_dict_in_generate": False,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)

        outputs_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], outputs_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs: List[str] = []
        pbar = tqdm(total=len(requests), desc="DDCoT Qwen-VL Responding")

        for inst in requests:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = inst.args
            sample_id = f"{task}|{split}|{doc_id}"

            if sample_id in self._ddcot_cache:
                outputs.append(self._ddcot_cache[sample_id]["text"])
                pbar.update(1)
                continue

            doc = self.task_dict[task][split][doc_id]
            images_raw = doc_to_visual(doc)
            images = [_ensure_pil(x) for x in images_raw]
            images = [resize_keep_ratio(img, max_long_side=self.qwen_max_long_side) for img in images]


            if self.ddcot_mode != "on":
                messages = [{
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": contexts},
                    ],
                }]
                text = self._qwen_generate(messages, max_new_tokens=all_gen_kwargs.get("max_gen_toks", 16), temperature=0.0)
                outputs.append(text)
                self._ddcot_cache[sample_id] = {"text": text}
                pbar.update(1)
                continue


            question = _strip_choices_keep_stem(contexts) if self.ddcot_strip_choices else (contexts or "").strip()

            res = self.ddcot.run(images=images, question=question)
            text = (res.get("final_answer") or "").strip()
            outputs.append(text)
            self._ddcot_cache[sample_id] = {"text": text, "detail": res}
            pbar.update(1)

        pbar.close()
        return outputs

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for DDCoT Qwen-VL")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
