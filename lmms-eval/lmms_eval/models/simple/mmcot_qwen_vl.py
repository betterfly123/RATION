# lmms_eval/models/simple/mmcot_qwen_vl.py

import torch
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

from lmms_eval.mmcot.runner import TwoPassGenConfig, two_pass_generate


def _ensure_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(x)}")


def resize_keep_ratio(img: Image.Image, max_long_side: int = 2048) -> Image.Image:
    w, h = img.size
    scale = min(max_long_side / w, max_long_side / h, 1.0)
    if scale >= 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


@register_model("mmcot_qwen_vl")
class MMCOTQwenVL(lmms):


    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        # ----- MMCOT configs -----
        rationale_max_new_tokens: int = 256,
        answer_max_new_tokens: int = 64,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(device)
        torch_dtype = getattr(torch, dtype)


        if "Qwen2" in pretrained:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                device_map=None,
            ).to(self.device)
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                device_map=None,
            ).to(self.device)

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.gen_cfg = TwoPassGenConfig(
            rationale_max_new_tokens=int(rationale_max_new_tokens),
            answer_max_new_tokens=int(answer_max_new_tokens),
        )


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:

        raise NotImplementedError("loglikelihood is not implemented for MMCOTQwenVL")

    @torch.no_grad()
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:


        outputs: List[str] = []

        pbar = tqdm(
            total=len(requests),
            desc="MMCOT Qwen-VL Responding",
        )

        for inst in requests:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = inst.args


            doc = self.task_dict[task][split][doc_id]


            images_raw = doc_to_visual(doc)
            images = [_ensure_pil(x) for x in images_raw]


            question = contexts


            out = two_pass_generate(
                model=self.model,
                processor=self.processor,
                images=images,
                question=question,
                gen_cfg=self.gen_cfg,
                device=self.device,
            )

            outputs.append(out["answer"])
            pbar.update(1)

        pbar.close()
        return outputs

 
    def generate_until(self, requests: List[Instance]) -> List[str]:
        return self.generate_until_multi_round(requests)
