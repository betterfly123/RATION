# lmms_eval/models/entropy_qwen2_vl.py
import re
import torch
from transformers import AutoProcessor
from typing import List, Optional, Tuple, Union

from tqdm import tqdm 

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

from lmms_eval.entropy_qwen_vl import EntropyQwen3VL, EntropyQwen2_5VL  

from PIL import Image

def resize_keep_ratio(img: Image.Image, max_long_side: int = 1024) -> Image.Image:

    w, h = img.size
    scale = min(max_long_side / w, max_long_side / h, 1.0) 
    if scale >= 1.0:
        return img 

    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


@register_model("entropy_qwen2_vl")
class Entropy_Qwen2VL(lmms):


    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        entropy_mode: str = "off",
        d_a: float = 1.0,
        d_b: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(device)
        torch_dtype = getattr(torch, dtype)


        if "Qwen2" in pretrained:
            self.model = EntropyQwen2_5VL.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                device_map=None,
            ).to(self.device)
        else:
            self.model = EntropyQwen3VL.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                device_map=None,
            ).to(self.device)
        self.model.eval()


        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.entropy_mode = entropy_mode
        self.model._entropy_d_a = float(d_a)
        self.model._entropy_d_b = float(d_b)


    def generate_until(self, requests: List[Instance]) -> List[str]:

        outputs: List[str] = []

        pbar = tqdm(
            total=len(requests),
            desc="Entropy Qwen2-VL Responding",
        )
        kkid = 0
        for inst in requests:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = inst.args


            sample_id = f"{task}|{split}|{doc_id}"
            doc = self.task_dict[task][split][doc_id]

            images = doc_to_visual(doc)
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": resize_keep_ratio(img, max_long_side=2048),}
                            for img in images
                        ],
                        {"type": "text", "text": contexts},
                    ],
                }
            ]

            # processor 
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)


            gen_kwargs = {
                "max_new_tokens": all_gen_kwargs.get("max_gen_toks", 16),
                "do_sample": False,
                "output_scores": False,
                "return_dict_in_generate": False,
            }

            outputs_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
                entropy_mode=self.entropy_mode,
                sample_id=sample_id,
            )

            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], outputs_ids)
            ]
            text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            outputs.append(text)

            pbar.update(1)

        pbar.close()

        return outputs

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def generate_until_multi_round(self, requests) -> List[str]:

        return self.generate_until(requests)
