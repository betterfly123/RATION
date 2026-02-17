# lmms_eval/models/magic_qwen2_vl.py
import os
import re
import torch
from transformers import AutoProcessor
from typing import List, Tuple, Optional, Dict, Any

from transformers import CLIPModel, CLIPProcessor
import numpy as np


from tqdm import tqdm

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model



from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from PIL import Image


from lmms_eval.magic_vqa.atomic_index import AtomicIndex
from lmms_eval.magic_vqa.postprocess import postprocess_triplets
from lmms_eval.magic_vqa.caption import QwenCaptioner
from lmms_eval.magic_vqa.prompt import augment_question


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


def _parse_ratios(ratios: str):

    if ":" in ratios:
        parts = ratios.split(":")
    elif "|" in ratios:
        parts = ratios.split("|")
    else:
        parts = ratios.split(",")

    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"magic_ratios must have 3 floats, got: {ratios}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))



class _HfCLIPEmbedder:


    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        clip_path: str = "/root/autodl-fs/Model/clip-vit-base-patch32",
        max_length: int = 77,
        use_autocast: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.clip_path = clip_path
        self.max_length = int(max_length)
        self.use_autocast = bool(use_autocast) and (device.type == "cuda")

        self.model = CLIPModel.from_pretrained(clip_path).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_path)


        with torch.no_grad():
            toks = self.processor(text=["hello"], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            feat = self.model.get_text_features(**toks).to(torch.float32)
        self.dim = int(feat.shape[-1])

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = self.model.get_text_features(**inputs)
        else:
            feat = self.model.get_text_features(**inputs)

        feat = feat.to(torch.float32)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat[0].detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(
            images=img,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = self.model.get_image_features(**inputs)
        else:
            feat = self.model.get_image_features(**inputs)

        feat = feat.to(torch.float32)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat[0].detach().cpu().numpy().astype("float32")


@register_model("magic_qwen2_vl")
class Magic_Qwen2VL(lmms):


    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        # ----- MAGIC configs -----
        magic_mode: str = "on",                 # "on" | "off"
        magic_index: Optional[str] = None,      # faiss.index
        magic_triplets: Optional[str] = None,   # triplets.jsonl
        magic_k_retrieve: int = 30,
        magic_k_keep: int = 6,
        magic_tau: float = 0.2,
        magic_ratios: str = "0.3333,0.3333,0.3333",  # (PE,EC,SI)
        # ----- CLIP configs -----
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        clip_max_long_side: int = 1024,
        # ----- Qwen image resize -----
        qwen_max_long_side: int = 2048,
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


        self.magic_mode = magic_mode
        self.magic_k_retrieve = int(magic_k_retrieve)
        self.magic_k_keep = int(magic_k_keep)
        self.magic_tau = float(magic_tau)
        self.magic_ratios = _parse_ratios(magic_ratios)
        self.qwen_max_long_side = int(qwen_max_long_side)
        self.clip_max_long_side = int(clip_max_long_side)


        self.captioner = QwenCaptioner(
            model=self.model,
            processor=self.processor,
            device=self.device,
            prompt="Describe the image in one short sentence.",
            do_sample=False,
        )


        self.atomic = AtomicIndex(magic_index, magic_triplets)

        self.clip = _HfCLIPEmbedder(
            device=self.device,
            dtype=torch_dtype,
            clip_path="/root/autodl-fs/Model/clip-vit-base-patch32",
            max_length=77,
            use_autocast=True,
        )

        self._magic_cache: Dict[str, Dict[str, Any]] = {}

    def _build_magic_ctx(self, sample_id: str, images: List[Image.Image], question: str) -> Dict[str, Any]:

        if sample_id in self._magic_cache:
            return self._magic_cache[sample_id]

        assert self.atomic is not None and self.clip is not None


        caption = self.captioner.caption(images, max_len=64)

        img0 = resize_keep_ratio(images[0], max_long_side=self.clip_max_long_side) if images else None
        cand: List[Dict[str, Any]] = []

        if img0 is not None:
            i_emb = self.clip.encode_image(img0)
            cand += self.atomic.search(i_emb, topk=self.magic_k_retrieve, src="image")

        q_emb = self.clip.encode_text(question)
        cand += self.atomic.search(q_emb, topk=self.magic_k_retrieve, src="question")

        if caption:
            c_emb = self.clip.encode_text(caption)
            cand += self.atomic.search(c_emb, topk=self.magic_k_retrieve, src="caption")

        kept = postprocess_triplets(
            cand,
            k_keep=self.magic_k_keep,
            tau=self.magic_tau,
            ratios=self.magic_ratios,
        )

        magic_ctx = {"triplets": kept, "confidence": None}
        self._magic_cache[sample_id] = magic_ctx
        return magic_ctx

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs: List[str] = []

        pbar = tqdm(
            total=len(requests),
            desc="MAGIC Qwen2-VL Responding",
        )

        for inst in requests:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = inst.args

            sample_id = f"{task}|{split}|{doc_id}"
            doc = self.task_dict[task][split][doc_id]
            images_raw = doc_to_visual(doc)
            images = [_ensure_pil(x) for x in images_raw]

            if self.magic_mode == "on":
                magic_ctx = self._build_magic_ctx(sample_id, images, contexts)
                contexts_aug = augment_question(contexts, magic_ctx)
            else:
                contexts_aug = contexts

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image",
                                "image": resize_keep_ratio(img, max_long_side=self.qwen_max_long_side),
                            }
                            for img in images
                        ],
                        {"type": "text", "text": contexts_aug},
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
                "max_new_tokens": all_gen_kwargs.get("max_gen_toks", 16),
                "do_sample": False,
                "output_scores": False,
                "return_dict_in_generate": False,
            }

            
            outputs_ids = self.model.generate(
                **inputs,
                **gen_kwargs
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
        raise NotImplementedError("Loglikelihood is not implemented for MAGIC Qwen2-VL")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
