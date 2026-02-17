import torch
import torch.nn.functional as F
from contextlib import contextmanager
import json
import time

from .config import GEN_KWARGS, IMAGE_TOKEN_ID, HEAD_DIM
from .task_select import task_select
from .controller import KVEntropyControllerLite


class EntropyMixin:
    entropy_mode: str = "off"

    @contextmanager
    def set_entropy_mode(self, mode: str):
        old = getattr(self, "entropy_mode", "off")
        self.entropy_mode = mode
        try:
            yield self
        finally:
            self.entropy_mode = old

    def _attach_layer_atten_hooks(self, layer_names, task_direct, idxs, d_a, d_b):
        handles = []

        def _to_idx(x):
            if isinstance(x, int):
                return x
            return int(str(x).split("_")[-1])

        num_heads = self.config.text_config.num_attention_heads
        head_dim = HEAD_DIM
        num_kv_heads = self.config.text_config.num_key_value_heads

        for name in layer_names:
            idx = _to_idx(name)
            layer = self.model.language_model.layers[idx]
            attn = layer.self_attn

            controller = KVEntropyControllerLite(
                attn_module=attn,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                layer_key=f"layer_{idx}",
                direct=task_direct,
                idxs=idxs,
                d_a=d_a,
                d_b=d_b,
            )
            handle_k = attn.k_proj.register_forward_hook(controller.k_hook())
            handles.append(handle_k)

        return handles

    def _split_kwargs(self, kwargs):
        model_kwargs = {}
        gen_kwargs = {}
        for k, v in kwargs.items():
            if k in GEN_KWARGS:
                gen_kwargs[k] = v
            else:
                model_kwargs[k] = v
        return model_kwargs, gen_kwargs

    def _attach_layer_forward_hooks(self, idxs):

        delta_score_cache, handles = {}, []

        def make_fwd_hook(name):
            def fwd_hook(module, inputs, output):
                y_out = output[0] if isinstance(output, tuple) else output
                y_in  = inputs[0]

                delta = (y_out - y_in)
                vis_delta = delta[:, idxs[0]: idxs[-1] + 1, :]
                score = vis_delta.abs().mean(dim=(-2, -1))
                score = score.detach()

                delta_score_cache[name] = score
            return fwd_hook

        for i, layer in enumerate(self.model.language_model.layers):
            handles.append(layer.register_forward_hook(make_fwd_hook(f"layer_{i}")))


        return handles, {"delta_score": delta_score_cache}

    def _compute_layer_scores(self, cache, **forward_inputs):
        cache["delta_score"].clear()

        _ = self(**forward_inputs, use_cache=False, return_dict=True)

        return dict(cache["delta_score"])

    def _mode_select_layer(self, *args, model_kwargs, gen_kwargs):

        input_ids = model_kwargs["input_ids"]
        idxs = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        t0 = time.perf_counter()
        handles_fwd, cache = self._attach_layer_forward_hooks(idxs)
        t1 = time.perf_counter()
        try:
            layer_scores = self._compute_layer_scores(cache, **model_kwargs)
        finally:
            for h in handles_fwd:
                h.remove()
        t2 = time.perf_counter()
        layer_scores = {
            name: float(score.detach().cpu().item())
            for name, score in layer_scores.items()
        }
        
        sorted_items = sorted(layer_scores.items(), key=lambda kv: kv[1], reverse=True)

        sample_id = model_kwargs.pop("sample_id", None)
        self._save_sorted_items(sample_id, sorted_items)
        t3 = time.perf_counter()
        outputs = super().generate(*args, **model_kwargs, **gen_kwargs)
        t4 = time.perf_counter()

        return outputs

    def _save_sorted_items(self, sample_id, sorted_items):

        if sample_id is None:
            sample_id = "none"

        record = {
            "sample_id": sample_id,
            "sorted_items": sorted_items,
        }

        with open("qwen3_chartqa_layer_sorted_items.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _mode_entropy_edit(self, *args, model_kwargs, gen_kwargs):

        sample_id = model_kwargs.pop("sample_id", None)
        sorted_items = self._load_sorted_items(sample_id, model_kwargs)

        sorted_items = [(name, score) for name, score in sorted_items]
        
        task_direct, layer_result = task_select(sorted_items)
        
        input_ids = model_kwargs["input_ids"]

        idxs = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        d_a = float(getattr(self, "_entropy_d_a", 0.8))
        d_b = float(getattr(self, "_entropy_d_b", 1.2))
        handles_entropy = self._attach_layer_atten_hooks(layer_result, task_direct, idxs, d_a, d_b)

        try:
            outputs = super().generate(*args, **model_kwargs, **gen_kwargs)
        finally:
            for h in handles_entropy:
                h.remove()

        return outputs
    
    def _load_sorted_items(self, sample_id, model_kwargs):

        if sample_id is None:
            sample_id = "none"

        with open("qwen3_chartqa_layer_sorted_items.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("sample_id") == sample_id:
                    return obj["sorted_items"]

        return []

    def generate(self, *args, **kwargs):

        mode = kwargs.pop("entropy_mode", None)
        if mode is None:
            mode = getattr(self, "entropy_mode", "off")

        model_kwargs, gen_kwargs = self._split_kwargs(kwargs)

        if mode == "off":
            kwargs.pop("sample_id")
            return super().generate(*args, **kwargs)
        
        if mode == "select":
            return self._mode_select_layer(*args, model_kwargs=model_kwargs, gen_kwargs=gen_kwargs)

        if mode == "edit":
            return self._mode_entropy_edit(*args, model_kwargs=model_kwargs, gen_kwargs=gen_kwargs)

