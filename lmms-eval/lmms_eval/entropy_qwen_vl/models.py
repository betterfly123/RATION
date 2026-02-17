from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from .mixin import EntropyMixin

class EntropyQwen3VL(EntropyMixin, Qwen3VLForConditionalGeneration):
    pass

class EntropyQwen2_5VL(EntropyMixin, Qwen2_5_VLForConditionalGeneration):
    pass
