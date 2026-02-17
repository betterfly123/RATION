#!/usr/bin/env bash

# QWEN
#model_name="pretrained=Qwen2.5-VL-7B-Instruct"

# 1) orign
# MODEL_ARGS="${model_name}"

# 2) Predefined Functional Layers
# MODEL_ARGS="${model_name}"

# 3) RATION
# MODEL_ARGS="${model_name}"

# 4) MAGIC_VQA
# MODEL_ARGS="pretrained=Qwen2.5-VL-7B-Instruct,magic_mode=on,magic_index=magic_index.faiss,magic_triplets=magic_triplets.jsonl,magic_k_keep=6,magic_tau=0.2,magic_ratios=0.3333:0.3333:0.3333" \

# 5) DCOT
# MODEL_ARGS="${model_name},ddcot_mode=on"

# 6) MMCOT
# MODEL_ARGS="${model_name},ddcot_mode=on"

# entropy_qwen_vl magic_qwen_vl ddcot_qwen_vl mmcot_qwen_vl

python -m lmms_eval \
  --model entropy_llava_next \
  --model_args "$MODEL_ARGS" \
  --tasks mme \
  --batch_size 1 \
