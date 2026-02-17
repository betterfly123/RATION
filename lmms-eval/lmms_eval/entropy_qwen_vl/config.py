

layer_to_task = {
    "recogn": [],
    "counting":  [],
    "grounding": [],
    "ocr":       [],
}

task_direct_map = {
    "recogn":   0,
    "counting": 1,
    "grounding":0,
    "ocr":      0,
}


GEN_KWARGS = {
    "max_length", "max_new_tokens", "min_length",
    "do_sample", "temperature", "top_k", "top_p",
    "num_beams", "num_return_sequences",
    "repetition_penalty", "length_penalty",
    "output_scores", "return_dict_in_generate",
    "output_logits", "pad_token_id", "eos_token_id",
    "bos_token_id"
}

IMAGE_TOKEN_ID = 151655
HEAD_DIM = 128
