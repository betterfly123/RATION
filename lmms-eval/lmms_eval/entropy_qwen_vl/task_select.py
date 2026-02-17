from .config import layer_to_task, task_direct_map
import re


def layer_id(name: str) -> int:
    return int(re.search(r"layer_(\d+)", name).group(1))

def rank_deltas(layers_ranked: list[str]) -> list[tuple[str, int]]:
    ids = [layer_id(x) for x in layers_ranked]
    max_id = max(ids)
    out = []
    for new_pos, name in enumerate(layers_ranked):
        lid = layer_id(name)
        orig_pos = max_id - lid
        delta = orig_pos - new_pos
        out.append((name, delta))
    return out


def task_select(layer_results):
    layers_nn = [name for name, _ in layer_results]
    layer_rank = rank_deltas(layers_nn)
    max_pos = {}
    for layer_name, score in layer_rank:
        max_pos[layer_name] = float(score)

    def is_ocr() -> bool:
        return (
            (max_pos.get("layer_13", 0) >= 2) and
            (
                (max_pos.get("layer_19", 0) >= 1) or
                (max_pos.get("layer_23", 0) >= 1) or
                (max_pos.get("layer_24", 0) >= 1)
            )
        )

    def is_position() -> bool:
        return (
            (max_pos.get("layer_12", 0) >= 4) and
            ((max_pos.get("layer_11", 0) >= 2) or (max_pos.get("layer_10", 0) >= 3)) and
            (max_pos.get("layer_24", 0) >= 1) and
            ((max_pos.get("layer_24", 0) >= 2) or (max_pos.get("layer_23", 0) >= 1))
        )

    def is_counting() -> bool:

        return (
            (
                (max_pos.get("layer_10", 0) >= 6) or
                (max_pos.get("layer_12", 0) >= 6) or
                (max_pos.get("layer_11", 0) >= 5) or
                (max_pos.get("layer_10", 0) >= 5 and max_pos.get("layer_12", 0) >= 4) or
                (max_pos.get("layer_10", 0) >= 5 and max_pos.get("layer_11", 0) >= 3) or
                (max_pos.get("layer_11", 0) >= 4 and max_pos.get("layer_12", 0) >= 4)
            ) and
            (
                (max_pos.get("layer_23", 0) >= 1) or
                (max_pos.get("layer_24", 0) >= 1)
            ) and
            (max_pos.get("layer_12", 0) >= 2) and
            (max_pos.get("layer_14", 0) >= 1)
        )

    def is_recogn() -> bool:
        return (
            (
                (max_pos.get("layer_9", 0) >= 2 and max_pos.get("layer_10", 0) >= 2) or
                (max_pos.get("layer_10", 0) >= 3 and max_pos.get("layer_12", 0) >= 3) or
                (max_pos.get("layer_12", 0) >= 4 and max_pos.get("layer_14", 0) >= 1)
            ) and
            (max_pos.get("layer_24", 0) >= 1) and
            (max_pos.get("layer_13", 0) <= 2)
        )

    if is_ocr():
        task_type = "ocr"
    elif is_position():
        task_type = "grounding"
    elif is_counting():
        task_type = "counting"
    elif is_recogn():
        task_type = "recogn"
    else:
        task_type = "counting"

    task_direct = task_direct_map[task_type]
    return task_direct, layer_to_task[task_type]

