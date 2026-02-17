from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

import numpy as np


class AtomicIndex:


    def __init__(self, index_path: str, triplets_path: str):
        self.index_path = index_path
        self.triplets_path = triplets_path

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(triplets_path):
            raise FileNotFoundError(f"triplets jsonl not found: {triplets_path}")

        import faiss  # lazy import
        self.faiss = faiss
        self.index = faiss.read_index(index_path)

        self.triplets: List[Dict[str, Any]] = []
        with open(triplets_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.triplets.append(json.loads(line))

        if len(self.triplets) == 0:
            raise ValueError("triplets.jsonl is empty.")

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def search(self, emb: np.ndarray, *, topk: int, src: str) -> List[Dict[str, Any]]:

        if emb is None:
            return []
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        if emb.shape[-1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {emb.shape[-1]}, expected {self.dim}")
        emb = emb.astype("float32", copy=False)

        scores, ids = self.index.search(emb, int(topk))  # (1, k)
        out: List[Dict[str, Any]] = []
        for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            if idx >= len(self.triplets):
                continue
            item = dict(self.triplets[idx])
            item["id"] = int(idx)
            item["score"] = float(s)
            item["src"] = str(src)
            out.append(item)
        return out
