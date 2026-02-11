import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

from bedrock_utils import embed_text

@dataclass
class ChunkMeta:
    chunk_id: int
    text: str

class RagIndex:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim after normalization
        self.metas: List[ChunkMeta] = []

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add_chunks(self, chunks: List[str]):
        """
        Embeds chunks via Nova, adds to FAISS with metadata.
        """
        vectors = []
        for i, chunk in enumerate(chunks, start=len(self.metas) + 1):
            v = embed_text(chunk, dim=self.dim)
            v = np.array(v, dtype=np.float32)
            v = self._normalize(v)
            vectors.append(v)
            self.metas.append(ChunkMeta(chunk_id=i, text=chunk))

        mat = np.vstack(vectors).astype(np.float32)
        self.index.add(mat)

    def search(self, query: str, k: int) -> Tuple[List[ChunkMeta], List[float]]:
        qv = embed_text(query, dim=self.dim)
        qv = np.array(qv, dtype=np.float32)
        qv = self._normalize(qv).reshape(1, -1)

        scores, ids = self.index.search(qv, k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()

        results = []
        result_scores = []
        for idx, score in zip(ids, scores):
            if idx == -1:
                continue
            results.append(self.metas[idx])
            result_scores.append(score)
        return results, result_scores

    def save(self, path_prefix: str):
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in self.metas], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path_prefix: str, dim: int = 1024):
        obj = cls(dim=dim)
        obj.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.json", "r", encoding="utf-8") as f:
            metas = json.load(f)
        obj.metas = [ChunkMeta(**m) for m in metas]
        return obj
