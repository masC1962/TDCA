from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..models import Passage, RetrievalHit
from .base import BaseRetriever, DenseDependencyError


class DenseRetriever(BaseRetriever):
    name = "dense"
    _model_cache: dict[str, object] = {}

    def __init__(self, passages: list[Passage], model_name: str, fallback: str = "error", index_path: str = "") -> None:
        self.passages = passages
        self.model_name = model_name
        self.fallback = fallback
        self.backend = "sentence_transformers"
        self.model = None
        self.matrix = None
        self.vectorizer = None
        if index_path:
            self._load(Path(index_path))
            return
        if importlib.util.find_spec("sentence_transformers") is not None:
            from sentence_transformers import SentenceTransformer

            if model_name not in self._model_cache:
                self._model_cache[model_name] = SentenceTransformer(model_name)
            self.model = self._model_cache[model_name]
            self.matrix = np.asarray(self.model.encode([f"{p.title} {p.text}" for p in passages], normalize_embeddings=True))
        elif fallback == "explicit_tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.backend = "explicit_tfidf_fallback"
            self.name = "dense[explicit_tfidf_fallback]"
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.matrix = self.vectorizer.fit_transform([f"{p.title} {p.text}" for p in passages])
        else:
            raise DenseDependencyError(
                "dense retrieval requires sentence-transformers; install the dense extra or explicitly set dense_fallback=explicit_tfidf"
            )

    @staticmethod
    def corpus_fingerprint(passages: list[Passage]) -> str:
        digest = sha256()
        for passage in passages:
            digest.update(json.dumps(
                [passage.passage_id, passage.title, passage.text], ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.backend == "sentence_transformers":
            np.save(target / "embeddings.npy", np.asarray(self.matrix, dtype=np.float32), allow_pickle=False)
        else:
            from joblib import dump
            from scipy.sparse import save_npz

            save_npz(target / "tfidf_matrix.npz", self.matrix)
            dump(self.vectorizer, target / "tfidf_vectorizer.joblib")
        (target / "dense_manifest.json").write_text(json.dumps({
            "format_version": 1,
            "backend": self.backend,
            "model_name": self.model_name,
            "passage_count": len(self.passages),
            "corpus_fingerprint": self.corpus_fingerprint(self.passages),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load(self, path: Path) -> None:
        manifest_path = path / "dense_manifest.json"
        if not manifest_path.exists():
            raise DenseDependencyError(f"dense index manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = self.corpus_fingerprint(self.passages)
        if manifest.get("corpus_fingerprint") != expected or manifest.get("passage_count") != len(self.passages):
            raise DenseDependencyError("dense index corpus fingerprint/count mismatch")
        if manifest.get("model_name") != self.model_name:
            raise DenseDependencyError("dense index encoder mismatch")
        self.backend = str(manifest.get("backend"))
        if self.backend == "sentence_transformers":
            if importlib.util.find_spec("sentence_transformers") is None:
                raise DenseDependencyError("query encoding requires sentence-transformers even when embeddings are prebuilt")
            from sentence_transformers import SentenceTransformer

            if self.model_name not in self._model_cache:
                self._model_cache[self.model_name] = SentenceTransformer(self.model_name)
            self.model = self._model_cache[self.model_name]
            self.matrix = np.load(path / "embeddings.npy", allow_pickle=False)
        elif self.backend == "explicit_tfidf_fallback":
            from joblib import load
            from scipy.sparse import load_npz

            self.name = "dense[explicit_tfidf_fallback]"
            self.vectorizer = load(path / "tfidf_vectorizer.joblib")
            self.matrix = load_npz(path / "tfidf_matrix.npz")
        else:
            raise DenseDependencyError(f"unsupported dense index backend: {self.backend}")

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.backend == "sentence_transformers":
            query_vector = np.asarray(self.model.encode([query], normalize_embeddings=True))[0]
            scores = np.asarray(self.matrix @ query_vector).reshape(-1)
        else:
            query_vector = self.vectorizer.transform([query])
            scores = (self.matrix @ query_vector.T).toarray().reshape(-1)
        indices = sorted(range(len(self.passages)), key=lambda index: (-float(scores[index]), self.passages[index].passage_id))[:top_k]
        return [
            RetrievalHit(self.passages[index], float(scores[index]), rank, self.name, query)
            for rank, index in enumerate(indices, start=1)
        ]
