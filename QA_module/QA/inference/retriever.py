"""
Chunk retriever for the QA inference pipeline.

Two backends:

  TfidfChunkRetriever — sklearn TF-IDF with a NumPy term-frequency
                        fallback. Fast and lexically precise.

  BM25ChunkRetriever  — Okapi BM25 (implemented from scratch, no
                        external deps). Handles long verbose queries
                        better than TF-IDF because its term-frequency
                        saturation prevents high-frequency words from
                        dominating. Key fix for CuAD template questions.

  HybridChunkRetriever — runs both TF-IDF and BM25, merges their
                         ranked lists via reciprocal rank fusion, and
                         returns the fused top-k. Default for the
                         legal pipeline.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict

from QA.data.chunking import chunk_document, Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


# -------------------------------------------------------
# TF-IDF
# -------------------------------------------------------
class TfidfChunkRetriever:

    def __init__(self, document: str, chunk_words: int = 400, overlap_words: int = 50):
        self.chunks: List[Chunk] = chunk_document(document, chunk_words=chunk_words, overlap_words=overlap_words)
        self._backend = "none"

        if not self.chunks:
            return

        texts = [c.text for c in self.chunks]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(lowercase=True, stop_words="english")
            mat = vec.fit_transform(texts)
            self._vectorizer = vec
            self._matrix = mat
            self._backend = "sklearn"
        except Exception:
            import numpy as np
            vocab: Dict[str, int] = {}
            rows = []
            for t in texts:
                counts = Counter(t.lower().split())
                for w in counts:
                    if w not in vocab:
                        vocab[w] = len(vocab)
                rows.append(counts)
            V = len(vocab)
            mat = np.zeros((len(texts), V), dtype="float32")
            for i, counts in enumerate(rows):
                for w, c in counts.items():
                    mat[i, vocab[w]] = c
                norm = np.linalg.norm(mat[i])
                if norm > 0:
                    mat[i] /= norm
            self._vocab = vocab
            self._matrix = mat
            self._backend = "numpy"

    def _score(self, query: str):
        if self._backend == "sklearn":
            q_vec = self._vectorizer.transform([query])
            return (self._matrix @ q_vec.T).toarray().ravel()
        elif self._backend == "numpy":
            import numpy as np
            counts = Counter(query.lower().split())
            qv = np.zeros(len(self._vocab), dtype="float32")
            for w, c in counts.items():
                j = self._vocab.get(w)
                if j is not None:
                    qv[j] = c
            norm = float(np.linalg.norm(qv))
            if norm > 0:
                qv /= norm
            return self._matrix @ qv
        return []

    def top_k(self, query: str, k: int = 3, history: Optional[List[str]] = None) -> List[Retrieved]:
        if not self.chunks:
            return []
        parts = [query] + (history[-2:] if history else [])
        full_query = " ".join(p for p in parts if p)
        sims = self._score(full_query)
        order = sorted(range(len(sims)), key=lambda i: -float(sims[i]))[:k]
        return [Retrieved(chunk=self.chunks[i], score=float(sims[i])) for i in order]


# -------------------------------------------------------
# BM25 (from scratch)
# -------------------------------------------------------
_WORD_RE = re.compile(r"\w+")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "with",
    "is", "was", "were", "be", "are", "that", "this", "it", "as",
    "at", "by", "from", "or", "but", "not", "no", "if", "do", "does",
    "did", "has", "have", "had", "will", "would", "should", "can",
    "may", "might", "he", "she", "they", "we", "i", "you",
})


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


class BM25ChunkRetriever:

    def __init__(self, document: str, chunk_words: int = 400, overlap_words: int = 50,
                 k1: float = 1.5, b: float = 0.75):
        self.chunks: List[Chunk] = chunk_document(document, chunk_words=chunk_words, overlap_words=overlap_words)
        self.k1 = k1
        self.b = b

        if not self.chunks:
            self._doc_freqs = {}
            self._doc_lens = []
            self._avgdl = 0
            self._tf = []
            return

        self._tf = []           # per-chunk term frequency
        self._doc_lens = []
        df: Dict[str, int] = {}

        for c in self.chunks:
            tokens = _tokenize(c.text)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_lens.append(len(tokens))
            for t in tf:
                df[t] = df.get(t, 0) + 1

        self._doc_freqs = df
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 1
        self._N = len(self.chunks)

    def _score_query(self, query: str):
        q_tokens = _tokenize(query)
        scores = [0.0] * self._N
        for t in q_tokens:
            if t not in self._doc_freqs:
                continue
            df_t = self._doc_freqs[t]
            idf = math.log((self._N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            for i in range(self._N):
                tf_t = self._tf[i].get(t, 0)
                dl = self._doc_lens[i]
                numerator = tf_t * (self.k1 + 1)
                denominator = tf_t + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * numerator / denominator
        return scores

    def top_k(self, query: str, k: int = 3, history: Optional[List[str]] = None) -> List[Retrieved]:
        if not self.chunks:
            return []
        parts = [query] + (history[-2:] if history else [])
        full_query = " ".join(p for p in parts if p)
        scores = self._score_query(full_query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Retrieved(chunk=self.chunks[i], score=scores[i]) for i in order]


# -------------------------------------------------------
# Hybrid (reciprocal rank fusion)
# -------------------------------------------------------
class HybridChunkRetriever:

    def __init__(self, document: str, chunk_words: int = 400, overlap_words: int = 50,
                 rrf_k: int = 60):
        self._tfidf = TfidfChunkRetriever(document, chunk_words, overlap_words)
        self._bm25  = BM25ChunkRetriever(document, chunk_words, overlap_words)
        self.chunks = self._tfidf.chunks
        self._rrf_k = rrf_k

    def top_k(self, query: str, k: int = 3, history: Optional[List[str]] = None) -> List[Retrieved]:
        if not self.chunks:
            return []

        n = len(self.chunks)
        tfidf_results = self._tfidf.top_k(query, k=n, history=history)
        bm25_results  = self._bm25.top_k(query, k=n, history=history)

        rrf_scores: Dict[int, float] = {}
        for rank, r in enumerate(tfidf_results):
            rrf_scores[r.chunk.chunk_idx] = rrf_scores.get(r.chunk.chunk_idx, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        for rank, r in enumerate(bm25_results):
            rrf_scores[r.chunk.chunk_idx] = rrf_scores.get(r.chunk.chunk_idx, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        ordered = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        chunk_lookup = {c.chunk_idx: c for c in self.chunks}
        return [Retrieved(chunk=chunk_lookup[idx], score=score) for idx, score in ordered]
