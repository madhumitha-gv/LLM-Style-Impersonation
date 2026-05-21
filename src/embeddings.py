"""
embeddings.py
-------------
Builds sentence embeddings for each person's training Q&A pairs
and creates a FAISS index for fast similarity retrieval.

Supports three retrieval modes:
  - retrieve()        : base RAG (semantic similarity to query)
  - retrieve_reranked(): RAG + style reranking (semantic + style score)
  - retrieve_style()  : style-biased retrieval (closest to style centroid)
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from data_loader import QAPair


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class StyleIndex:
    """
    FAISS index over a person's training Q&A pairs.
    Supports base retrieval, reranked retrieval, and style-biased retrieval.
    """

    def __init__(self, pairs: List[QAPair], model: SentenceTransformer):
        self.pairs = pairs
        self.model = model

        # Encode all training answers
        answers = [p.answer for p in pairs]
        self.answer_embeddings = model.encode(answers, normalize_embeddings=True)
        self.answer_embeddings = np.array(self.answer_embeddings, dtype=np.float32)

        # Style centroid = mean of all answer embeddings (normalized)
        centroid = self.answer_embeddings.mean(axis=0)
        self.style_centroid = (centroid / np.linalg.norm(centroid)).astype(np.float32)

        # FAISS index over questions (for semantic retrieval)
        questions = [p.question for p in pairs]
        question_embeddings = model.encode(questions, normalize_embeddings=True)
        question_embeddings = np.array(question_embeddings, dtype=np.float32)

        dim = question_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(question_embeddings)

    # ── Base RAG ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[QAPair, float]]:
        """
        Retrieves top-k Q&A pairs by semantic similarity to the query question.
        Standard RAG baseline.
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype=np.float32)

        scores, indices = self.index.search(query_emb, top_k)

        return [
            (self.pairs[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx < len(self.pairs)
        ]

    # ── RAG + Reranking ────────────────────────────────────────────────────────

    def retrieve_reranked(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 15,
        semantic_weight: float = 0.5,
        style_weight: float = 0.5,
    ) -> List[Tuple[QAPair, float]]:
        """
        Two-stage retrieval:
          1. Fetch fetch_k candidates by semantic similarity (broad recall)
          2. Rerank by combined score: semantic_sim * w1 + style_sim * w2

        style_sim measures how similar each candidate answer is to the
        person's overall style centroid — ensures retrieved examples are
        not just topically relevant but also stylistically representative.

        Args:
            query          - new question to answer
            top_k          - final number of examples to return
            fetch_k        - candidates to fetch before reranking (fetch_k > top_k)
            semantic_weight - weight for topic relevance score
            style_weight   - weight for style similarity score

        Returns:
            top_k examples sorted by combined score
        """
        # Stage 1: fetch broad candidates
        query_emb = self.model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype=np.float32)

        scores, indices = self.index.search(query_emb, min(fetch_k, len(self.pairs)))

        candidates = [
            (self.pairs[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx < len(self.pairs)
        ]

        # Stage 2: rerank by combined semantic + style score
        reranked = []
        for pair, semantic_score in candidates:
            # Style score: cosine similarity of answer to style centroid
            answer_emb = self.model.encode([pair.answer], normalize_embeddings=True)[0]
            style_score = float(np.dot(answer_emb, self.style_centroid))

            combined = semantic_weight * semantic_score + style_weight * style_score
            reranked.append((pair, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    # ── Style-Biased Retrieval ─────────────────────────────────────────────────

    def retrieve_style_biased(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 15,
        query_weight: float = 0.3,
        style_weight: float = 0.7,
    ) -> List[Tuple[QAPair, float]]:
        """
        Style-biased retrieval: prioritizes examples that are closest to
        the person's style centroid over topic relevance.

        Useful for terse or highly distinctive styles where stylistic
        consistency matters more than topical match.

        The query_weight is lower here by default — we care more about
        retrieving stylistically pure examples than topically matched ones.

        Args:
            query        - new question to answer
            top_k        - number of examples to return
            fetch_k      - initial candidate pool size
            query_weight - weight for semantic relevance (lower = more style-driven)
            style_weight - weight for style centroid similarity (higher = more style-driven)

        Returns:
            top_k examples biased toward person's style centroid
        """
        return self.retrieve_reranked(
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            semantic_weight=query_weight,
            style_weight=style_weight,
        )


def build_indexes(
    people_data: dict,
    model_name: str = EMBEDDING_MODEL
) -> Tuple[SentenceTransformer, dict]:
    """
    Loads the embedding model once and builds a StyleIndex per person.

    Returns:
        (model, indexes) where indexes = { "k": StyleIndex, "m": StyleIndex }
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    indexes = {}
    for name, person in people_data.items():
        print(f"Building FAISS index for {person.name} ({len(person.train)} pairs)...")
        indexes[name] = StyleIndex(pairs=person.train, model=model)

    print("Indexes ready.\n")
    return model, indexes