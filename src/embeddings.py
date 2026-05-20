"""
embeddings.py
-------------
Builds sentence embeddings for each person's training Q&A pairs
and creates a FAISS index for fast similarity retrieval (used in RAG).
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from data_loader import QAPair


# ── Embedding Model ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, good quality, 384-dim


# ── Index ──────────────────────────────────────────────────────────────────────

class StyleIndex:
    """
    Wraps a FAISS index over a person's training Q&A pairs.
    Given a new question, retrieves the top-k most semantically
    similar Q&A pairs from that person's training set.
    """

    def __init__(self, pairs: List[QAPair], model: SentenceTransformer):
        """
        Args:
            pairs  - person's training Q&A pairs
            model  - shared SentenceTransformer instance
        """
        self.pairs = pairs
        self.model = model
        self.index = self._build_index(pairs)

    def _build_index(self, pairs: List[QAPair]) -> faiss.IndexFlatIP:
        """
        Encodes all training questions and builds a FAISS
        inner-product index (equivalent to cosine sim on normalized vectors).
        """
        questions = [p.question for p in pairs]
        embeddings = self.model.encode(questions, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[QAPair, float]]:
        """
        Retrieves the top-k most similar Q&A pairs for a given query.

        Args:
            query  - the new question to answer
            top_k  - number of examples to retrieve

        Returns:
            List of (QAPair, similarity_score) sorted by descending similarity
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            (self.pairs[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx < len(self.pairs)
        ]

        return results


# ── Factory ────────────────────────────────────────────────────────────────────

def build_indexes(
    people_data: dict,
    model_name: str = EMBEDDING_MODEL
) -> Tuple[SentenceTransformer, dict]:
    """
    Loads the embedding model once and builds a StyleIndex
    for each person in people_data.

    Args:
        people_data - dict from data_loader.load_all()
        model_name  - sentence-transformers model to use

    Returns:
        (model, indexes) where indexes = { "khushi": StyleIndex, ... }
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    indexes = {}
    for name, person in people_data.items():
        print(f"Building FAISS index for {person.name} ({len(person.train)} pairs)...")
        indexes[name] = StyleIndex(pairs=person.train, model=model)

    print("Indexes ready.\n")
    return model, indexes


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_loader import load_all

    people = load_all("data/raw")
    model, indexes = build_indexes(people)

    # Test retrieval for Khushi
    query = "How do you handle stress at work?"
    results = indexes["khushi"].retrieve(query, top_k=3)

    print(f"Query: '{query}'")
    print(f"Top-3 retrieved examples for Khushi:")
    for pair, score in results:
        print(f"  [{score:.3f}] Q: {pair.question}")
        print(f"           A: {pair.answer[:80]}...")
