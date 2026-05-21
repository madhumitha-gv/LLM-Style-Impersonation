"""
evaluator.py
------------
Computes all evaluation metrics for both tasks:

  Task 1 - Impersonation:
    - BLEU score (n-gram overlap with ground truth)
    - ROUGE-L (longest common subsequence)
    - Style Similarity (Sentence-BERT cosine vs ground truth)

  Task 2 - Style Transfer:
    - Style Similarity only (no ground truth exists for rewriting)
    - Style Classifier accuracy (can we tell whose style it is?)

  Both tasks:
    - Summary table comparing Zero-Shot vs Few-Shot vs RAG
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Download NLTK tokenizer data (first run only)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── BLEU ───────────────────────────────────────────────────────────────────────

def bleu_score(reference: str, hypothesis: str) -> float:
    """
    Computes sentence-level BLEU score with smoothing.
    Handles short sequences gracefully.

    Args:
        reference  - ground truth answer
        hypothesis - model generated answer

    Returns:
        BLEU score in [0, 1]
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """
    Computes ROUGE-L F1 score between reference and hypothesis.

    Args:
        reference  - ground truth answer
        hypothesis - model generated answer

    Returns:
        ROUGE-L F1 score in [0, 1]
    """
    scores = _rouge.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


# ── Style Similarity ───────────────────────────────────────────────────────────

def style_similarity(
    texts_a: List[str],
    texts_b: List[str],
    model: SentenceTransformer
) -> List[float]:
    """
    Computes pairwise cosine similarity between two lists of texts
    using Sentence-BERT embeddings.

    For impersonation: texts_a = ground truth, texts_b = generated
    For style transfer: texts_a = person's training answers, texts_b = generated rewrites

    Returns:
        List of cosine similarity scores in [-1, 1]
    """
    emb_a = model.encode(texts_a, normalize_embeddings=True)
    emb_b = model.encode(texts_b, normalize_embeddings=True)

    # Pairwise dot product of normalized vectors = cosine similarity
    similarities = [float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)]
    return similarities


# ── Task 1: Impersonation Eval ─────────────────────────────────────────────────

def evaluate_impersonation(
    results: List[Dict],
    model: SentenceTransformer
) -> pd.DataFrame:
    """
    Evaluates impersonation results across all three strategies.

    Args:
        results - output of generator.run_impersonation()
        model   - SentenceTransformer for style similarity

    Returns:
        DataFrame with per-question scores and mean summary row
    """
    strategies = ["zero_shot", "few_shot", "rag", "rag_reranked", "rag_style", "contrastive"]
    rows = []

    for item in results:
        ref = item["ground_truth"]
        row = {"question": item["question"][:60] + "..."}

        for strategy in strategies:
            hyp = item[strategy]
            row[f"{strategy}_bleu"] = bleu_score(ref, hyp)
            row[f"{strategy}_rouge_l"] = rouge_l_score(ref, hyp)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute style similarity per strategy (batch for efficiency)
    ground_truths = [item["ground_truth"] for item in results]
    for strategy in strategies:
        generated = [item[strategy] for item in results]
        sims = style_similarity(ground_truths, generated, model)
        df[f"{strategy}_style_sim"] = sims

    # Add mean summary row
    numeric_cols = [c for c in df.columns if c != "question"]
    mean_row = df[numeric_cols].mean().to_dict()
    mean_row["question"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


# ── Task 2: Style Transfer Eval ────────────────────────────────────────────────

def evaluate_style_transfer(
    results: List[Dict],
    person_train_answers: List[str],
    model: SentenceTransformer
) -> pd.DataFrame:
    """
    Evaluates style transfer results.
    No ground truth exists — we measure how similar the rewrites are
    to the person's actual writing style (using their training answers as reference).

    Args:
        results              - output of generator.run_style_transfer()
        person_train_answers - person's training answers as style reference
        model                - SentenceTransformer

    Returns:
        DataFrame with per-text similarity scores
    """
    strategies = ["zero_shot", "few_shot", "rag", "rag_reranked", "rag_style", "contrastive"]

    # Use the mean of training answer embeddings as the "style centroid"
    style_embs = model.encode(person_train_answers, normalize_embeddings=True)
    style_centroid = style_embs.mean(axis=0)
    style_centroid = style_centroid / np.linalg.norm(style_centroid)

    rows = []
    for item in results:
        row = {"topic": item["topic"]}
        for strategy in strategies:
            generated = item[strategy]
            gen_emb = model.encode([generated], normalize_embeddings=True)[0]
            sim = float(np.dot(gen_emb, style_centroid))
            row[f"{strategy}_style_sim"] = sim
        rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [c for c in df.columns if c != "topic"]
    mean_row = df[numeric_cols].mean().to_dict()
    mean_row["topic"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


# ── Style Classifier ───────────────────────────────────────────────────────────

def train_style_classifier(
    people_data: dict,
    model: SentenceTransformer
) -> Tuple[LogisticRegression, float]:
    """
    Trains a logistic regression classifier to distinguish between
    people's writing styles using Sentence-BERT embeddings.

    This answers: "Are the two people's styles distinguishable?"
    High accuracy = strong style separation.
    High confusion = styles are similar.

    Args:
        people_data - dict from data_loader.load_all()
        model       - SentenceTransformer

    Returns:
        (classifier, cross_val_accuracy)
    """
    texts, labels = [], []

    for name, person in people_data.items():
        for pair in person.train + person.test:
            texts.append(pair.answer)
            labels.append(name)

    embeddings = model.encode(texts, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, embeddings, labels, cv=5)
    cv_accuracy = float(cv_scores.mean())

    clf.fit(embeddings, labels)

    print(f"\nStyle Classifier (5-fold CV accuracy): {cv_accuracy:.3f}")
    print("  > 0.8 = styles are clearly distinguishable")
    print("  < 0.6 = styles are very similar\n")

    return clf, cv_accuracy


def classify_generated(
    generated_texts: List[str],
    true_person: str,
    classifier: LogisticRegression,
    model: SentenceTransformer
) -> Dict:
    """
    Classifies generated texts — checks if they're identified as
    belonging to the correct person.

    Args:
        generated_texts - list of generated answers/rewrites
        true_person     - the person whose style was targeted
        classifier      - trained LogisticRegression
        model           - SentenceTransformer

    Returns:
        Dict with predictions and accuracy
    """
    embeddings = model.encode(generated_texts, normalize_embeddings=True)
    predictions = classifier.predict(embeddings)

    accuracy = sum(p == true_person for p in predictions) / len(predictions)

    return {
        "predictions": list(predictions),
        "true_person": true_person,
        "accuracy": accuracy,
    }


# ── Summary Printer ────────────────────────────────────────────────────────────

def print_summary(
    impersonation_df: pd.DataFrame,
    style_transfer_df: pd.DataFrame,
    person_name: str
):
    """Prints a clean summary table for a single person."""

    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {person_name.upper()}")
    print(f"{'='*60}")

    mean_row = impersonation_df[impersonation_df["question"] == "MEAN"].iloc[0]

    print("\n── Task 1: Impersonation ──")
    print(f"{'Strategy':<12} {'BLEU':>8} {'ROUGE-L':>10} {'Style Sim':>12}")
    print("-" * 44)
    for strategy in ["zero_shot", "few_shot", "rag", "rag_reranked", "rag_style", "contrastive"]:
        label = strategy.replace("_", "-").title()
        bleu = mean_row[f"{strategy}_bleu"]
        rouge = mean_row[f"{strategy}_rouge_l"]
        sim = mean_row[f"{strategy}_style_sim"]
        print(f"{label:<12} {bleu:>8.3f} {rouge:>10.3f} {sim:>12.3f}")

    mean_st = style_transfer_df[style_transfer_df["topic"] == "MEAN"].iloc[0]

    print("\n── Task 2: Style Transfer ──")
    print(f"{'Strategy':<12} {'Style Sim':>12}")
    print("-" * 26)
    for strategy in ["zero_shot", "few_shot", "rag", "rag_reranked", "rag_style", "contrastive"]:
        label = strategy.replace("_", "-").title()
        sim = mean_st[f"{strategy}_style_sim"]
        print(f"{label:<12} {sim:>12.3f}")