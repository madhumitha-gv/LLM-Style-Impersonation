"""
evaluator.py
------------
Computes all evaluation metrics for both tasks:

  Task 1 - Impersonation:
    - BLEU score (n-gram overlap with ground truth)
    - ROUGE-L (longest common subsequence with ground truth)
    - Style Similarity (Sentence-BERT cosine vs style centroid)

  Task 2 - Style Transfer:
    - Style Similarity (Sentence-BERT cosine vs style centroid)
    - Style Classifier accuracy (can we tell whose style it is?)

  Both tasks:
    - Style Classifier run on ALL 6 conditions
    - Summary table comparing all strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# All 6 conditions
STRATEGIES = ["zero_shot", "few_shot", "rag", "rag_reranked", "rag_style", "contrastive"]


# ── BLEU ───────────────────────────────────────────────────────────────────────

def bleu_score(reference: str, hypothesis: str) -> float:
    """
    Sentence-level BLEU with smoothing.
    Measures n-gram word overlap between generated and ground truth answer.
    Low scores are expected — the model generates new text, not copies.
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L F1 score. Measures longest common subsequence between
    generated and ground truth answer. More lenient than BLEU.
    """
    scores = _rouge.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


# ── Style Centroid ─────────────────────────────────────────────────────────────

def compute_style_centroid(
    train_answers: List[str],
    model: SentenceTransformer
) -> np.ndarray:
    """
    Computes the style centroid — mean of all training answer embeddings,
    normalized to unit length.

    Used as the reference vector for style similarity measurement.
    Measuring against a centroid is more robust than comparing against
    individual ground truth answers, which may not be representative.

    Args:
        train_answers - person's training Q&A answers
        model         - SentenceTransformer

    Returns:
        Normalized centroid vector (384-dim)
    """
    embs = model.encode(train_answers, normalize_embeddings=True)
    centroid = embs.mean(axis=0)
    return (centroid / np.linalg.norm(centroid)).astype(np.float32)


# ── Style Similarity ───────────────────────────────────────────────────────────

def style_similarity_to_centroid(
    texts: List[str],
    centroid: np.ndarray,
    model: SentenceTransformer
) -> List[float]:
    """
    Measures how similar each generated text is to the person's style centroid.
    Higher = closer to the person's overall writing style.

    Args:
        texts    - list of generated texts
        centroid - person's style centroid from compute_style_centroid()
        model    - SentenceTransformer

    Returns:
        List of cosine similarity scores in [-1, 1]
    """
    embs = model.encode(texts, normalize_embeddings=True)
    return [float(np.dot(e, centroid)) for e in embs]


# ── Task 1: Impersonation Eval ─────────────────────────────────────────────────

def evaluate_impersonation(
    results: List[Dict],
    model: SentenceTransformer,
    train_answers: List[str]
) -> pd.DataFrame:
    """
    Evaluates impersonation results across all 6 strategies.

    Metrics:
        BLEU       - n-gram overlap with ground truth answer
        ROUGE-L    - LCS overlap with ground truth answer
        Style Sim  - cosine similarity to person's style centroid
                     (more meaningful than vs individual ground truth)

    Args:
        results       - merged impersonation results (all 6 conditions)
        model         - SentenceTransformer
        train_answers - person's training answers for centroid computation

    Returns:
        DataFrame with per-question scores + MEAN row
    """
    # Compute style centroid from training answers
    centroid = compute_style_centroid(train_answers, model)

    rows = []
    for item in results:
        ref = item["ground_truth"]
        row = {"question": item["question"][:60] + "..."}

        for strategy in STRATEGIES:
            hyp = item[strategy]
            row[f"{strategy}_bleu"]    = bleu_score(ref, hyp)
            row[f"{strategy}_rouge_l"] = rouge_l_score(ref, hyp)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Style similarity vs centroid (batched per strategy)
    for strategy in STRATEGIES:
        generated = [item[strategy] for item in results]
        sims = style_similarity_to_centroid(generated, centroid, model)
        df[f"{strategy}_style_sim"] = sims

    # Mean summary row
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
    Evaluates style transfer results across all 6 strategies.

    No ground truth exists for rewriting tasks — we measure style similarity
    to the person's style centroid. Higher = output sounds more like the person.

    Args:
        results              - merged style transfer results
        person_train_answers - person's training answers for centroid
        model                - SentenceTransformer

    Returns:
        DataFrame with per-text similarity scores + MEAN row
    """
    centroid = compute_style_centroid(person_train_answers, model)

    rows = []
    for item in results:
        row = {"topic": item["topic"]}
        for strategy in STRATEGIES:
            gen_emb = model.encode([item[strategy]], normalize_embeddings=True)[0]
            row[f"{strategy}_style_sim"] = float(np.dot(gen_emb, centroid))
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
    Trains a logistic regression classifier on Sentence-BERT embeddings
    to distinguish between people's writing styles.

    5-fold cross-validation accuracy:
        > 0.8 = styles clearly distinguishable
        < 0.6 = styles very similar

    Returns:
        (fitted classifier, cv_accuracy)
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
    Classifies generated texts and checks if they're attributed
    to the correct person. High accuracy = successful impersonation.

    Args:
        generated_texts - list of generated answers or rewrites
        true_person     - person whose style was targeted
        classifier      - trained LogisticRegression
        model           - SentenceTransformer

    Returns:
        Dict with predictions, true person, and accuracy
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

    mean_imp = impersonation_df[impersonation_df["question"] == "MEAN"].iloc[0]

    print("\n── Task 1: Impersonation ──")
    print(f"{'Strategy':<16} {'BLEU':>8} {'ROUGE-L':>10} {'Style Sim':>12}")
    print("-" * 48)
    for strategy in STRATEGIES:
        label = strategy.replace("_", "-").title()
        bleu  = mean_imp[f"{strategy}_bleu"]
        rouge = mean_imp[f"{strategy}_rouge_l"]
        sim   = mean_imp[f"{strategy}_style_sim"]
        print(f"{label:<16} {bleu:>8.3f} {rouge:>10.3f} {sim:>12.3f}")

    mean_st = style_transfer_df[style_transfer_df["topic"] == "MEAN"].iloc[0]

    print("\n── Task 2: Style Transfer ──")
    print(f"{'Strategy':<16} {'Style Sim':>12}")
    print("-" * 30)
    for strategy in STRATEGIES:
        label = strategy.replace("_", "-").title()
        sim   = mean_st[f"{strategy}_style_sim"]
        print(f"{label:<16} {sim:>12.3f}")