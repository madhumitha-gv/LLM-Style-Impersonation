"""
run_constrained.py
------------------
Runs a single additional condition: RAG + length constraint.

For each person, the max generation length is set to 1.5x their
average answer word count from the training set. This forces the
model to generate outputs that match the person's natural length,
testing whether the verbosity bias can be overcome with a token budget.

Person K: avg 26 words → max_tokens = 39
Person M: avg 9 words  → max_tokens = 14

Results are appended to existing CSV/JSON files from run_experiments.py.
Run this AFTER run_experiments.py has already completed.

Usage:
    python src/run_constrained.py
"""

import json
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all
from embeddings import build_indexes
from generator import load_model, generate
from evaluator import (
    compute_style_centroid,
    style_similarity_to_centroid,
    train_style_classifier,
    classify_generated,
)
from prompts import rag_prompt, style_transfer_rag, NEUTRAL_TEXTS

from google.colab import userdata
from huggingface_hub import login

RESULTS_DIR = Path("results")
CONDITION = "rag_constrained"


def compute_max_tokens(person) -> int:
    """
    Computes the token budget for a person based on their average
    answer word count. Uses 1.5x word count as a generous token budget
    (tokens ≈ words * 1.3 on average for English text).
    """
    avg_words = person.style_fingerprint["avg_words"]
    max_tokens = max(20, int(avg_words * 1.5))
    print(f"  {person.name}: avg_words={avg_words:.1f} → max_tokens={max_tokens}")
    return max_tokens


def run_rag_constrained_impersonation(person, index, tokenizer, model, max_tokens):
    """
    Runs RAG impersonation with length constraint on the person's test set.
    """
    results = []
    print(f"\nRunning RAG-constrained impersonation for {person.name} "
          f"(max_tokens={max_tokens})...")

    for pair in tqdm(person.test):
        retrieved = index.retrieve(pair.question, top_k=5)
        output = generate(
            rag_prompt(person, pair.question, retrieved),
            tokenizer, model,
            max_new_tokens=max_tokens
        )
        results.append({
            "question":        pair.question,
            "ground_truth":    pair.answer,
            CONDITION:         output,
        })

    return results


def run_rag_constrained_style_transfer(person, index, tokenizer, model, max_tokens):
    """
    Runs RAG style transfer with length constraint on NEUTRAL_TEXTS.
    """
    results = []
    print(f"\nRunning RAG-constrained style transfer for {person.name} "
          f"(max_tokens={max_tokens})...")

    for item in tqdm(NEUTRAL_TEXTS):
        neutral_text = item["text"]
        retrieved = index.retrieve(neutral_text, top_k=5)
        output = generate(
            style_transfer_rag(person, neutral_text, retrieved),
            tokenizer, model,
            max_new_tokens=max_tokens
        )
        results.append({
            "topic":        item["topic"],
            "neutral_text": neutral_text,
            CONDITION:      output,
        })

    return results


def append_to_results(name, imp_results, st_results):
    """
    Appends the rag_constrained condition to existing raw JSON files
    and existing eval CSV files.
    """
    # ── Update raw JSON ────────────────────────────────────────────────────────
    for task, new_results, filename in [
        ("impersonation", imp_results, f"{name}_impersonation_raw.json"),
        ("style_transfer", st_results, f"{name}_style_transfer_raw.json"),
    ]:
        path = RESULTS_DIR / filename
        with open(path) as f:
            existing = json.load(f)

        for i, item in enumerate(existing):
            item[CONDITION] = new_results[i][CONDITION]

        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Updated: {path}")

    # ── Update eval CSVs ───────────────────────────────────────────────────────
    for task, new_results, filename, key_col in [
        ("impersonation", imp_results,
         f"{name}_impersonation_eval.csv", "question"),
        ("style_transfer", st_results,
         f"{name}_style_transfer_eval.csv", "topic"),
    ]:
        path = RESULTS_DIR / filename
        df = pd.read_csv(path)

        generated = [r[CONDITION] for r in new_results]
        print(f"Updated eval CSV: {path}")

    return imp_results, st_results


def main():
    # Login
    login(token=userdata.get('HF_TOKEN'))

    # ── Load ───────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    people = load_all("data/raw")

    print("Building indexes...")
    embed_model, indexes = build_indexes(people)

    print("Loading model...")
    tokenizer, llm, device = load_model()

    # ── Train classifier once ──────────────────────────────────────────────────
    print("\nTraining style classifier...")
    classifier, cv_acc = train_style_classifier(people, embed_model)

    all_imp, all_st = {}, {}

    for name, person in people.items():
        print(f"\n{'='*60}")
        print(f"RAG-CONSTRAINED: {person.name.upper()}")
        print(f"{'='*60}")

        index = indexes[name]
        max_tokens = compute_max_tokens(person)
        train_answers = [p.answer for p in person.train]
        centroid = compute_style_centroid(train_answers, embed_model)

        # Run constrained impersonation
        imp_results = run_rag_constrained_impersonation(
            person, index, tokenizer, llm, max_tokens
        )

        # Run constrained style transfer
        st_results = run_rag_constrained_style_transfer(
            person, index, tokenizer, llm, max_tokens
        )

        # Save raw results
        imp_path = RESULTS_DIR / f"{name}_constrained_impersonation.json"
        st_path  = RESULTS_DIR / f"{name}_constrained_style_transfer.json"
        with open(imp_path, "w") as f:
            json.dump(imp_results, f, indent=2)
        with open(st_path, "w") as f:
            json.dump(st_results, f, indent=2)
        print(f"Saved: {imp_path}")
        print(f"Saved: {st_path}")

        # Evaluate style similarity
        imp_generated = [r[CONDITION] for r in imp_results]
        st_generated  = [r[CONDITION] for r in st_results]

        imp_sims = style_similarity_to_centroid(imp_generated, centroid, embed_model)
        st_sims  = style_similarity_to_centroid(st_generated,  centroid, embed_model)

        print(f"\n── {person.name} — RAG-Constrained Results ──")
        print(f"  Impersonation  style sim: {sum(imp_sims)/len(imp_sims):.3f}")
        print(f"  Style Transfer style sim: {sum(st_sims)/len(st_sims):.3f}")

        # Classifier check
        imp_clf = classify_generated(imp_generated, name.lower(), classifier, embed_model)
        st_clf  = classify_generated(st_generated,  name.lower(), classifier, embed_model)
        print(f"  Impersonation  classified correctly: {imp_clf['accuracy']:.1%}")
        print(f"  Style Transfer classified correctly: {st_clf['accuracy']:.1%}")

        # Show sample output
        print(f"\n── Sample output (Q: {imp_results[0]['question'][:50]}...) ──")
        print(f"  Ground truth : {imp_results[0]['ground_truth']}")
        print(f"  Constrained  : {imp_results[0][CONDITION]}")

        all_imp[name] = imp_results
        all_st[name]  = st_results

    print("\nDone.")


if __name__ == "__main__":
    main()