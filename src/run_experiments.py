"""
run_experiments.py
------------------
Main entry point. Runs the full experiment pipeline:

  1. Load datasets (Khushi + Madhumitha)
  2. Build FAISS indexes for RAG
  3. Load LLaMA 3.1 8B
  4. Run Task 1: Impersonation (Zero-Shot, Few-Shot, RAG)
  5. Run Task 2: Style Transfer (Zero-Shot, Few-Shot, RAG)
  6. Evaluate all outputs
  7. Train style classifier
  8. Save results to results/

Usage (from repo root):
    python src/run_experiments.py

On Colab:
    !python src/run_experiments.py
"""

import json
import sys
import os
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all
from embeddings import build_indexes
from generator import load_model, run_impersonation, run_style_transfer
from evaluator import (
    evaluate_impersonation,
    evaluate_style_transfer,
    train_style_classifier,
    classify_generated,
    print_summary,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def save_results(data: dict, filename: str):
    """Saves a dict as JSON to results/."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def main():
    # ── 1. Load Data ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)
    people = load_all("data/raw")

    # ── 2. Build Indexes ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Building embedding indexes")
    print("=" * 60)
    embed_model, indexes = build_indexes(people)

    # ── 3. Load LLM ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Loading LLaMA 3.1 8B Instruct")
    print("=" * 60)
    tokenizer, llm = load_model()

    all_results = {}

    for name, person in people.items():
        print(f"\n{'=' * 60}")
        print(f"RUNNING EXPERIMENTS FOR: {person.name.upper()}")
        print(f"{'=' * 60}")

        index = indexes[name]

        # ── 4. Task 1: Impersonation ───────────────────────────────────────────
        print("\nSTEP 4: Task 1 — Impersonation")
        imp_results = run_impersonation(person, index, tokenizer, llm)
        save_results(imp_results, f"{name}_impersonation_raw.json")

        # ── 5. Task 2: Style Transfer ──────────────────────────────────────────
        print("\nSTEP 5: Task 2 — Style Transfer")
        st_results = run_style_transfer(person, index, tokenizer, llm)
        save_results(st_results, f"{name}_style_transfer_raw.json")

        # ── 6. Evaluate ────────────────────────────────────────────────────────
        print("\nSTEP 6: Evaluating outputs")
        imp_df = evaluate_impersonation(imp_results, embed_model)
        st_df = evaluate_style_transfer(
            st_results,
            person_train_answers=[p.answer for p in person.train],
            model=embed_model,
        )

        # Save eval tables
        imp_df.to_csv(RESULTS_DIR / f"{name}_impersonation_eval.csv", index=False)
        st_df.to_csv(RESULTS_DIR / f"{name}_style_transfer_eval.csv", index=False)

        print_summary(imp_df, st_df, person.name)

        all_results[name] = {
            "impersonation": imp_results,
            "style_transfer": st_results,
        }

    # ── 7. Style Classifier ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 7: Style Classifier")
    print("=" * 60)
    classifier, cv_acc = train_style_classifier(people, embed_model)

    # Check if generated outputs are classified to the correct person
    for name, person in people.items():
        imp_generated = [r["rag"] for r in all_results[name]["impersonation"]]
        clf_result = classify_generated(
            imp_generated, name.lower(), classifier, embed_model
        )
        print(f"{person.name} RAG outputs classified correctly: "
              f"{clf_result['accuracy']:.1%}")

    print(f"\nAll results saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()