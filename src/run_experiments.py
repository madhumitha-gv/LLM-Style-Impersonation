"""
run_experiments.py
------------------
Main entry point. Runs the full experiment pipeline across 6 conditions:

  Impersonation:
    1. Zero-Shot
    2. Few-Shot
    3. RAG (base)
    4. RAG + Reranking
    5. RAG + Style-Biased Retrieval
    6. Contrastive Decoding (RAG expert vs neutral amateur)

  Style Transfer:
    Same 6 conditions on NEUTRAL_TEXTS corpus.

Usage:
    python src/run_experiments.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all
from embeddings import build_indexes
from generator import load_model, run_impersonation, run_style_transfer
from contrastive import run_contrastive_impersonation, run_contrastive_style_transfer
from evaluator import (
    evaluate_impersonation,
    evaluate_style_transfer,
    train_style_classifier,
    classify_generated,
    print_summary,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def save_results(data, filename: str):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def merge_results(base_results, reranked_results, style_biased_results, contrastive_results):
    """
    Merges all 6 conditions into a single list of dicts per question.
    base_results already has zero_shot, few_shot, rag keys.
    """
    merged = []
    for i, base in enumerate(base_results):
        merged.append({
            "question":      base["question"],
            "ground_truth":  base["ground_truth"],
            "zero_shot":     base["zero_shot"],
            "few_shot":      base["few_shot"],
            "rag":           base["rag"],
            "rag_reranked":  reranked_results[i]["rag"],
            "rag_style":     style_biased_results[i]["rag"],
            "contrastive":   contrastive_results[i]["contrastive"],
        })
    return merged


def merge_st_results(base, reranked, style_biased, contrastive):
    """Merges style transfer results across all conditions."""
    merged = []
    for i, b in enumerate(base):
        merged.append({
            "topic":         b["topic"],
            "neutral_text":  b["neutral_text"],
            "zero_shot":     b["zero_shot"],
            "few_shot":      b["few_shot"],
            "rag":           b["rag"],
            "rag_reranked":  reranked[i]["rag"],
            "rag_style":     style_biased[i]["rag"],
            "contrastive":   contrastive[i]["contrastive"],
        })
    return merged


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

    # ── 3. Load Model ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Loading model (LLaMA 3.1 8B on GPU / Mistral 7B on CPU)")
    print("=" * 60)
    tokenizer, llm, device = load_model()
    print(f"Running on: {device.upper()}")

    all_results = {}

    for name, person in people.items():
        print(f"\n{'=' * 60}")
        print(f"RUNNING EXPERIMENTS FOR: {person.name.upper()}")
        print(f"{'=' * 60}")

        index = indexes[name]

        # ── 4. Base conditions (Zero-Shot, Few-Shot, RAG) ──────────────────────
        print("\nSTEP 4: Zero-Shot | Few-Shot | RAG")
        base_imp = run_impersonation(person, index, tokenizer, llm)
        base_st  = run_style_transfer(person, index, tokenizer, llm)

        # ── 5. RAG + Reranking ─────────────────────────────────────────────────
        print("\nSTEP 5: RAG + Reranking")
        # Temporarily swap index retrieve to reranked version
        from prompts import rag_prompt, style_transfer_rag
        from tqdm import tqdm

        reranked_imp = []
        for pair in tqdm(person.test, desc="Reranked impersonation"):
            retrieved = index.retrieve_reranked(pair.question)
            from generator import generate
            output = generate(rag_prompt(person, pair.question, retrieved), tokenizer, llm)
            reranked_imp.append({
                "question": pair.question, "ground_truth": pair.answer, "rag": output
            })

        reranked_st = []
        from prompts import NEUTRAL_TEXTS
        for item in tqdm(NEUTRAL_TEXTS, desc="Reranked style transfer"):
            retrieved = index.retrieve_reranked(item["text"])
            output = generate(style_transfer_rag(person, item["text"], retrieved), tokenizer, llm)
            reranked_st.append({
                "topic": item["topic"], "neutral_text": item["text"], "rag": output
            })

        # ── 6. RAG + Style-Biased Retrieval ───────────────────────────────────
        print("\nSTEP 6: RAG + Style-Biased Retrieval")

        style_biased_imp = []
        for pair in tqdm(person.test, desc="Style-biased impersonation"):
            retrieved = index.retrieve_style_biased(pair.question)
            output = generate(rag_prompt(person, pair.question, retrieved), tokenizer, llm)
            style_biased_imp.append({
                "question": pair.question, "ground_truth": pair.answer, "rag": output
            })

        style_biased_st = []
        for item in tqdm(NEUTRAL_TEXTS, desc="Style-biased style transfer"):
            retrieved = index.retrieve_style_biased(item["text"])
            output = generate(style_transfer_rag(person, item["text"], retrieved), tokenizer, llm)
            style_biased_st.append({
                "topic": item["topic"], "neutral_text": item["text"], "rag": output
            })

        # ── 7. Contrastive Decoding ────────────────────────────────────────────
        print("\nSTEP 7: Contrastive Decoding")
        contrastive_imp = run_contrastive_impersonation(
            person, index, tokenizer, llm, alpha=0.1
        )
        contrastive_st = run_contrastive_style_transfer(
            person, index, tokenizer, llm, alpha=0.1
        )

        # ── 8. Merge all conditions ────────────────────────────────────────────
        merged_imp = merge_results(
            base_imp, reranked_imp, style_biased_imp, contrastive_imp
        )
        merged_st = merge_st_results(
            base_st, reranked_st, style_biased_st, contrastive_st
        )

        save_results(merged_imp, f"{name}_impersonation_raw.json")
        save_results(merged_st,  f"{name}_style_transfer_raw.json")

        # ── 9. Evaluate ────────────────────────────────────────────────────────
        print("\nSTEP 8: Evaluating all conditions")
        imp_df = evaluate_impersonation(merged_imp, embed_model)
        st_df  = evaluate_style_transfer(
            merged_st,
            person_train_answers=[p.answer for p in person.train],
            model=embed_model,
        )

        imp_df.to_csv(RESULTS_DIR / f"{name}_impersonation_eval.csv", index=False)
        st_df.to_csv(RESULTS_DIR  / f"{name}_style_transfer_eval.csv", index=False)

        print_summary(imp_df, st_df, person.name)

        all_results[name] = {"impersonation": merged_imp, "style_transfer": merged_st}

    # ── 10. Style Classifier ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 9: Style Classifier")
    print("=" * 60)
    classifier, cv_acc = train_style_classifier(people, embed_model)

    for name, person in people.items():
        for condition in ["rag", "rag_reranked", "rag_style", "contrastive"]:
            generated = [r[condition] for r in all_results[name]["impersonation"]]
            from evaluator import classify_generated
            clf_result = classify_generated(generated, name.lower(), classifier, embed_model)
            print(f"{person.name} [{condition}] classified correctly: "
                  f"{clf_result['accuracy']:.1%}")

    print(f"\nAll results saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()