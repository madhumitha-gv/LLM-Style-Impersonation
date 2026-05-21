"""
contrastive.py
--------------
Contrastive decoding for style impersonation.

Standard generation picks the most probable next token given the prompt.
Contrastive decoding instead picks tokens that are:
  - MORE likely given the person-specific prompt (expert model)
  - LESS likely given a neutral prompt (amateur model)

This amplifies what is uniquely person-specific in the output,
rather than what the model would generate generically.

Reference: Li et al. (2022) "Contrastive Decoding: Open-ended Text Generation
as Optimization"

Implementation:
  logits_contrastive = logits_expert - alpha * logits_amateur

Where:
  logits_expert  = token scores with full person context (RAG prompt)
  logits_amateur = token scores with no person context (neutral prompt)
  alpha          = amplification strength (default 0.1)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict

from data_loader import PersonData
from embeddings import StyleIndex
from prompts import (
    rag_prompt,
    neutral_prompt,
    style_transfer_rag,
    neutral_style_transfer_prompt,
    NEUTRAL_TEXTS,
)


# ── Core Contrastive Generation ────────────────────────────────────────────────

def generate_contrastive(
    expert_prompt: str,
    amateur_prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 150,
    alpha: float = 0.1,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generates text using contrastive decoding.

    At each step:
      1. Compute logits for both expert and amateur prompts
      2. Subtract alpha * amateur_logits from expert_logits
      3. Sample from the adjusted distribution

    Args:
        expert_prompt  - person-specific prompt (RAG context)
        amateur_prompt - neutral prompt (no person context)
        tokenizer      - loaded tokenizer
        model          - loaded model
        max_new_tokens - max tokens to generate
        alpha          - contrastive amplification strength (0.0 = no contrast)
        temperature    - sampling temperature
        top_p          - nucleus sampling threshold

    Returns:
        Generated text with person-specific style amplified
    """
    device = model.device

    # Tokenize both prompts
    expert_inputs = tokenizer(
        expert_prompt, return_tensors="pt"
    ).to(device)
    amateur_inputs = tokenizer(
        amateur_prompt, return_tensors="pt"
    ).to(device)

    expert_input_len = expert_inputs["input_ids"].shape[1]

    # Start with expert input_ids as the running sequence
    generated_ids = expert_inputs["input_ids"].clone()
    amateur_generated_ids = amateur_inputs["input_ids"].clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # ── Expert logits ──────────────────────────────────────────────────
            expert_out = model(input_ids=generated_ids)
            expert_logits = expert_out.logits[:, -1, :]  # (1, vocab_size)

            # ── Amateur logits ─────────────────────────────────────────────────
            amateur_out = model(input_ids=amateur_generated_ids)
            amateur_logits = amateur_out.logits[:, -1, :]  # (1, vocab_size)

            # ── Contrastive adjustment ─────────────────────────────────────────
            # Amplify tokens that expert prefers over amateur
            contrastive_logits = expert_logits - alpha * amateur_logits

            # ── Temperature + top-p sampling ───────────────────────────────────
            contrastive_logits = contrastive_logits / temperature
            probs = F.softmax(contrastive_logits, dim=-1)

            # Top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_idx)

            # ── Append to both sequences ───────────────────────────────────────
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            amateur_generated_ids = torch.cat(
                [amateur_generated_ids, next_token], dim=-1
            )

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode only newly generated tokens
    new_ids = generated_ids[0][expert_input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Task 1: Contrastive Impersonation ─────────────────────────────────────────

def run_contrastive_impersonation(
    person: PersonData,
    index: StyleIndex,
    tokenizer,
    model,
    top_k: int = 5,
    alpha: float = 0.1,
) -> List[Dict]:
    """
    Runs contrastive decoding impersonation on the person's test set.

    Expert prompt  = RAG prompt (person context + retrieved examples)
    Amateur prompt = neutral prompt (no person context)

    Args:
        person    - PersonData with test set
        index     - StyleIndex for RAG retrieval
        tokenizer - loaded tokenizer
        model     - loaded model
        top_k     - retrieved examples for RAG
        alpha     - contrastive amplification strength

    Returns:
        List of dicts with question, ground_truth, contrastive output
    """
    from tqdm import tqdm

    results = []
    print(f"\nRunning contrastive impersonation for {person.name} "
          f"({len(person.test)} test questions, alpha={alpha})...")

    for pair in tqdm(person.test):
        retrieved = index.retrieve(pair.question, top_k=top_k)

        expert = rag_prompt(person, pair.question, retrieved)
        amateur = neutral_prompt(pair.question)

        output = generate_contrastive(
            expert_prompt=expert,
            amateur_prompt=amateur,
            tokenizer=tokenizer,
            model=model,
            alpha=alpha,
        )

        results.append({
            "question":     pair.question,
            "ground_truth": pair.answer,
            "contrastive":  output,
        })

    return results


# ── Task 2: Contrastive Style Transfer ────────────────────────────────────────

def run_contrastive_style_transfer(
    person: PersonData,
    index: StyleIndex,
    tokenizer,
    model,
    top_k: int = 5,
    alpha: float = 0.1,
) -> List[Dict]:
    """
    Runs contrastive decoding style transfer on NEUTRAL_TEXTS.

    Expert prompt  = RAG style transfer prompt
    Amateur prompt = neutral rewrite prompt (no style target)

    Returns:
        List of dicts with topic, neutral_text, contrastive output
    """
    from tqdm import tqdm

    results = []
    print(f"\nRunning contrastive style transfer for {person.name} "
          f"({len(NEUTRAL_TEXTS)} texts, alpha={alpha})...")

    for item in tqdm(NEUTRAL_TEXTS):
        neutral_text = item["text"]
        retrieved = index.retrieve(neutral_text, top_k=top_k)

        expert = style_transfer_rag(person, neutral_text, retrieved)
        amateur = neutral_style_transfer_prompt(neutral_text)

        output = generate_contrastive(
            expert_prompt=expert,
            amateur_prompt=amateur,
            tokenizer=tokenizer,
            model=model,
            alpha=alpha,
        )

        results.append({
            "topic":        item["topic"],
            "neutral_text": neutral_text,
            "contrastive":  output,
        })

    return results