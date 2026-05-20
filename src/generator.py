"""
generator.py
------------
Loads LLaMA 3.1 8B Instruct in 4-bit quantization and generates
responses for all prompting strategies across both tasks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from typing import List, Dict

from data_loader import PersonData, QAPair
from embeddings import StyleIndex
from prompts import (
    zero_shot_prompt,
    few_shot_prompt,
    rag_prompt,
    style_transfer_zero_shot,
    style_transfer_few_shot,
    style_transfer_rag,
    NEUTRAL_TEXTS,
)


# ── Model Config ───────────────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 4-bit quantization — fits on T4 (16GB) and runs fast on H100
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


# ── Model Loader ───────────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID):
    """
    Loads LLaMA 3.1 8B Instruct with 4-bit quantization.
    Requires a HuggingFace token for gated model access.

    Set your token via:
        huggingface-cli login
    or:
        export HUGGINGFACE_TOKEN=your_token_here

    Returns:
        (tokenizer, model)
    """
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    print("Model ready.\n")
    return tokenizer, model


# ── Single Generation ──────────────────────────────────────────────────────────

def generate(prompt: str, tokenizer, model) -> str:
    """
    Generates a single response for a given prompt.
    Strips the prompt prefix from the output.

    Args:
        prompt    - fully formatted Llama 3 chat prompt
        tokenizer - loaded tokenizer
        model     - loaded model

    Returns:
        Generated text (answer only, no prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated_ids = output[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Task 1: Impersonation ──────────────────────────────────────────────────────

def run_impersonation(
    person: PersonData,
    index: StyleIndex,
    tokenizer,
    model,
    top_k: int = 5,
) -> List[Dict]:
    """
    Runs all three impersonation strategies on the person's test set.

    For each test question:
        - Zero-Shot: generate with style description only
        - Few-Shot:  generate with 5 random training examples
        - RAG:       generate with top-5 retrieved examples

    Returns:
        List of dicts, one per test question:
        {
            "question": str,
            "ground_truth": str,
            "zero_shot": str,
            "few_shot": str,
            "rag": str,
        }
    """
    results = []

    print(f"\nRunning impersonation for {person.name} ({len(person.test)} test questions)...")

    for pair in tqdm(person.test):
        retrieved = index.retrieve(pair.question, top_k=top_k)

        zs = generate(zero_shot_prompt(person, pair.question), tokenizer, model)
        fs = generate(few_shot_prompt(person, pair.question), tokenizer, model)
        rg = generate(rag_prompt(person, pair.question, retrieved), tokenizer, model)

        results.append({
            "question": pair.question,
            "ground_truth": pair.answer,
            "zero_shot": zs,
            "few_shot": fs,
            "rag": rg,
        })

    return results


# ── Task 2: Style Transfer ─────────────────────────────────────────────────────

def run_style_transfer(
    person: PersonData,
    index: StyleIndex,
    tokenizer,
    model,
    top_k: int = 5,
) -> List[Dict]:
    """
    Runs all three style transfer strategies on the NEUTRAL_TEXTS corpus.

    For each neutral text:
        - Zero-Shot: rewrite with style description only
        - Few-Shot:  rewrite with 5 random training examples
        - RAG:       rewrite with top-5 retrieved examples (query = neutral text)

    Returns:
        List of dicts, one per neutral text:
        {
            "topic": str,
            "neutral_text": str,
            "zero_shot": str,
            "few_shot": str,
            "rag": str,
        }
    """
    results = []

    print(f"\nRunning style transfer for {person.name} ({len(NEUTRAL_TEXTS)} texts)...")

    for item in tqdm(NEUTRAL_TEXTS):
        neutral_text = item["text"]

        # For RAG, retrieve using the neutral text as the query
        retrieved = index.retrieve(neutral_text, top_k=top_k)

        zs = generate(style_transfer_zero_shot(person, neutral_text), tokenizer, model)
        fs = generate(style_transfer_few_shot(person, neutral_text), tokenizer, model)
        rg = generate(style_transfer_rag(person, neutral_text, retrieved), tokenizer, model)

        results.append({
            "topic": item["topic"],
            "neutral_text": neutral_text,
            "zero_shot": zs,
            "few_shot": fs,
            "rag": rg,
        })

    return results