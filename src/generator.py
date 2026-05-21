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

GPU_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CPU_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

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

def load_model(model_id: str = None):
    """
    Loads the appropriate model based on hardware availability.

    GPU available : LLaMA 3.1 8B Instruct with 4-bit quantization.
                    Requires HuggingFace token for gated model access.
                    Set via: huggingface-cli login
                          or: export HUGGINGFACE_TOKEN=your_token_here

    CPU only      : Falls back to Mistral-7B-Instruct-v0.3 without
                    quantization. Slower but fully reproducible locally.

    Args:
        model_id - optional override; defaults to GPU_MODEL_ID on CUDA,
                   CPU_MODEL_ID on CPU.

    Returns:
        (tokenizer, model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_id is None:
        model_id = GPU_MODEL_ID if device == "cuda" else CPU_MODEL_ID

    print(f"Device : {device.upper()}")
    print(f"Model  : {model_id}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=QUANT_CONFIG,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # CPU fallback — no quantization, float32 for numerical stability
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    model.eval()
    print("Model ready.\n")
    return tokenizer, model, device


# ── Single Generation ──────────────────────────────────────────────────────────

def generate(prompt: str, tokenizer, model, max_new_tokens: int = None) -> str:
    """
    Generates a single response for a given prompt.
    Strips the prompt tokens from the output, returning only new text.

    Args:
        prompt         - fully formatted Llama 3 / Mistral chat prompt
        tokenizer      - loaded tokenizer
        model          - loaded model
        max_new_tokens - optional token limit override; defaults to
                         GENERATION_CONFIG value (150). Pass a lower
                         value to constrain output to match a person's
                         natural writing length.

    Returns:
        Generated text (answer only, no prompt echo)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Override max_new_tokens if specified, otherwise use default config
    config = {**GENERATION_CONFIG}
    if max_new_tokens is not None:
        config["max_new_tokens"] = max_new_tokens

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **config,
            pad_token_id=tokenizer.eos_token_id,
        )

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
            "question":     str,
            "ground_truth": str,
            "zero_shot":    str,
            "few_shot":     str,
            "rag":          str,
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
            "question":     pair.question,
            "ground_truth": pair.answer,
            "zero_shot":    zs,
            "few_shot":     fs,
            "rag":          rg,
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
            "topic":        str,
            "neutral_text": str,
            "zero_shot":    str,
            "few_shot":     str,
            "rag":          str,
        }
    """
    results = []
    print(f"\nRunning style transfer for {person.name} ({len(NEUTRAL_TEXTS)} texts)...")

    for item in tqdm(NEUTRAL_TEXTS):
        neutral_text = item["text"]
        retrieved = index.retrieve(neutral_text, top_k=top_k)

        zs = generate(style_transfer_zero_shot(person, neutral_text), tokenizer, model)
        fs = generate(style_transfer_few_shot(person, neutral_text), tokenizer, model)
        rg = generate(style_transfer_rag(person, neutral_text, retrieved), tokenizer, model)

        results.append({
            "topic":        item["topic"],
            "neutral_text": neutral_text,
            "zero_shot":    zs,
            "few_shot":     fs,
            "rag":          rg,
        })

    return results