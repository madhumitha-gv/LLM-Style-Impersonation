# LLM Style Impersonation & Style Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-LLaMA%203.1%208B-orange)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Problem

Can a large language model capture a real person's communication style — and which prompting strategy does it best?

This project investigates two tasks:
1. **Impersonation** — given a new question, answer it in a specific person's voice
2. **Style Transfer** — rewrite neutral text to match a specific person's writing style

Three prompting strategies are compared: **Zero-Shot**, **Few-Shot**, and **RAG** (Retrieval-Augmented Generation).

## Approach

**Data:** 45 conversational Q&A pairs per person (2 people). 80/20 train/test split.

**Model:** LLaMA 3.1 8B Instruct, 4-bit quantized via bitsandbytes.

**Strategies:**
| Strategy | Context Given to Model |
|---|---|
| Zero-Shot | Style description derived from answer statistics |
| Few-Shot | 5 randomly sampled training Q&A examples |
| RAG | Top-5 semantically similar Q&A examples (FAISS + Sentence-BERT) |

**Evaluation:**
| Metric | Task 1 | Task 2 |
|---|---|---|
| BLEU | ✓ | — |
| ROUGE-L | ✓ | — |
| Style Similarity (Sentence-BERT) | ✓ | ✓ |
| Style Classifier Accuracy | ✓ | ✓ |

## Results

*(Populated after running experiments)*

## Repository Structure

```
llm-style-impersonation/
├── data/
│   ├── raw/                  # Per-person Q&A JSON files
│   └── processed/            # Train/test splits (auto-generated)
├── src/
│   ├── data_loader.py        # Load datasets, compute style fingerprints
│   ├── embeddings.py         # Sentence-BERT + FAISS index for RAG
│   ├── prompts.py            # Prompt builders for all strategies
│   ├── generator.py          # LLaMA 3.1 8B inference
│   ├── evaluator.py          # BLEU, ROUGE-L, style similarity, classifier
│   └── run_experiments.py    # Main pipeline entry point
├── notebooks/
│   └── experiments.ipynb     # Colab-ready notebook (clone → run)
├── results/                  # Generated after running experiments
├── requirements.txt
└── README.md
```

## How to Run

**On Google Colab (recommended):**

Open `notebooks/experiments.ipynb` and run cells top to bottom.  
Requires a HuggingFace account with LLaMA 3.1 access approved.

**Locally (requires GPU):**

```bash
git clone https://github.com/madhumitha-gv/llm-style-impersonation.git
cd llm-style-impersonation
pip install -r requirements.txt
huggingface-cli login
python src/run_experiments.py
```

## Tech Stack

- **Model:** LLaMA 3.1 8B Instruct (Meta, via HuggingFace)
- **Quantization:** bitsandbytes 4-bit NF4
- **Embeddings:** Sentence-BERT `all-MiniLM-L6-v2`
- **Vector Search:** FAISS
- **Evaluation:** NLTK, rouge-score, scikit-learn
