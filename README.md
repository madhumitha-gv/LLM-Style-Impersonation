# LLM Style Impersonation & Style Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-LLaMA%203.1%208B-orange)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Question

> ** Can a large language model capture a specific person's communication style ? if yes, which prompting strategy best achieves this?**

When we interact with AI systems, they generate generic, averaged responses. But real people have distinct voices — some are verbose and hedging, others terse and direct. This project asks: given a small set of real Q&A examples from a person, can we prompt an LLM to answer new questions *as that person*, or rewrite text *in their voice*?

This matters for personalized AI assistants, writing tools, and understanding the limits and risks of LLM impersonation.

---

## Tasks

**Task 1 — Impersonation:** Given a new question, generate an answer in a specific person's voice.

**Task 2 — Style Transfer:** Given a neutral piece of text, rewrite it to sound like it was written by a specific person.

Both tasks are evaluated across **7 prompting strategies** on **2 people with distinctly different writing styles**.

---

## Dataset

45 conversational Q&A pairs collected per person covering personal preferences, opinions on technology, lifestyle, and work. Topics include: daily routine, coding, AI, travel, success, failure, and more.

- **Person K** — conversational, structured, balanced opinions, hedged language ("I think", "I believe"), averages 26 words per answer
- **Person M** — direct, terse, opinionated, minimal hedging, averages 9 words per answer

**Split:** 80% train (36 pairs), 20% test (9 pairs) per person.

The style contrast between the two people is intentional — the style classifier confirms **88.9% separability**, meaning their writing styles are clearly distinguishable.

---

## Strategies Compared

Seven conditions tested, representing a progression from no context to increasingly targeted context:

| # | Strategy | Description |
|---|---|---|
| 1 | **Zero-Shot** | Style description only — derived from answer statistics (avg length, hedging frequency, sentence openers) |
| 2 | **Few-Shot** | 5 randomly sampled training Q&A examples provided as demonstrations |
| 3 | **RAG** | Top-5 semantically similar Q&A pairs retrieved via FAISS + Sentence-BERT cosine similarity |
| 4 | **RAG + Reranking** | Fetch 15 candidates, rerank by combined semantic relevance (50%) + style centroid similarity (50%) |
| 5 | **RAG + Style-Biased** | Style centroid prioritized over topic relevance (30% semantic, 70% style) — retrieves stylistically pure examples |
| 6 | **Contrastive Decoding** | Token-level: amplifies tokens more likely given person context vs. a neutral baseline prompt |
| 7 | **RAG + Constrained** | RAG with per-person token budget (1.5x avg word count) — addresses verbosity bias for terse styles |

---

## Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)
Measures n-gram word overlap between the generated answer and the ground truth answer. Counts how many 1–4 word sequences in the generated text also appear in the reference. Uses smoothing to handle short sequences.

- **Range:** 0–1 (higher = more word-for-word similarity)
- **Why it's low here:** The model generates new text in someone's style — it doesn't copy their exact words. Low BLEU is expected and does not indicate failure.
- **Used for:** Task 1 (Impersonation) only.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
Measures the longest common subsequence (LCS) between generated and reference text — words that appear in the same order, not necessarily consecutively. Reports F1 score balancing precision and recall.

- **Range:** 0–1 (higher = more structural and content overlap)
- **More lenient than BLEU** — captures partial matches and shared phrasing patterns.
- **Used for:** Task 1 only.

### Style Similarity (Sentence-BERT Cosine Similarity)
Encodes generated text into 384-dimensional semantic vectors using Sentence-BERT (`all-MiniLM-L6-v2`) and computes cosine similarity against the person's **style centroid** — the mean of all 36 training answer embeddings. This measures how close the generated output is to the person's overall writing voice, not just one specific answer.

- **Range:** -1 to 1 (higher = closer to the person's style)
- **Most meaningful metric for style** — captures tone, phrasing patterns, and semantic closeness.
- **Used for:** Both Task 1 and Task 2.

### Style Classifier Accuracy
A logistic regression classifier trained on Sentence-BERT embeddings to distinguish between the two people's writing. Evaluated with 5-fold cross-validation on real answers, then applied to generated outputs.

- **Baseline:** 50% (random chance with 2 people)
- **> 80%** = styles clearly distinguishable
- **Classifier accuracy on generated outputs** = how often the model correctly imitates the target person's style

---

## Results

### Style Classifier on Real Data
**5-fold CV accuracy: 88.9%** — the two people's styles are clearly distinguishable, making this a meaningful evaluation.

---

### Task 1 — Impersonation

**Person K** (conversational, hedged style):

| Strategy | BLEU | ROUGE-L | Style Sim | Classifier |
|---|---|---|---|---|
| Zero-Shot | 0.009 | 0.123 | 0.359 | 100% |
| Few-Shot | 0.016 | 0.136 | 0.321 | 100% |
| RAG | 0.014 | 0.152 | 0.343 | 100% |
| RAG + Reranking | 0.019 | 0.155 | 0.351 | 100% |
| **RAG + Style-Biased** | 0.013 | 0.152 | **0.413** | 100% |
| Contrastive Decoding | 0.017 | 0.147 | 0.350 | 100% |
| RAG + Constrained | — | — | 0.364 | 100% |

**Person M** (direct, terse style):

| Strategy | BLEU | ROUGE-L | Style Sim | Classifier |
|---|---|---|---|---|
| Zero-Shot | 0.004 | 0.052 | 0.244 | 0% |
| Few-Shot | 0.005 | 0.066 | 0.237 | 0% |
| RAG | 0.005 | 0.070 | 0.251 | 0% |
| RAG + Reranking | 0.005 | 0.064 | 0.258 | 0% |
| RAG + Style-Biased | 0.004 | 0.057 | 0.251 | 0% |
| Contrastive Decoding | 0.007 | 0.101 | 0.234 | 0% |
| **RAG + Constrained** | — | — | **0.292** | **11%** |

---

### Task 2 — Style Transfer

**Person K:**

| Strategy | Style Sim | Classifier |
|---|---|---|
| Zero-Shot | 0.359 | 100% |
| Few-Shot | 0.381 | 100% |
| RAG | 0.364 | 100% |
| RAG + Reranking | 0.387 | 100% |
| **RAG + Style-Biased** | **0.411** | 100% |
| Contrastive Decoding | 0.373 | 100% |
| RAG + Constrained | 0.382 | 100% |

**Person M:**

| Strategy | Style Sim | Classifier |
|---|---|---|
| Zero-Shot | 0.282 | 0% |
| Few-Shot | 0.304 | 0% |
| RAG | 0.312 | 0% |
| RAG + Reranking | 0.289 | 0% |
| RAG + Style-Biased | 0.290 | 0% |
| Contrastive Decoding | 0.286 | 0% |
| **RAG + Constrained** | **0.348** | **20%** |

---

### Best Strategy Per Person

| Person | Best Strategy | Style Sim | Classifier |
|---|---|---|---|
| K | RAG + Style-Biased | 0.413 | 100% |
| M | RAG + Constrained* | 0.292 | 11–20% |

*Token budget = 1.5x average word count (max 20 tokens for M)

---

## Key Findings

**1. RAG + Style-Biased retrieval is the best strategy for conversational styles.**
For Person K, prioritizing stylistically representative examples (70% style weight, 30% semantic) outperforms all other strategies in both impersonation (0.413) and style transfer (0.411). Retrieving examples that sound like the person matters more than retrieving topically relevant ones.

**2. Standard prompting completely fails for terse, minimalist styles.**
For Person M, all 6 standard strategies score between 0.234–0.258 on style similarity with 0% classifier accuracy across both tasks. The model generates verbose, hedged outputs regardless of the target style — a fundamental verbosity bias in instruction-tuned LLMs.

**3. Length constraint partially recovers style accuracy for terse writers.**
By forcing generation within the person's natural token budget (RAG + Constrained), classifier accuracy improves from 0% to 11% on impersonation and 20% on style transfer for Person M. This confirms that the verbosity bias is partially length-driven — shorter outputs are more likely to be classified correctly even if phrasing remains non-terse.

**4. Length constraint hurts conversational styles.**
For Person K, constraining to 39 tokens (1.5x avg) produces incomplete answers that score lower than unconstrained RAG-Style (0.364 vs 0.413). Style-appropriate length is a person-specific property — one-size constraints do not generalize.

**5. More context does not always help.**
For Person K, Few-Shot (0.321) performs worse than Zero-Shot (0.359) on style similarity. Random example selection in Few-Shot introduces noise — the 5 sampled examples may not represent the person's style well. RAG-Style addresses this by selecting stylistically consistent examples rather than random ones.

**6. Contrastive decoding adds compute cost with no consistent gain.**
With alpha=0.1, contrastive decoding matches or slightly underperforms base RAG across both tasks and people. The amplification signal may be too weak, or the neutral amateur prompt is too similar to the expert prompt to generate meaningful contrast.

**7. The verbosity bias is a weight-level problem, not a prompting problem.**
All 7 prompting strategies — including length constraint — fail to fully impersonate Person M's terse style. The model's training on verbose human feedback data creates a persistent generation prior that prompting alone cannot fully override. Weight-level intervention (e.g. LoRA fine-tuning on terse examples) would be the natural next step.

---

## Repository Structure

```
LLM-Style-Impersonation/
├── data/
│   └── raw/                   # Per-person Q&A JSON files (k.json, m.json)
├── src/
│   ├── data_loader.py         # Load datasets, train/test split, style fingerprint
│   ├── embeddings.py          # Sentence-BERT + FAISS (base, reranked, style-biased)
│   ├── prompts.py             # Prompt builders for all strategies
│   ├── generator.py           # LLaMA 3.1 8B / Mistral 7B inference
│   ├── contrastive.py         # Contrastive decoding implementation
│   ├── evaluator.py           # BLEU, ROUGE-L, style similarity, classifier
│   ├── run_experiments.py     # Main pipeline — runs all 6 standard conditions
│   └── run_constrained.py     # RAG + Constrained condition (7th strategy)
├── notebooks/
│   └── experiments.ipynb      # Colab-ready notebook (clone → run)
├── results/                   # CSVs and JSONs generated after experiments
├── requirements.txt
└── README.md
```

---

## How to Run

**On Google Colab (recommended — requires GPU):**

```python
!git clone https://github.com/madhumitha-gv/LLM-Style-Impersonation.git
%cd LLM-Style-Impersonation
!pip install -r requirements.txt -q

from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))

# Run all 6 standard conditions
!python src/run_experiments.py

# Run 7th condition (length-constrained)
!python src/run_constrained.py
```

Requires a HuggingFace account with LLaMA 3.1 8B access approved at:
`huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`

On CPU (no GPU), falls back automatically to Mistral 7B Instruct.

**Locally:**

```bash
git clone https://github.com/madhumitha-gv/LLM-Style-Impersonation.git
cd LLM-Style-Impersonation
pip install -r requirements.txt
huggingface-cli login
python src/run_experiments.py
python src/run_constrained.py
```

---

## Tech Stack

- **Model:** LLaMA 3.1 8B Instruct (Meta) — 4-bit NF4 quantization via bitsandbytes
- **CPU Fallback:** Mistral 7B Instruct v0.3
- **Embeddings:** Sentence-BERT `all-MiniLM-L6-v2`
- **Vector Search:** FAISS (IndexFlatIP — inner product on normalized vectors)
- **Evaluation:** NLTK (BLEU), rouge-score (ROUGE-L), scikit-learn (classifier)
- **Platform:** Google Colab Pro (A100 GPU)
