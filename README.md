# LLM Style Impersonation & Style Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-LLaMA%203.1%208B-orange)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Question

> **Can a large language model capture a specific person's communication style — and which prompting strategy best achieves this?**

When we interact with AI systems, they generate generic, averaged responses. But real people have distinct voices — some are verbose and hedging, others terse and direct. This project asks: given a small set of real Q&A examples from a person, can we prompt an LLM to answer new questions *as that person*, or rewrite text *in their voice*?

This matters for personalized AI assistants, writing tools, and understanding the limits and risks of LLM impersonation.

---

## Tasks

**Task 1 — Impersonation:** Given a new question, generate an answer in a specific person's voice.

**Task 2 — Style Transfer:** Given a neutral piece of text, rewrite it to sound like it was written by a specific person.

Both tasks are evaluated across **6 prompting strategies** on **2 people with distinctly different writing styles**.

---

## Dataset

45 conversational Q&A pairs collected per person covering personal preferences, opinions on technology, lifestyle, and work. Topics include: daily routine, coding, AI, travel, success, failure, and more.

- **Person K** — conversational, structured, balanced opinions, hedged language ("I think", "I believe"), averages 26 words per answer
- **Person M** — direct, terse, opinionated, minimal hedging, averages 9 words per answer

**Split:** 80% train (36 pairs), 20% test (9 pairs) per person.

The style contrast between the two people is intentional — the style classifier confirms 88.9% separability, meaning their writing styles are clearly distinguishable.

---

## Strategies Compared

Six conditions tested, representing a progression from no context to increasingly targeted context:

| # | Strategy | Description |
|---|---|---|
| 1 | **Zero-Shot** | Style description only — derived from answer statistics (avg length, hedging frequency, sentence openers) |
| 2 | **Few-Shot** | 5 randomly sampled training Q&A examples provided as demonstrations |
| 3 | **RAG** | Top-5 semantically similar Q&A pairs retrieved via FAISS + Sentence-BERT cosine similarity |
| 4 | **RAG + Reranking** | Fetch 15 candidates, rerank by combined semantic relevance (50%) + style centroid similarity (50%) |
| 5 | **RAG + Style-Biased** | Style centroid prioritized over topic relevance (30% semantic, 70% style) — retrieves stylistically pure examples |
| 6 | **Contrastive Decoding** | Token-level: amplifies tokens that are more likely given the person context vs. a neutral baseline prompt |

---

## Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)
Measures n-gram word overlap between the generated answer and the ground truth answer. Counts how many 1–4 word sequences in the generated text also appear in the reference. Uses smoothing to handle short sequences.

- **Range:** 0–1 (higher = more word-for-word similarity)
- **Why it's low here:** The model generates new text in someone's style — it doesn't copy their exact words. Low BLEU is expected and does not indicate failure. It would only be high if the model memorized answers verbatim.
- **Used for:** Task 1 (Impersonation) only — Task 2 has no ground truth to compare against.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
Measures the longest common subsequence (LCS) between generated and reference text — words that appear in the same order, not necessarily consecutively. Reports F1 score balancing precision and recall.

- **Range:** 0–1 (higher = more structural and content overlap)
- **More lenient than BLEU** — captures partial matches and shared phrasing patterns.
- **Used for:** Task 1 only.

### Style Similarity (Sentence-BERT Cosine Similarity)
Encodes both texts into 384-dimensional semantic vectors using Sentence-BERT (`all-MiniLM-L6-v2`) and computes cosine similarity. Captures tone, phrasing patterns, and semantic closeness — not just word overlap.

- **Range:** -1 to 1 (higher = more similar in meaning and tone)
- **Most meaningful metric for style** — measures whether the generated answer *feels* like the person, not just whether it uses the same words.
- **Task 1:** Generated answer vs. ground truth answer
- **Task 2:** Generated rewrite vs. person's style centroid (mean of all training answer embeddings)

### Style Classifier Accuracy
A logistic regression classifier trained on Sentence-BERT embeddings to distinguish between the two people's writing. Evaluated with 5-fold cross-validation.

- **Baseline:** 50% (random chance with 2 people)
- **> 80%** = styles are clearly distinguishable
- **< 60%** = styles are very similar, impersonation is harder to evaluate

Also used to check whether generated outputs are classified to the correct person — a high-quality impersonation should "fool" the classifier.

---

## Results

### Task 1 — Impersonation

**Person K** (conversational, hedged style):

| Strategy | BLEU | ROUGE-L | Style Sim |
|---|---|---|---|
| Zero-Shot | 0.009 | 0.117 | 0.630 |
| Few-Shot | 0.009 | 0.143 | 0.683 |
| RAG | 0.012 | 0.148 | 0.725 |
| RAG + Reranking | — | — | — |
| RAG + Style-Biased | — | — | — |
| Contrastive Decoding | — | — | — |

**Person M** (direct, terse style):

| Strategy | BLEU | ROUGE-L | Style Sim |
|---|---|---|---|
| Zero-Shot | 0.003 | 0.049 | 0.404 |
| Few-Shot | 0.004 | 0.059 | 0.384 |
| RAG | 0.004 | 0.072 | 0.411 |
| RAG + Reranking | — | — | — |
| RAG + Style-Biased | — | — | — |
| Contrastive Decoding | — | — | — |

*Extended strategy results (RAG+Reranking, Style-Biased, Contrastive) to be updated after full experiment run.*

### Style Classifier

| Metric | Value |
|---|---|
| 5-fold CV accuracy | 88.9% |
| K RAG outputs classified correctly | 100% |
| M RAG outputs classified correctly | 0% |

### Key Findings

**RAG consistently outperforms Few-Shot and Zero-Shot for conversational styles.** For Person K, style similarity improves from 0.630 (Zero-Shot) → 0.683 (Few-Shot) → 0.725 (RAG), showing a clear benefit of targeted context retrieval.

**Prompting strategies struggle with terse, minimalist styles.** For Person M, all three strategies score between 0.384–0.411 with no clear winner. The model has a verbosity bias — it generates elaborate answers even when the target style is minimal.

**The model defaults to Person K's style when impersonating Person M.** M's RAG outputs are classified as K by the style classifier 100% of the time, suggesting the model cannot fully suppress its tendency toward verbose, hedged generation.

**Style classifier at 88.9% confirms the two styles are clearly separable** — making this a meaningful comparison. The impersonation difficulty is a genuine model limitation, not a dataset artifact.

---

## Repository Structure

```
LLM-Style-Impersonation/
├── data/
│   └── raw/                   # Per-person Q&A JSON files (k.json, m.json)
├── src/
│   ├── data_loader.py         # Load datasets, train/test split, style fingerprint
│   ├── embeddings.py          # Sentence-BERT + FAISS (base, reranked, style-biased)
│   ├── prompts.py             # Prompt builders for all 6 strategies
│   ├── generator.py           # LLaMA 3.1 8B / Mistral 7B inference
│   ├── contrastive.py         # Contrastive decoding implementation
│   ├── evaluator.py           # BLEU, ROUGE-L, style similarity, classifier
│   └── run_experiments.py     # Main pipeline — runs all 6 conditions
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

!python src/run_experiments.py
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
```

---

## Tech Stack

- **Model:** LLaMA 3.1 8B Instruct (Meta) — 4-bit NF4 quantization via bitsandbytes
- **CPU Fallback:** Mistral 7B Instruct v0.3
- **Embeddings:** Sentence-BERT `all-MiniLM-L6-v2`
- **Vector Search:** FAISS (IndexFlatIP — inner product on normalized vectors)
- **Evaluation:** NLTK (BLEU), rouge-score (ROUGE-L), scikit-learn (classifier)
- **Platform:** Google Colab Pro (A100 GPU)
