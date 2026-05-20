"""
data_loader.py
--------------
Loads per-person Q&A datasets, computes a style fingerprint,
and returns train/test splits ready for the pipeline.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    id: int
    question: str
    answer: str


@dataclass
class PersonData:
    name: str
    train: List[QAPair]
    test: List[QAPair]
    style_fingerprint: Dict  # computed style stats from training answers


# ── Style Fingerprint ──────────────────────────────────────────────────────────

def compute_style_fingerprint(pairs: List[QAPair]) -> Dict:
    """
    Computes lightweight style statistics from a person's answers.
    Used to build the Zero-Shot style description prompt.

    Returns dict with:
        avg_words        - average words per answer
        avg_sentences    - average sentences per answer
        uses_hedging     - bool: uses phrases like 'I think', 'I believe'
        uses_examples    - bool: uses 'like', 'such as', 'for example'
        starts_with_i    - fraction of answers starting with 'I'
        common_openers   - top 3 opening words
    """
    answers = [p.answer for p in pairs]

    word_counts = [len(a.split()) for a in answers]
    sentence_counts = [a.count('.') + a.count('!') + a.count('?') for a in answers]

    hedging_phrases = ["i think", "i believe", "i feel", "in my opinion", "i guess"]
    example_phrases = ["like", "such as", "for example", "for instance"]

    uses_hedging = any(
        phrase in a.lower() for a in answers for phrase in hedging_phrases
    )
    uses_examples = any(
        phrase in a.lower() for a in answers for phrase in example_phrases
    )

    starts_with_i = sum(1 for a in answers if a.strip().lower().startswith("i")) / len(answers)

    openers = [a.strip().split()[0].lower() for a in answers if a.strip()]
    top_openers = sorted(set(openers), key=lambda w: openers.count(w), reverse=True)[:3]

    return {
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "avg_sentences": round(sum(sentence_counts) / len(sentence_counts), 1),
        "uses_hedging": uses_hedging,
        "uses_examples": uses_examples,
        "starts_with_i": round(starts_with_i, 2),
        "common_openers": top_openers,
    }


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_person(
    json_path: str,
    name: str,
    test_size: int = 9,
    seed: int = 42
) -> PersonData:
    """
    Loads a person's Q&A JSON, shuffles with fixed seed,
    splits into train/test, and computes their style fingerprint.

    Args:
        json_path  - path to the person's JSON file
        name       - person's display name (used in prompts)
        test_size  - number of pairs to hold out for evaluation (default 9 → 80/20)
        seed       - random seed for reproducibility

    Returns:
        PersonData with train, test, and style_fingerprint
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(path, "r") as f:
        raw = json.load(f)

    pairs = [QAPair(id=item["id"], question=item["question"], answer=item["answer"])
             for item in raw]

    random.seed(seed)
    random.shuffle(pairs)

    test = pairs[:test_size]
    train = pairs[test_size:]

    fingerprint = compute_style_fingerprint(train)

    print(f"Loaded {name}: {len(train)} train | {len(test)} test")
    print(f"  Style fingerprint: {fingerprint}\n")

    return PersonData(name=name, train=train, test=test, style_fingerprint=fingerprint)


def load_all(data_dir: str = "data/raw", seed: int = 42) -> Dict[str, PersonData]:
    """
    Loads all person datasets from data/raw/.
    Expects files named <person_name>.json.

    Returns dict: { "khushi": PersonData, "madhumitha": PersonData, ... }
    """
    data_dir = Path(data_dir)
    people = {}

    for json_file in sorted(data_dir.glob("*.json")):
        name = json_file.stem
        people[name] = load_person(str(json_file), name=name.capitalize(), seed=seed)

    if not people:
        raise ValueError(f"No JSON files found in {data_dir}")

    return people


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    people = load_all("data/raw")
    for name, person in people.items():
        print(f"--- {person.name} ---")
        print(f"  Train: {len(person.train)} | Test: {len(person.test)}")
        print(f"  Fingerprint: {person.style_fingerprint}")
        print(f"  Sample train Q: {person.train[0].question}")
        print(f"  Sample train A: {person.train[0].answer[:80]}...")
        print()
