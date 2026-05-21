"""
prompts.py
----------
Builds prompts for all four experimental conditions:
  - Zero-Shot impersonation
  - Few-Shot impersonation
  - RAG impersonation
  - Style Transfer (rewrite neutral text in person's voice)

All prompts use the Llama 3 instruct chat template format.
"""

import random
from typing import List, Tuple

from data_loader import PersonData, QAPair


# ── Style Description Builder ──────────────────────────────────────────────────

def build_style_description(person: PersonData) -> str:
    """
    Converts a person's style fingerprint into a natural language
    description for use in system prompts.
    """
    fp = person.style_fingerprint
    parts = []

    # Sentence length
    if fp["avg_words"] < 20:
        parts.append("You write in short, concise sentences.")
    elif fp["avg_words"] < 35:
        parts.append("You write in moderately detailed sentences.")
    else:
        parts.append("You write in detailed, elaborate sentences.")

    # Hedging
    if fp["uses_hedging"]:
        parts.append("You often use phrases like 'I think' or 'I believe' to soften opinions.")
    else:
        parts.append("You state opinions directly without hedging.")

    # Examples
    if fp["uses_examples"]:
        parts.append("You frequently support points with concrete examples.")

    # First-person
    if fp["starts_with_i"] > 0.5:
        parts.append("You often begin answers with 'I'.")

    return " ".join(parts)


# ── Llama 3 Chat Template ──────────────────────────────────────────────────────

def format_llama3_prompt(system: str, user: str) -> str:
    """
    Formats a prompt using Llama 3's instruct chat template.

    Format:
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ── Task 1: Impersonation Prompts ──────────────────────────────────────────────

def zero_shot_prompt(person: PersonData, question: str) -> str:
    """
    Zero-Shot: style description only, no examples.
    The model must infer the person's voice from the description alone.
    """
    style_desc = build_style_description(person)

    system = (
        f"You are {person.name}. Answer questions exactly as {person.name} would, "
        f"matching their personal communication style.\n\n"
        f"Style guide: {style_desc}\n\n"
        f"Keep your answer natural and in first person. Do not explain your reasoning."
    )

    return format_llama3_prompt(system=system, user=question)


def few_shot_prompt(
    person: PersonData,
    question: str,
    n_examples: int = 5,
    seed: int = 42
) -> str:
    """
    Few-Shot: random sample of n training Q&A pairs as examples.
    The model sees real examples of the person's writing before answering.
    """
    random.seed(seed)
    examples = random.sample(person.train, min(n_examples, len(person.train)))

    style_desc = build_style_description(person)

    examples_text = "\n\n".join(
        f"Q: {ex.question}\nA: {ex.answer}" for ex in examples
    )

    system = (
        f"You are {person.name}. Answer questions exactly as {person.name} would.\n\n"
        f"Style guide: {style_desc}\n\n"
        f"Here are real examples of how {person.name} answers questions:\n\n"
        f"{examples_text}\n\n"
        f"Now answer the next question in the same style. "
        f"Do not copy the examples — answer naturally as {person.name}."
    )

    return format_llama3_prompt(system=system, user=question)


def rag_prompt(
    person: PersonData,
    question: str,
    retrieved: List[Tuple[QAPair, float]]
) -> str:
    """
    RAG: top-k semantically similar Q&A pairs retrieved for this specific question.
    More targeted than Few-Shot — examples are relevant to the current question.

    Args:
        person     - PersonData with name and fingerprint
        question   - the new question to answer
        retrieved  - output of StyleIndex.retrieve()
    """
    style_desc = build_style_description(person)

    examples_text = "\n\n".join(
        f"Q: {pair.question}\nA: {pair.answer}" for pair, _ in retrieved
    )

    system = (
        f"You are {person.name}. Answer questions exactly as {person.name} would.\n\n"
        f"Style guide: {style_desc}\n\n"
        f"Here are the most relevant examples of how {person.name} answers "
        f"similar questions:\n\n"
        f"{examples_text}\n\n"
        f"Answer the next question in {person.name}'s voice. "
        f"Do not copy the examples — answer naturally."
    )

    return format_llama3_prompt(system=system, user=question)


# ── Task 2: Style Transfer Prompts ─────────────────────────────────────────────

# Neutral source texts to rewrite — cover different topics
NEUTRAL_TEXTS = [
    {
        "id": 1,
        "topic": "remote work",
        "text": "Remote work has become increasingly common. It offers flexibility but also presents challenges related to communication and collaboration. Many organizations are now adopting hybrid models."
    },
    {
        "id": 2,
        "topic": "learning new skills",
        "text": "Acquiring new skills requires consistent effort and practice. Initial difficulty is normal and should not discourage learners. Over time, competence develops through repetition and feedback."
    },
    {
        "id": 3,
        "topic": "technology and society",
        "text": "Technology has transformed how people communicate and access information. While it provides significant benefits, there are also concerns about privacy and social isolation."
    },
    {
        "id": 4,
        "topic": "work-life balance",
        "text": "Maintaining a healthy work-life balance is important for long-term productivity and wellbeing. Setting clear boundaries between professional and personal time helps reduce burnout."
    },
    {
        "id": 5,
        "topic": "failure and resilience",
        "text": "Failure is an inevitable part of any significant endeavor. How individuals respond to setbacks often determines their long-term success. Resilience can be developed through experience."
    }
]


def style_transfer_zero_shot(person: PersonData, neutral_text: str) -> str:
    """
    Zero-Shot style transfer: rewrite neutral text in person's voice
    using only their style description.
    """
    style_desc = build_style_description(person)

    system = (
        f"You are a writing assistant that rewrites text to match a specific person's style.\n\n"
        f"Target person: {person.name}\n"
        f"Their style: {style_desc}\n\n"
        f"Rewrite the given text so it sounds like {person.name} wrote it personally. "
        f"Keep the core meaning but match their tone, sentence structure, and vocabulary. "
        f"Write in first person."
    )

    user = f"Rewrite this in {person.name}'s voice:\n\n{neutral_text}"

    return format_llama3_prompt(system=system, user=user)


def style_transfer_few_shot(
    person: PersonData,
    neutral_text: str,
    n_examples: int = 5,
    seed: int = 42
) -> str:
    """
    Few-Shot style transfer: same as zero-shot but with real writing examples.
    """
    random.seed(seed)
    examples = random.sample(person.train, min(n_examples, len(person.train)))

    style_desc = build_style_description(person)
    examples_text = "\n\n".join(
        f"Q: {ex.question}\nA: {ex.answer}" for ex in examples
    )

    system = (
        f"You are a writing assistant that rewrites text to match a specific person's style.\n\n"
        f"Target person: {person.name}\n"
        f"Their style: {style_desc}\n\n"
        f"Here are real examples of {person.name}'s writing:\n\n"
        f"{examples_text}\n\n"
        f"Rewrite the given text so it sounds exactly like {person.name} wrote it. "
        f"Match their tone, phrasing, and sentence length. Write in first person."
    )

    user = f"Rewrite this in {person.name}'s voice:\n\n{neutral_text}"

    return format_llama3_prompt(system=system, user=user)


def style_transfer_rag(
    person: PersonData,
    neutral_text: str,
    retrieved: List[Tuple[QAPair, float]]
) -> str:
    """
    RAG style transfer: retrieve topic-relevant examples, then rewrite.
    """
    style_desc = build_style_description(person)
    examples_text = "\n\n".join(
        f"Q: {pair.question}\nA: {pair.answer}" for pair, _ in retrieved
    )

    system = (
        f"You are a writing assistant that rewrites text to match a specific person's style.\n\n"
        f"Target person: {person.name}\n"
        f"Their style: {style_desc}\n\n"
        f"Here are the most relevant examples of {person.name}'s writing on this topic:\n\n"
        f"{examples_text}\n\n"
        f"Rewrite the given text so it sounds exactly like {person.name} wrote it. "
        f"Match their tone, phrasing, and sentence length. Write in first person."
    )

    user = f"Rewrite this in {person.name}'s voice:\n\n{neutral_text}"

    return format_llama3_prompt(system=system, user=user)


# ── Contrastive Decoding Prompts ───────────────────────────────────────────────

def neutral_prompt(question: str) -> str:
    """
    Baseline prompt with NO person-specific context.
    Used as the 'without context' side of contrastive decoding.
    The model answers generically — contrastive decoding amplifies
    what the person-specific prompt adds on top of this.
    """
    system = (
        "You are a helpful assistant. Answer the following question naturally "
        "and conversationally in first person."
    )
    return format_llama3_prompt(system=system, user=question)


def neutral_style_transfer_prompt(neutral_text: str) -> str:
    """
    Baseline style transfer prompt with NO person-specific context.
    Used as the 'without context' side of contrastive decoding.
    """
    system = (
        "You are a helpful assistant. Rewrite the following text in a natural, "
        "conversational first-person voice."
    )
    user = f"Rewrite this naturally:\n\n{neutral_text}"
    return format_llama3_prompt(system=system, user=user)