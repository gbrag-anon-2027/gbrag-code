"""GraphRAG-compatible answer metrics for all ball experiments.

The implementation below is intentionally kept byte-for-byte equivalent in
behaviour to ``baseline/GraphRAG/graphrag/metrics.py``.  GraphRAG is the
comparison standard for this project, so the public ``exact_match``,
``token_f1`` and ``best_em_f1`` functions must not add subset, acronym,
similarity, or other semantic matching rules.

The original ball evaluator's permissive matcher remains in each evaluator as
``compute_relaxed_accuracy``/``compute_relaxed_token_f1`` and is reported only
as an auxiliary diagnostic (``relaxed_accuracy`` and ``relaxed_f1``).
"""

import collections
import re
import string
from typing import Any, Dict, Iterable, List


def normalize_answer(value: Any) -> str:
    """Normalize text exactly as GraphRAG's standard metric does."""

    text = str(value or "").lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: Any, gold: Any) -> float:
    pred = normalize_answer(prediction).split()
    target = normalize_answer(gold).split()
    if not pred or not target:
        return float(pred == target)
    overlap = sum((collections.Counter(pred) & collections.Counter(target)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def ensure_answer_list(value: Any) -> List[str]:
    """Match GraphRAG benchmark._answers for strings, lists and nulls."""

    return [str(x) for x in value] if isinstance(value, list) else [str(value or "")]


def best_em_f1(prediction: Any, gold_answers: Iterable[Any]) -> Dict[str, Any]:
    """Return the best GraphRAG EM/F1 pair over all gold answers."""

    answers = [str(x) for x in gold_answers] or [""]
    em, f1, answer = max(
        ((exact_match(prediction, answer), token_f1(prediction, answer), answer)
         for answer in answers),
        key=lambda item: (item[0], item[1]),
    )
    return {"em": em, "f1": f1, "matched_gold_answer": answer}
