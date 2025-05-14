import logging
import os
from collections.abc import Callable
from copy import deepcopy

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("rlm_eval")
logger.setLevel("INFO")
handler = RichHandler(rich_tracebacks=True)
handler.setLevel("NOTSET")
handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
logger.handlers = []
logger.addHandler(handler)

console = Console()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


class optional_status:
    # context manager encapsulating console.status() with an additional argument to disable it
    def __init__(self, status, enabled=True, **kwargs):
        self.enabled = enabled
        self.status = console.status(status, **kwargs)

    def __enter__(self):
        if self.enabled:
            self.status.__enter__()
        return self.status

    def __exit__(self, *args):
        if self.enabled:
            self.status.__exit__(*args)


def merge_consecutive_roles(messages):
    """Merge consecutive messages with the same role.
    Some APIs don't allow sending multiple consecutive messages with the same role."""
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages


def clean_messages(messages):
    messages = deepcopy(messages)
    messages = merge_consecutive_roles(messages)
    for message in messages:
        message["content"] = message["content"].strip()
    return messages


def compute_bleu_score(prediction: str, reference: str) -> float:
    return sentence_bleu(  # type: ignore
        [reference.split()],
        prediction.split(),
        smoothing_function=SmoothingFunction().method4,
    )


def self_consistency(
    generations: list[str],
    pair_similarity_fn: Callable[[str, str], float] = compute_bleu_score,
    return_score=False,
    return_idx=False,
) -> str | tuple[str, float] | tuple[str, int] | tuple[str, float, int] | None:
    # Source: More Agents Is All You Need (https://arxiv.org/abs/2402.05120)
    # return the generation with the highest cumulative similarity score
    if len(generations) == 0:
        return None

    bleu_scores = [
        sum(pair_similarity_fn(generations[i], generations[j]) for j in range(len(generations)) if i != j)
        for i in range(len(generations))
    ]
    max_score = max(bleu_scores)
    idx = bleu_scores.index(max_score)
    normalized_score = max_score / (len(generations) - 1) if len(generations) > 1 else 1.0
    if return_score and return_idx:
        return generations[idx], normalized_score, idx
    if return_idx:
        return generations[idx], idx
    if return_score:
        return generations[idx], normalized_score
    return generations[idx]


def self_consistency_scores(
    generations: list[str],
    pair_similarity_fn: Callable[[str, str], float] = compute_bleu_score,
) -> list[float]:
    return [
        sum(pair_similarity_fn(generations[i], generations[j]) for j in range(len(generations)) if i != j)
        for i in range(len(generations))
    ]


def similarity_fn_wrapper(args):
    similarity_fn, idx, x = args[0], args[1], args[2]
    return idx, sum(similarity_fn(a, b) for b in x for a in x) / (len(x) ** 2)


def similarity_matrix_fn_wrapper(args):
    similarity_fn, idx, x = args[0], args[1], args[2]
    return idx, [[similarity_fn(a, b) for b in x] for a in x]


def extract_lean_codes(content: str) -> list[str]:
    """Extract Lean code snippets from a string"""
    lean_starts = ["```lean", "```lean4"]

    # If no lean code blocks found, return content as is
    if not any(start in content for start in lean_starts):
        return [content.strip()]

    lean_codes = []
    while any(start in content for start in lean_starts):
        # Find the earliest occurrence of any lean start marker
        start_indices = [(content.index(start), start) for start in lean_starts if start in content]
        idx_start, start_marker = min(start_indices, key=lambda x: (x[0], -len(x[1])))

        content = content[idx_start + len(start_marker) :]
        try:
            idx_end = content.index("```")
        except ValueError:
            idx_end = len(content)
        lean_codes.append(content[:idx_end].strip())
        content = content[idx_end + 3 :]
    return lean_codes


def generate_n_samples_sequence(max_n: int, sequence_type: str = "125") -> list[int]:
    """Generate a sequence of sample sizes up to max_n.

    Args:
        max_n: The maximum number of samples to generate.
        sequence_type: The type of sequence to generate. Options are:
            - "125": Generate the sequence 1, 2, 5, 10, 20, 50, 100, ... (default)
            - "pow2": Generate the sequence 1, 2, 4, 8, 16, 32, 64, ... (powers of 2)

    Returns:
        A list of integers representing the sample sizes.
    """
    if sequence_type == "125":
        repeating_seq = [1, 2, 5]
        sequence = []
        i = 1
        while i <= max_n:
            sequence.extend([i * r for r in repeating_seq if i * r <= max_n])
            i *= 10
    elif sequence_type == "pow2":
        sequence = []
        i = 1
        while i <= max_n:
            sequence.append(i)
            i *= 2
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}. Valid options are '125' or 'pow2'.")

    if len(sequence) > 0 and sequence[-1] != max_n:
        sequence.append(max_n)
    return sequence
