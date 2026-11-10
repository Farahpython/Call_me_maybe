import json
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model


NORMAL = 0
ESCAPE = 1
string_cache: dict[int, tuple[list[int], set[int], set[int]]] = {}
opening_cache: list[list[int]] = []


def choose_constrained_token(scores: list[float],
                             allowed_ids: list[int]
                             ) -> int:
    """Choose the highest-logit token after masking disallowed tokens."""
    if not allowed_ids:
        raise ValueError("No valid tokens are available for"
                         "constrained decoding.")

    logits = np.asarray(scores, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError("Token logits must be a one-dimensional sequence.")

    if any(token_id < 0 or token_id >= logits.size for
           token_id in allowed_ids
           ):
        raise ValueError("An allowed token ID is outside the logits range.")

    allowed_mask = np.zeros(logits.size, dtype=bool)
    allowed_mask[allowed_ids] = True
    logits[~allowed_mask] = -np.inf
    return int(np.argmax(logits))


def walk_string(text: str, state: int) -> int | None:
    """Feed text through the string rules, starting at `state`.
    Returns the state reached, or None if some character is not allowed.
    """
    for char in text:
        if state == NORMAL:
            if char == "\\":
                state = ESCAPE
            elif char == '"' or ord(char) < 0x20:
                return None
        else:
            if char in '"\\/bfnrt':
                state = NORMAL
            else:
                return None
    return state


def string_options(vocab: list[str],
                   state: int
                   ) -> tuple[list[int], set[int], set[int]]:
    """Return the valid vocabulary tokens for the current JSON"""
    """string state."""
    if state not in string_cache:
        usable, closing, ending = [], set(), set()
        for token_id, token in enumerate(vocab):
            if not token:
                continue
            if state == NORMAL and token[0] == '"':
                usable.append(token_id)
                closing.add(token_id)
            elif token[-1] == '"' and walk_string(token[:-1], state) == NORMAL:
                usable.append(token_id)
                ending.add(token_id)
            elif walk_string(token, state) is not None:
                usable.append(token_id)
        string_cache[state] = (usable, closing, ending)
    return string_cache[state]


def opening_options(vocab: list[str]) -> list[int]:
    """Tokens that can start a string value: an optional space, the opening
    quote, then ordinary characters (so a token like ' "/' is allowed)."""
    if not opening_cache:
        usable = []  # list of token IDs that can start a JSON string.
        for token_id, token in enumerate(vocab):
            body = token[1:] if token.startswith(" ") else token
            if (
                body.startswith('"')
                and walk_string(body[1:], NORMAL) == NORMAL
            ):
                usable.append(token_id)
        opening_cache.append(usable)
    return opening_cache[0]


def generate_string(model: Small_LLM_Model, vocab: list[str],
                    input_ids: list[int]
                    ) -> Any:
    """Generate a JSON-compatible string using constrained token selection."""
    scores = model.get_logits_from_input_ids(input_ids)
    first = choose_constrained_token(scores, opening_options(vocab))
    raw = vocab[first].lstrip(" ")[1:]
    input_ids.append(first)
    state = NORMAL
    for _ in range(100):
        scores = model.get_logits_from_input_ids(input_ids)
        usable, closing, ending = string_options(vocab, state)
        best = choose_constrained_token(scores, usable)
        if best in closing:
            input_ids += model.encode('"').flatten().tolist()
            break
        if best in ending:
            raw += vocab[best][:-1]
            input_ids.append(best)
            break
        raw += vocab[best]
        next_state = walk_string(vocab[best], state)
        if next_state is None:
            break
        state = next_state
        input_ids.append(best)
    try:
        value = json.loads('"' + raw + '"')
    except ValueError:
        value = raw
    return value.strip()
