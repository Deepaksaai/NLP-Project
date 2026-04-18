"""
Synthetic examples for the Stage 1 sanity check.

Produces 32 toy QA examples — 24 answerable + 8 unanswerable — and
runs them through build_features() so they're drop-in compatible
with Stage1QADataset. No network access required; tokenizer must
already exist from add_qa_tokens.py.

The only purpose of these examples is to verify that the training
loop can overfit a single fixed batch to F1 > 0.90. If it can't,
something in masking / span indexing / loss is broken.
"""

import os
import json
import random
from typing import List

from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens,
)
from QA.data.qa_dataset import build_features


# Simple templates: the answer appears verbatim in the context.
_ANSWERABLE_TEMPLATES = [
    ("What is the capital of {country}?",
     "The capital of {country} is {city}. It has been the political center for centuries.",
     "{city}"),
    ("Who wrote the novel {book}?",
     "The novel {book} was written by {author} and first published in 1925.",
     "{author}"),
    ("What color are most {animal}?",
     "Most {animal} are {color} with small darker markings on their back.",
     "{color}"),
    ("When did the {event} take place?",
     "The {event} took place in the year {year}, changing the region forever.",
     "{year}"),
    ("How many legs does a {creature} have?",
     "A {creature} has exactly {number} legs and uses them to walk across surfaces.",
     "{number}"),
    ("What element has the symbol {symbol}?",
     "The element with the symbol {symbol} is called {element} on the periodic table.",
     "{element}"),
]

_FILLERS = {
    "country":  [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Brazil", "Brasilia")],
    "book":     [("Ulysses", "Joyce"), ("Dune", "Herbert"), ("Beloved", "Morrison")],
    "animal":   [("foxes", "orange"), ("ravens", "black"), ("flamingos", "pink")],
    "event":    [("revolution", "1789"), ("treaty", "1648"), ("festival", "1969")],
    "creature": [("spider", "eight"), ("ant", "six"), ("dog", "four")],
    "symbol":   [("Fe", "iron"), ("Au", "gold"), ("Na", "sodium")],
}

# Unanswerable: question is about topic A, context is about topic B.
_UNANSWERABLE = [
    ("What is the tallest mountain in Africa?",
     "Guitars have six strings and are commonly used in folk music."),
    ("When was the telephone invented?",
     "Spinach is rich in iron and grows best in cool weather."),
    ("Who painted the Mona Lisa?",
     "The Olympic games are held every four years in a different city."),
    ("Where is the Eiffel Tower located?",
     "Honey bees communicate the location of flowers through a waggle dance."),
    ("What is the speed of light?",
     "Knitting patterns use abbreviations to save space and simplify instructions."),
    ("Who discovered penicillin?",
     "A hurricane gains strength as it passes over warm ocean waters."),
    ("How deep is the ocean?",
     "Sourdough bread is made by fermenting flour with wild yeast and bacteria."),
    ("What year did World War II end?",
     "The maple tree drops its seeds inside winged pods that spin as they fall."),
]


def _render_answerable(i: int):
    q_tpl, c_tpl, a_tpl = _ANSWERABLE_TEMPLATES[i % len(_ANSWERABLE_TEMPLATES)]
    # Pick the slot key the template uses
    key = None
    for k in _FILLERS:
        if "{" + k + "}" in q_tpl:
            key = k
            break
    pair = _FILLERS[key][(i // len(_ANSWERABLE_TEMPLATES)) % len(_FILLERS[key])]
    q = q_tpl.format(**{key: pair[0]})
    c = c_tpl.format(**{key: pair[0], **dict(zip(["city","author","color","year","number","element"], [pair[1]]*6))})
    a = a_tpl.format(**{"city": pair[1], "author": pair[1], "color": pair[1],
                        "year": pair[1], "number": pair[1], "element": pair[1]})
    start = c.find(a)
    return {
        "question": q, "context": c,
        "answer_text": a, "answer_start": start,
        "is_answerable": True, "domain": "general",
    }


def _render_unanswerable(i: int):
    q, c = _UNANSWERABLE[i % len(_UNANSWERABLE)]
    return {
        "question": q, "context": c,
        "answer_text": "", "answer_start": -1,
        "is_answerable": False, "domain": "general",
    }


def build_sanity_examples(n_answerable: int = 24, n_unanswerable: int = 8) -> List[dict]:
    meta = load_qa_special_tokens()
    tok = Tokenizer.from_file(QA_TOKENIZER_PATH)

    raw = [_render_answerable(i) for i in range(n_answerable)]
    raw += [_render_unanswerable(i) for i in range(n_unanswerable)]

    feats = []
    for r in raw:
        f = build_features(
            r, tok,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
        )
        if f is None:
            raise RuntimeError(f"build_features returned None for sanity example: {r!r}")
        feats.append(f)
    return feats


def write_sanity_file(path: str = None):
    if path is None:
        path = os.path.join(QA_DATA_ROOT, "sanity", "sanity.json")
    feats = build_sanity_examples()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feats, f)
    ans = sum(1 for x in feats if x["is_answerable"])
    print(f"Wrote {len(feats)} sanity examples ({ans} ans / {len(feats)-ans} unans) -> {path}")
    return path


if __name__ == "__main__":
    write_sanity_file()
