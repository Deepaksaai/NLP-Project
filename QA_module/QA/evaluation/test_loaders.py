"""
Load raw (question, document, gold_answers, is_answerable) triples
from every QA test split we care about. Returns plain dicts — no
tokenization, no chunking — so the full inference pipeline can run
over the untouched input exactly as a user would feed it.

Every loader yields records of the form:

    {
      "example_id":   str,
      "source":       "cuad"|"coliee"|"ledgar"|"squad",
      "question":     str,
      "document":     str,             # full context
      "gold_answers": List[str],       # possibly empty
      "is_answerable":bool,
      "extras":       dict,            # e.g. clause title, label, ...
    }

Downstream code never touches HuggingFace again — it reads from the
list returned here. Use `limit` for smoke tests.
"""

from typing import Iterator, Dict, List, Optional


def load_cuad_test(limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split="test")
    except Exception:
        # Some mirrors only publish a validation split — treat it as test
        ds = load_dataset("theatticusproject/cuad-qa", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        texts  = ex["answers"].get("text") or []
        starts = ex["answers"].get("answer_start") or []
        is_ans = bool(texts) and bool(starts) and texts[0].strip() != ""
        out.append({
            "example_id":   f"cuad_{i}",
            "source":       "cuad",
            "question":     ex["question"],
            "document":     ex["context"],
            "gold_answers": [t for t in texts if t.strip()] if is_ans else [],
            "is_answerable": is_ans,
            "extras":       {"doc_title": ex.get("title", "")},
        })
    return out


def load_ledgar_test(limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    from QA.data.legal_templates import template_for

    ds = load_dataset("lex_glue", "ledgar", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        label_name = ds.features["label"].int2str(ex["label"])
        q = template_for(label_name)
        if q is None:
            continue
        out.append({
            "example_id":   f"ledgar_{i}",
            "source":       "ledgar",
            "question":     q,
            "document":     ex["text"],
            # LEDGAR answer == provision itself; treat first 200 words as gold
            "gold_answers": [" ".join(ex["text"].split()[:200])],
            "is_answerable": True,
            "extras":       {"label": label_name},
        })
    return out


def load_coliee_test(limit: Optional[int] = None) -> List[Dict]:
    """Reads a locally-placed data/coliee/test.json if present."""
    import os, json
    from QA.qa_config import QA_DATA_ROOT
    path = os.path.join(os.path.dirname(QA_DATA_ROOT), "coliee", "test.json")
    if not os.path.exists(path):
        print(f"[test_loaders] coliee test file missing ({path}) — returning []")
        return []
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if limit:
        items = items[:limit]
    out = []
    for i, ex in enumerate(items):
        out.append({
            "example_id":   f"coliee_{i}",
            "source":       "coliee",
            "question":     ex["question"],
            "document":     ex["context"],
            "gold_answers": [ex["answer_text"]] if ex.get("answer_text") else [],
            "is_answerable": bool(ex.get("is_answerable", False)),
            "extras":       {},
        })
    return out


def load_squad_holdout(limit: Optional[int] = 1000) -> List[Dict]:
    """Diagnostic general-domain holdout — first 1k SQuAD 2.0 validation."""
    from datasets import load_dataset
    ds = load_dataset("squad_v2", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        texts  = ex["answers"].get("text") or []
        starts = ex["answers"].get("answer_start") or []
        is_ans = bool(texts) and bool(starts) and texts[0].strip() != ""
        out.append({
            "example_id":   f"squad_{i}",
            "source":       "squad",
            "question":     ex["question"],
            "document":     ex["context"],
            "gold_answers": [t for t in texts if t.strip()] if is_ans else [],
            "is_answerable": is_ans,
            "extras":       {},
        })
    return out


# -------------------------------------------------------
def load_all(limit_per_source: Optional[int] = None) -> Dict[str, List[Dict]]:
    data = {}
    for name, fn in (
        ("cuad",   load_cuad_test),
        ("ledgar", load_ledgar_test),
        ("coliee", load_coliee_test),
        ("squad",  load_squad_holdout),
    ):
        try:
            data[name] = fn(limit=limit_per_source)
            print(f"[test_loaders] {name}: {len(data[name])} examples")
        except Exception as e:
            print(f"[test_loaders] {name} FAILED: {e}")
            data[name] = []
    return data
