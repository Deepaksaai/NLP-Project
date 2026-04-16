"""
Stage 4 — All 6 evaluation metrics.

  Metric 1: ROUGE        (n-gram overlap)
  Metric 2: BERTScore    (semantic similarity via deberta-xlarge-mnli)
  Metric 3: Faithfulness (NLI-based — does summary contradict source?)
  Metric 4: Entity coverage (named entity preservation)
  Metric 5: Outcome preservation (rule-based legal outcome matching)
  Metric 6: Length analysis

Each metric is a standalone function returning a dict of scores.
The orchestrator script run_evaluation.py imports these and runs all 6.

External tools used here are PRETRAINED and used for evaluation only.
They never touch the trained summarizer's weights.
"""

import re
import json
import statistics
import numpy as np


# =========================================================
# Metric 1 — ROUGE
# =========================================================
def compute_rouge(generated, references):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Args:
        generated: dict {test_id: summary_string}
        references: dict {test_id: reference_string}
    Returns:
        dict with rouge1_f1, rouge2_f1, rougeL_f1, etc.
    """
    print("Computing ROUGE...")
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("  rouge_score not installed: pip install rouge-score")
        return {"error": "rouge_score not installed"}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    scores = {
        "rouge1_p": [], "rouge1_r": [], "rouge1_f": [],
        "rouge2_p": [], "rouge2_r": [], "rouge2_f": [],
        "rougeL_p": [], "rougeL_r": [], "rougeL_f": [],
    }

    for tid in references:
        ref = references[tid]
        gen = generated.get(tid, "")
        if not gen or not ref:
            continue
        s = scorer.score(ref, gen)
        scores["rouge1_p"].append(s["rouge1"].precision)
        scores["rouge1_r"].append(s["rouge1"].recall)
        scores["rouge1_f"].append(s["rouge1"].fmeasure)
        scores["rouge2_p"].append(s["rouge2"].precision)
        scores["rouge2_r"].append(s["rouge2"].recall)
        scores["rouge2_f"].append(s["rouge2"].fmeasure)
        scores["rougeL_p"].append(s["rougeL"].precision)
        scores["rougeL_r"].append(s["rougeL"].recall)
        scores["rougeL_f"].append(s["rougeL"].fmeasure)

    return {
        "rouge1_precision": float(np.mean(scores["rouge1_p"])),
        "rouge1_recall": float(np.mean(scores["rouge1_r"])),
        "rouge1_f1": float(np.mean(scores["rouge1_f"])),
        "rouge2_precision": float(np.mean(scores["rouge2_p"])),
        "rouge2_recall": float(np.mean(scores["rouge2_r"])),
        "rouge2_f1": float(np.mean(scores["rouge2_f"])),
        "rougeL_precision": float(np.mean(scores["rougeL_p"])),
        "rougeL_recall": float(np.mean(scores["rougeL_r"])),
        "rougeL_f1": float(np.mean(scores["rougeL_f"])),
        "n_evaluated": len(scores["rouge1_f"]),
    }


# =========================================================
# Metric 2 — BERTScore
# =========================================================
def compute_bertscore(generated, references, model_type="microsoft/deberta-xlarge-mnli",
                     batch_size=8):
    """Semantic similarity via contextual embeddings."""
    print("Computing BERTScore...")
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("  bert_score not installed: pip install bert-score")
        return {"error": "bert_score not installed"}

    ids = [tid for tid in references if generated.get(tid, "")]
    cands = [generated[tid] for tid in ids]
    refs = [references[tid] for tid in ids]

    if not cands:
        return {"error": "no valid pairs"}

    P, R, F1 = bert_score_fn(
        cands, refs, model_type=model_type, lang="en",
        batch_size=batch_size, verbose=True,
    )

    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
        "n_evaluated": len(cands),
        "model_type": model_type,
    }


# =========================================================
# Metric 3 — Faithfulness (NLI-based)
# =========================================================
def split_sentences(text):
    """Simple sentence splitter (avoid spacy dependency for this)."""
    protected = text
    abbrevs = ["Dr.", "Mr.", "Mrs.", "Inc.", "Corp.", "Ltd.", "LLC",
               "v.", "vs.", "et al.", "i.e.", "e.g.", "U.S.", "Sec.",
               "Co.", "No."]
    for a in abbrevs:
        protected = protected.replace(a, a.replace(".", "<DOT>"))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def compute_faithfulness(generated, sources, model_name="cross-encoder/nli-deberta-v3-base",
                        contradiction_threshold=0.7, max_source_chars=4000):
    """
    NLI-based faithfulness. For each summary sentence, check if the source
    document entails it. Scores: entailment / neutral / contradiction.
    """
    print("Computing faithfulness (NLI)...")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("  sentence_transformers not installed")
        return {"error": "sentence_transformers not installed"}

    nli = CrossEncoder(model_name)
    label_map = nli.config.id2label  # e.g. {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
    # Find label indices
    label_to_idx = {v.lower(): k for k, v in label_map.items()}
    ent_idx = label_to_idx.get("entailment", 1)
    con_idx = label_to_idx.get("contradiction", 0)

    per_summary_faithfulness = []
    n_contradicted_sentences = 0
    n_total_sentences = 0
    n_summaries_with_contradiction = 0
    n_summaries = 0

    for tid in generated:
        gen = generated[tid]
        src = sources.get(tid, "")
        if not gen or not src:
            continue
        n_summaries += 1

        # Truncate source for NLI efficiency
        src_truncated = src[:max_source_chars]

        sents = split_sentences(gen)
        if not sents:
            continue

        pairs = [(src_truncated, s) for s in sents]
        scores = nli.predict(pairs, apply_softmax=True)

        n_entailed = 0
        has_contradiction = False
        for s_scores in scores:
            n_total_sentences += 1
            ent_score = s_scores[ent_idx]
            con_score = s_scores[con_idx]
            if ent_score >= 0.5:
                n_entailed += 1
            if con_score >= contradiction_threshold:
                n_contradicted_sentences += 1
                has_contradiction = True

        if has_contradiction:
            n_summaries_with_contradiction += 1

        faith = n_entailed / len(sents)
        per_summary_faithfulness.append(faith)

    return {
        "mean_faithfulness": float(np.mean(per_summary_faithfulness)) if per_summary_faithfulness else 0.0,
        "median_faithfulness": float(np.median(per_summary_faithfulness)) if per_summary_faithfulness else 0.0,
        "hallucination_rate": n_summaries_with_contradiction / max(n_summaries, 1),
        "contradiction_sentence_rate": n_contradicted_sentences / max(n_total_sentences, 1),
        "n_summaries": n_summaries,
        "n_total_sentences": n_total_sentences,
    }


# =========================================================
# Metric 4 — Entity Coverage (spaCy NER)
# =========================================================
def compute_entity_coverage(generated, sources, model_name="en_core_web_trf"):
    """
    Named entity preservation: how many entities from source appear in summary.
    Tracks PERSON, ORG, GPE, LAW, DATE separately.
    """
    print("Computing entity coverage (spaCy NER)...")
    try:
        import spacy
    except ImportError:
        print("  spacy not installed")
        return {"error": "spacy not installed"}

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Fall back to smaller model
        try:
            print(f"  {model_name} not found, falling back to en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {"error": f"no spacy model available"}

    legal_labels = {"PERSON", "ORG", "GPE", "LAW", "DATE"}

    coverages = {label: [] for label in legal_labels}
    overall = []

    for tid in generated:
        gen = generated[tid]
        src = sources.get(tid, "")
        if not gen or not src:
            continue

        # Truncate source for NER efficiency
        src_doc = nlp(src[:5000])
        gen_doc = nlp(gen)

        src_ents = {(e.text.lower(), e.label_) for e in src_doc.ents
                    if e.label_ in legal_labels}
        gen_ents = {(e.text.lower(), e.label_) for e in gen_doc.ents
                    if e.label_ in legal_labels}

        if not src_ents:
            continue

        # Per-label coverage
        for label in legal_labels:
            src_l = {t for t, l in src_ents if l == label}
            gen_l = {t for t, l in gen_ents if l == label}
            if src_l:
                cov = len(src_l & gen_l) / len(src_l)
                coverages[label].append(cov)

        overall_cov = len(src_ents & gen_ents) / len(src_ents)
        overall.append(overall_cov)

    result = {
        "overall_entity_coverage": float(np.mean(overall)) if overall else 0.0,
        "n_evaluated": len(overall),
    }
    for label in legal_labels:
        result[f"{label.lower()}_coverage"] = (
            float(np.mean(coverages[label])) if coverages[label] else 0.0
        )
    return result


# =========================================================
# Metric 5 — Legal Outcome Preservation (rule-based)
# =========================================================
PLAINTIFF_FAVORABLE = [
    "granted", "affirmed", "awarded", "judgment for plaintiff",
    "judgment for the plaintiff", "reversed in favor", "in favor of plaintiff",
    "in favor of the plaintiff", "court found for", "ruled for plaintiff",
    "settled in favor", "prevailed",
]

DEFENDANT_FAVORABLE = [
    "denied", "dismissed", "judgment for defendant",
    "judgment for the defendant", "affirmed dismissal", "in favor of defendant",
    "in favor of the defendant", "court found for the defendant",
    "ruled for defendant", "no liability",
]

MIXED_OUTCOME = [
    "partially granted", "partially denied", "remanded",
    "affirmed in part", "reversed in part", "vacated in part",
    "settled", "consent decree",
]


def detect_outcome(text):
    """Return 'plaintiff', 'defendant', 'mixed', or 'unknown'."""
    text_lower = text.lower()

    # Mixed first (more specific patterns)
    for kw in MIXED_OUTCOME:
        if kw in text_lower:
            return "mixed"
    for kw in PLAINTIFF_FAVORABLE:
        if kw in text_lower:
            return "plaintiff"
    for kw in DEFENDANT_FAVORABLE:
        if kw in text_lower:
            return "defendant"
    return "unknown"


def compute_outcome_preservation(generated, sources):
    """Compare detected outcome in source (last 20%) vs generated summary."""
    print("Computing outcome preservation...")
    n_total = 0
    n_match = 0
    n_inverted = 0
    n_known = 0

    for tid in generated:
        gen = generated[tid]
        src = sources.get(tid, "")
        if not gen or not src:
            continue
        n_total += 1

        # Extract last 20% of source for outcome detection
        last_part = src[int(len(src) * 0.8):]
        src_outcome = detect_outcome(last_part)
        gen_outcome = detect_outcome(gen)

        if src_outcome == "unknown" or gen_outcome == "unknown":
            continue

        n_known += 1

        if src_outcome == gen_outcome:
            n_match += 1
        elif (src_outcome == "plaintiff" and gen_outcome == "defendant") or \
             (src_outcome == "defendant" and gen_outcome == "plaintiff"):
            n_inverted += 1

    return {
        "outcome_preservation_rate": n_match / max(n_known, 1),
        "outcome_inversion_rate": n_inverted / max(n_known, 1),
        "n_total": n_total,
        "n_with_known_outcome": n_known,
        "n_correct": n_match,
        "n_inverted": n_inverted,
    }


# =========================================================
# Metric 6 — Length Analysis
# =========================================================
def compute_length_analysis(generated, sources):
    """Word counts, sentence counts, compression ratios."""
    print("Computing length analysis...")
    word_counts = []
    sent_counts = []
    compression_ratios = []

    for tid in generated:
        gen = generated[tid]
        src = sources.get(tid, "")
        if not gen:
            continue
        gw = len(gen.split())
        gs = len(split_sentences(gen))
        word_counts.append(gw)
        sent_counts.append(gs)
        if src:
            sw = len(src.split())
            if sw > 0:
                compression_ratios.append(gw / sw)

    return {
        "mean_summary_words": float(np.mean(word_counts)) if word_counts else 0.0,
        "std_summary_words": float(np.std(word_counts)) if word_counts else 0.0,
        "median_summary_words": float(np.median(word_counts)) if word_counts else 0.0,
        "mean_summary_sentences": float(np.mean(sent_counts)) if sent_counts else 0.0,
        "mean_compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
        "min_summary_words": min(word_counts) if word_counts else 0,
        "max_summary_words": max(word_counts) if word_counts else 0,
        "n_evaluated": len(word_counts),
    }
