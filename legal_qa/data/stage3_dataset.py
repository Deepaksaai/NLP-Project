"""
Stage 3 — Legal QA Fine-Tuning Dataset.

Sources:
    • CUAD QA     — real contract clause extraction
    • LEDGAR      — provision-type classification converted to QA
    • COLIEE      — competition legal QA (loaded if available)

Special features:
    • "legal: " prefix prepended to every context before tokenisation
    • Cross-document negative examples (question from contract A paired
      with context from contract B)
    • 15 % QASPER anchor + 5 % SQuAD anchor per epoch

Usage:
    from legal_qa.data.stage3_dataset import LegalQADataset, collate_fn
    ds = LegalQADataset(split="train")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
"""

import logging
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from legal_qa.data.dataset_utils import (
    get_tokenizer,
    load_and_validate_squad,
    convert_to_features,
    verify_span,
    MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)

# Domain prefix prepended to every context in stage 3
LEGAL_PREFIX = "legal: "


# =========================================================================
# LEDGAR category → natural-language question mapping
# =========================================================================
LEDGAR_QUESTION_MAP: Dict[str, str] = {
    "Adjustments":
        "What adjustments can be made under this agreement?",
    "Agreements":
        "What are the key agreements between the parties?",
    "Amendments":
        "How can this agreement be amended?",
    "Anti-assignment":
        "Are there any restrictions on assigning this agreement?",
    "Anti-dilution":
        "What anti-dilution protections exist?",
    "Applicable Laws":
        "What laws apply to this agreement?",
    "Arbitration":
        "How are disputes resolved through arbitration?",
    "Base Coverage and Coverages":
        "What base coverage is provided under this agreement?",
    "Benefits":
        "What benefits does this agreement provide?",
    "Brokers":
        "What are the broker-related provisions?",
    "Cap On Liability":
        "What is the cap on liability under this agreement?",
    "Change Of Control":
        "What happens upon a change of control?",
    "Closing Conditions":
        "What conditions must be met before closing?",
    "Competent Courts":
        "Which courts have jurisdiction over this agreement?",
    "Compliance With Laws":
        "What compliance obligations exist under this agreement?",
    "Conditions":
        "What conditions apply to this agreement?",
    "Confidentiality":
        "What are the confidentiality obligations?",
    "Consent To Jurisdiction":
        "Is there a consent to jurisdiction clause?",
    "Consequences":
        "What are the consequences of breach or non-compliance?",
    "Costs":
        "How are costs allocated under this agreement?",
    "Counterparts":
        "Can this agreement be executed in counterparts?",
    "Covenant":
        "What covenants do the parties agree to?",
    "Damages":
        "What damages provisions are included?",
    "Definitions":
        "What are the key definitions in this agreement?",
    "Deliverables":
        "What deliverables are required under this agreement?",
    "Delivery":
        "What are the delivery terms?",
    "Dispute Resolution":
        "How are disputes resolved?",
    "Effectiveness":
        "When does this agreement become effective?",
    "Entire Agreement":
        "Does this agreement contain an entire agreement clause?",
    "Escrow":
        "What escrow arrangements are specified?",
    "Events of Default":
        "What events constitute a default?",
    "Exclusions":
        "What exclusions apply under this agreement?",
    "Execution":
        "How is this agreement executed?",
    "Expenses":
        "How are expenses handled?",
    "Fees":
        "What fees are specified in this agreement?",
    "Force Majeure":
        "What are the force majeure provisions?",
    "Further Assurances":
        "Is there a further assurances clause?",
    "General":
        "What are the general provisions of this agreement?",
    "Governing Law":
        "What is the governing law of this agreement?",
    "Guarantees":
        "What guarantees are provided?",
    "Headings":
        "What does the headings clause say?",
    "Indemnification":
        "What indemnification obligations exist?",
    "Independent Contractor":
        "Is there an independent contractor clause?",
    "Insurance":
        "What insurance requirements are specified?",
    "Intellectual Property":
        "What are the intellectual property provisions?",
    "Interest":
        "What interest terms apply?",
    "Interpretation":
        "How should this agreement be interpreted?",
    "Jury Trial Waiver":
        "Is there a jury trial waiver?",
    "Liability":
        "What liability provisions are included?",
    "Limitation Of Liability":
        "What are the limitations of liability?",
    "Liquidated Damages":
        "Are there liquidated damages provisions?",
    "Miscellaneous":
        "What miscellaneous provisions are included?",
    "No Waiver":
        "What does the no waiver clause say?",
    "Non-compete":
        "What are the non-competition restrictions?",
    "Non-disparagement":
        "Is there a non-disparagement clause?",
    "Non-solicitation":
        "What are the non-solicitation restrictions?",
    "Notice":
        "How should notices be given under this agreement?",
    "Notices":
        "What are the notice requirements?",
    "Obligations":
        "What obligations do the parties have?",
    "Ownership":
        "What are the ownership provisions?",
    "Payments":
        "What payment terms are specified?",
    "Penalties":
        "What penalties apply for breach?",
    "Permits":
        "What permits or approvals are required?",
    "Pricing":
        "What pricing terms are specified?",
    "Publicity":
        "What publicity restrictions exist?",
    "Receivables":
        "What receivables provisions are included?",
    "Remedies":
        "What remedies are available?",
    "Renewal":
        "What are the renewal terms?",
    "Representations":
        "What representations do the parties make?",
    "Representations & Warranties":
        "What representations and warranties are made?",
    "Resignation":
        "What are the resignation provisions?",
    "Returns":
        "What are the return terms?",
    "Scope Of Work":
        "What is the scope of work?",
    "Severability":
        "Is there a severability clause?",
    "Sharing":
        "What sharing obligations exist?",
    "Successors":
        "What are the successors and assigns provisions?",
    "Successors & Assigns":
        "What happens with successors and assigns?",
    "Survival":
        "Which provisions survive termination?",
    "Tax":
        "What are the tax obligations?",
    "Term":
        "What is the term of this agreement?",
    "Termination":
        "Under what conditions can this agreement be terminated?",
    "Termination For Cause":
        "When can a party terminate this agreement for cause?",
    "Third-Party Beneficiaries":
        "Are there any third-party beneficiary provisions?",
    "Time Of Essence":
        "Is time of the essence in this agreement?",
    "Title":
        "What title provisions are included?",
    "Transfers":
        "What transfer restrictions exist?",
    "Waiver":
        "What waiver provisions are included?",
    "Warranties":
        "What warranties are provided?",
    "Warranty":
        "What warranty is provided?",
}

# Fallback for any label not in the map
_DEFAULT_QUESTION = "What does this provision of the agreement say?"


# =========================================================================
# Internal loaders
# =========================================================================

# -------------------------------------------------------------------------
# CUAD QA
# -------------------------------------------------------------------------
def _load_cuad_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Load CUAD (Contract Understanding Atticus Dataset) in SQuAD format."""
    logger.info("Loading CUAD QA — %s …", split)
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split=split)
    except Exception:
        try:
            ds = load_dataset("cuad", split=split)
        except Exception as e:
            logger.warning("Could not load CUAD: %s — skipping", e)
            return []

    examples: List[dict] = []

    for row in ds:
        if max_examples and len(examples) >= max_examples:
            break

        question = row.get("question", "")
        context = row.get("context", "")
        answers = row.get("answers", {})

        if not question or not context:
            continue

        if answers and answers.get("text") and answers["text"][0]:
            ans_text = answers["text"][0]
            ans_start = answers["answer_start"][0]

            # Validate
            end = ans_start + len(ans_text)
            if 0 <= ans_start < len(context) and end <= len(context):
                extracted = context[ans_start:end]
                if extracted == ans_text:
                    examples.append({
                        "question": question,
                        "context": context,
                        "answer_text": ans_text,
                        "answer_start": ans_start,
                        "is_answerable": True,
                    })
                    continue

            # If validation failed, try to find the text
            pos = context.find(ans_text)
            if pos >= 0:
                examples.append({
                    "question": question,
                    "context": context,
                    "answer_text": ans_text,
                    "answer_start": pos,
                    "is_answerable": True,
                })
            else:
                examples.append({
                    "question": question,
                    "context": context,
                    "answer_text": "",
                    "answer_start": -1,
                    "is_answerable": False,
                })
        else:
            examples.append({
                "question": question,
                "context": context,
                "answer_text": "",
                "answer_start": -1,
                "is_answerable": False,
            })

    logger.info("CUAD %s: %d examples", split, len(examples))
    return examples


# -------------------------------------------------------------------------
# LEDGAR → QA conversion
# -------------------------------------------------------------------------
def _load_ledgar_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load LEDGAR from lex_glue and convert each provision to QA format
    using the LEDGAR_QUESTION_MAP.

    The provision text becomes both the context and the answer with
    answer_start = 0.
    """
    logger.info("Loading LEDGAR — %s …", split)
    try:
        ds = load_dataset("lex_glue", "ledgar", split=split)
    except Exception as e:
        logger.warning("Could not load LEDGAR: %s — skipping", e)
        return []

    # Get label names from dataset features
    try:
        label_names = ds.features["label"].names
    except Exception:
        label_names = None

    examples: List[dict] = []

    for row in ds:
        if max_examples and len(examples) >= max_examples:
            break

        text = row.get("text", "")
        label_id = row.get("label", -1)

        if not text:
            continue

        # Resolve label name
        if label_names and 0 <= label_id < len(label_names):
            label_name = label_names[label_id]
        else:
            label_name = str(label_id)

        question = LEDGAR_QUESTION_MAP.get(label_name, _DEFAULT_QUESTION)

        examples.append({
            "question": question,
            "context": text,
            "answer_text": text,
            "answer_start": 0,
            "is_answerable": True,
        })

    logger.info("LEDGAR %s: %d examples", split, len(examples))
    return examples


# -------------------------------------------------------------------------
# COLIEE (optional — only if available)
# -------------------------------------------------------------------------
def _load_coliee_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Attempt to load COLIEE legal QA.  This dataset may not be freely
    available on HuggingFace, so we degrade gracefully.
    """
    logger.info("Attempting to load COLIEE — %s …", split)
    try:
        ds = load_dataset("nguha/legalbench", split=split)
        examples: List[dict] = []

        for row in ds:
            if max_examples and len(examples) >= max_examples:
                break

            question = row.get("question", row.get("text", ""))
            context = row.get("context", row.get("passage", ""))
            answer = row.get("answer", "")

            if not question or not context:
                continue

            if answer:
                pos = context.find(answer)
                if pos >= 0:
                    examples.append({
                        "question": question,
                        "context": context,
                        "answer_text": answer,
                        "answer_start": pos,
                        "is_answerable": True,
                    })
                    continue

            examples.append({
                "question": question,
                "context": context,
                "answer_text": "",
                "answer_start": -1,
                "is_answerable": False,
            })

        logger.info("COLIEE/LegalBench %s: %d examples", split, len(examples))
        return examples

    except Exception as e:
        logger.warning("COLIEE/LegalBench not available: %s — skipping", e)
        return []


# =========================================================================
# Cross-document negative generation
# =========================================================================
def _generate_cross_doc_negatives(
    examples: List[dict],
    n_negatives: int = 0,
    seed: int = 42,
) -> List[dict]:
    """
    Generate cross-document negative examples by pairing a question
    from one contract with context from a different contract.

    These are realistic unanswerable examples because users often ask
    about clauses that do not exist in the document they uploaded.
    """
    if not examples or n_negatives <= 0:
        return []

    rng = random.Random(seed)
    negatives: List[dict] = []

    # Separate by (question, context) pairs
    contexts = [ex["context"] for ex in examples]

    for _ in range(n_negatives):
        q_idx = rng.randint(0, len(examples) - 1)
        c_idx = rng.randint(0, len(contexts) - 1)

        # Ensure question and context come from different examples
        attempts = 0
        while c_idx == q_idx and attempts < 10:
            c_idx = rng.randint(0, len(contexts) - 1)
            attempts += 1

        if c_idx == q_idx:
            continue

        negatives.append({
            "question": examples[q_idx]["question"],
            "context": contexts[c_idx],
            "answer_text": "",
            "answer_start": -1,
            "is_answerable": False,
        })

    logger.info("Generated %d cross-document negatives", len(negatives))
    return negatives


# =========================================================================
# Full Stage-3 Legal QA Dataset
# =========================================================================
class LegalQADataset(Dataset):
    """
    Stage-3 legal QA dataset with domain prefix, cross-document negatives,
    15 % QASPER anchor, and 5 % SQuAD anchor.
    """

    def __init__(
        self,
        split: str = "train",
        qasper_anchor_frac: float = 0.15,
        squad_anchor_frac: float = 0.05,
        cross_neg_ratio: float = 0.3,
        max_cuad: Optional[int] = None,
        max_ledgar: Optional[int] = None,
        max_coliee: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__()
        if tokenizer is None:
            tokenizer = get_tokenizer()

        # ---- Legal sources ----
        cuad_raw = _load_cuad_examples(split, max_examples=max_cuad)
        ledgar_raw = _load_ledgar_examples(split, max_examples=max_ledgar)
        coliee_raw = _load_coliee_examples(split, max_examples=max_coliee)

        legal_raw = cuad_raw + ledgar_raw + coliee_raw
        logger.info(
            "Stage 3 legal raw: %d (CUAD=%d, LEDGAR=%d, COLIEE=%d)",
            len(legal_raw), len(cuad_raw), len(ledgar_raw), len(coliee_raw),
        )

        # ---- Cross-document negatives ----
        n_cross = int(len(legal_raw) * cross_neg_ratio)
        cross_negs = _generate_cross_doc_negatives(legal_raw, n_negatives=n_cross)
        legal_raw.extend(cross_negs)

        # ---- Convert to features with "legal: " prefix ----
        self.features: List[dict] = []
        self.is_answerable_flags: List[bool] = []
        discarded = 0

        for ex in legal_raw:
            feats = convert_to_features(
                ex, tokenizer, domain_prefix=LEGAL_PREFIX,
            )
            if feats is None:
                discarded += 1
                continue
            self.features.append(feats)
            self.is_answerable_flags.append(ex["is_answerable"])

        logger.info(
            "Legal features: %d (%d discarded)", len(self.features), discarded
        )

        # ---- QASPER anchor (15 %) ----
        n_qasper = int(len(self.features) * qasper_anchor_frac
                       / (1 - qasper_anchor_frac - squad_anchor_frac))
        try:
            from legal_qa.data.stage2_dataset import _load_qasper_examples
            qasper_raw = _load_qasper_examples(split, max_examples=n_qasper)
            for ex in qasper_raw:
                feats = convert_to_features(ex, tokenizer)
                if feats is not None:
                    self.features.append(feats)
                    self.is_answerable_flags.append(ex["is_answerable"])
            logger.info("QASPER anchor: %d examples added", len(qasper_raw))
        except Exception as e:
            logger.warning("QASPER anchor failed: %s", e)

        # ---- SQuAD anchor (5 %) ----
        n_squad = int(len(self.features) * squad_anchor_frac
                      / (1 - squad_anchor_frac))
        squad_train, _ = load_and_validate_squad()
        n_squad = min(n_squad, len(squad_train))
        squad_sample = random.sample(squad_train, n_squad) if n_squad > 0 else []

        for ex in squad_sample:
            feats = convert_to_features(ex, tokenizer)
            if feats is not None:
                self.features.append(feats)
                self.is_answerable_flags.append(ex["is_answerable"])
        logger.info("SQuAD anchor: %d examples added", len(squad_sample))

        # ---- Shuffle ----
        combined = list(zip(self.features, self.is_answerable_flags))
        random.shuffle(combined)
        self.features = [c[0] for c in combined]
        self.is_answerable_flags = [c[1] for c in combined]

        n_ans = sum(self.is_answerable_flags)
        n_unans = len(self.is_answerable_flags) - n_ans
        logger.info(
            "LegalQADataset ready — %d features, ans=%d, unans=%d",
            len(self.features), n_ans, n_unans,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# =========================================================================
# Collation
# =========================================================================
def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Stack feature dicts into a batched tensor dict."""
    keys = ["input_ids", "attention_mask", "token_type_ids",
            "start_positions", "end_positions", "is_answerable"]

    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [item[k] for item in batch]
        if tensors[0].dim() == 0:
            out[k] = torch.stack(tensors)
        else:
            shape0 = tensors[0].shape
            for i, t in enumerate(tensors):
                if t.shape != shape0:
                    raise ValueError(
                        f"Shape mismatch for key '{k}': "
                        f"item 0 has {shape0}, item {i} has {t.shape}"
                    )
            out[k] = torch.stack(tensors)
    return out


# =========================================================================
# Main — smoke test
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tok = get_tokenizer()
    print(f"Tokenizer: {tok.__class__.__name__}, vocab size: {tok.vocab_size}\n")

    print("=" * 60)
    print("STAGE 3 — Legal QA Dataset  (100-example smoke test)")
    print("=" * 60)

    # Try CUAD first; fall back to LEDGAR
    cuad_raw = _load_cuad_examples("train", max_examples=60)
    ledgar_raw = _load_ledgar_examples("train", max_examples=40)
    test_raw = cuad_raw[:60] + ledgar_raw[:40]

    # Generate some cross-doc negatives
    cross_negs = _generate_cross_doc_negatives(test_raw, n_negatives=20)
    test_raw.extend(cross_negs)

    passed = 0
    failed = 0
    discarded = 0
    n_answerable = 0
    n_unanswerable = 0

    for ex in test_raw[:100]:
        feats = convert_to_features(ex, tok, domain_prefix=LEGAL_PREFIX)
        if feats is None:
            discarded += 1
            continue

        if ex["is_answerable"]:
            n_answerable += 1
        else:
            n_unanswerable += 1

        ok = verify_span(
            feats["input_ids"],
            feats["start_positions"].item(),
            feats["end_positions"].item(),
            ex["answer_text"],
            tok,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\nVerification summary (up to 100 examples):")
    print(f"  Passed:         {passed}/{total}")
    print(f"  Failed:         {failed}/{total}")
    print(f"  Discarded:      {discarded}")
    print(f"  Answerable:     {n_answerable}")
    print(f"  Unanswerable:   {n_unanswerable}")
    print(f"  Ratio ans/unans: {n_answerable}/{n_unanswerable}")
    print(f"\n  Domain prefix:  '{LEGAL_PREFIX}'")
