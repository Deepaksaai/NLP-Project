"""
QA module configuration — single source of truth for paths, shapes,
and hyperparameters shared across model, data, training, validation.
"""

import os
import json

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Tokenizer: original BPE from the summarizer pipeline.
# The summarizer saved it into checkpoints_backup/ during training.
SUMMARIZER_TOKENIZER_PATH = os.path.join(_ROOT, "checkpoints", "tokenizer.json")

# QA-updated tokenizer (with [CLS]/[SEP] added) — written by add_qa_tokens.py
QA_TOKENIZER_PATH = os.path.join(_ROOT, "tokenizer", "qa_tokenizer.json")
QA_SPECIAL_TOKENS_JSON = os.path.join(_ROOT, "tokenizer", "qa_special_tokens.json")

# Summarizer checkpoint to load encoder weights from.
SUMMARIZER_CKPT_PATH = os.path.join(_ROOT, "checkpoints", "stage3_best.pt")

# Preprocessed data roots
QA_DATA_ROOT = os.path.join(_ROOT, "data", "qa")

# -------------------------------------------------------
# Encoder architecture — must match summarizer exactly
# -------------------------------------------------------
ORIG_VOCAB_SIZE = 32000
D_MODEL         = 384
N_HEADS         = 6
N_ENC_LAYERS    = 6
D_FF            = 1536
DROPOUT         = 0.1

# -------------------------------------------------------
# Sequence shape
# -------------------------------------------------------
MAX_QUESTION_LEN = 60
MAX_CONTEXT_LEN  = 400
# [CLS] q [SEP] ctx [SEP]  →  1 + 60 + 1 + 400 + 1 = 463, rounded to 462 per spec
MAX_TOTAL_LEN    = 462

# -------------------------------------------------------
# QA heads + loss
# -------------------------------------------------------
MAX_ANSWER_LEN          = 150
HAS_ANSWER_LOSS_WEIGHT  = 1.5
PAD_ID                  = 0

# -------------------------------------------------------
# Stage-3 legal overrides
# -------------------------------------------------------
LEGAL_MAX_ANSWER_LEN        = 200
LEGAL_LENGTH_PENALTY        = 0.05
LEGAL_TOKEN                 = "<legal>"
# The <legal> token was reserved at id 4 by the summarizer tokenizer
LEGAL_ID                    = 4


def load_qa_special_tokens():
    """Return dict with runtime ids for [CLS], [SEP], plus pad id."""
    if not os.path.exists(QA_SPECIAL_TOKENS_JSON):
        raise FileNotFoundError(
            f"{QA_SPECIAL_TOKENS_JSON} not found — run QA/add_qa_tokens.py first"
        )
    with open(QA_SPECIAL_TOKENS_JSON) as f:
        return json.load(f)
