"""
DeBERTa-v3 based QA Model — self-contained.

Architecture:
  - Backbone: deepset/deberta-v3-base-squad2 (DeBERTa-v3 pretrained on
    general English then fine-tuned on SQuAD 2.0). We load only the
    encoder; deepset's QA head is discarded.
  - SpanHead: from heads.py
  - HasAnswerHead: from heads.py
  - Loss: compute_loss() from loss.py

DeBERTa-v3 differences from BERT:
  - Uses SentencePiece tokenizer (requires sentencepiece + protobuf).
  - No token_type_ids — segment info is inferred from [SEP].
  - Context-only masking uses an explicit context_mask tensor built by
    the dataset during tokenization, not token_type_ids.

Requires: pip install transformers sentencepiece protobuf
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from transformers import AutoModel, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

from QA_deberta.heads import SpanHead, HasAnswerHead
from QA_deberta.loss import MAX_ANSWER_LEN

DEBERTA_MODEL_NAME = "deepset/deberta-v3-base-squad2"
DEBERTA_HIDDEN     = 768
DEBERTA_MAX_LEN    = 512

# Sliding window settings for long legal documents
WINDOW_SIZE   = 384
WINDOW_STRIDE = 128


class DebertaQAModel(nn.Module):
    """DeBERTa-v3 encoder + SpanHead (start) + SpanHead (end) + HasAnswerHead."""

    def __init__(
        self,
        model_name: str = DEBERTA_MODEL_NAME,
        freeze_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _HF_AVAILABLE:
            raise ImportError("pip install transformers sentencepiece protobuf")

        self.deberta = AutoModel.from_pretrained(model_name)
        self.deberta_dropout = nn.Dropout(dropout)

        self._freeze_layers(freeze_layers)

        self.start_head      = SpanHead(DEBERTA_HIDDEN)
        self.end_head        = SpanHead(DEBERTA_HIDDEN)
        self.has_answer_head = HasAnswerHead(DEBERTA_HIDDEN)

        for head in (self.start_head, self.end_head, self.has_answer_head):
            for p in head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _freeze_layers(self, n: int):
        """Freeze embedding + bottom n encoder layers."""
        for p in self.deberta.embeddings.parameters():
            p.requires_grad = False
        for layer in self.deberta.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_top_layers(self, keep_frozen: int = 6):
        """Unfreeze layers above `keep_frozen` for progressive fine-tuning."""
        for i, layer in enumerate(self.deberta.encoder.layer):
            if i >= keep_frozen:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        context_mask: torch.Tensor,     # (B, L) bool — True where context token
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # DeBERTa-v3 does NOT take token_type_ids
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        seq_out    = self.deberta_dropout(outputs.last_hidden_state)  # (B, L, 768)
        cls_hidden = seq_out[:, 0, :]                                  # (B, 768)

        start_logits = self.start_head(seq_out)           # (B, L)
        end_logits   = self.end_head(seq_out)             # (B, L)
        has_answer   = self.has_answer_head(cls_hidden)   # (B,)

        # Allow [CLS] at position 0 as the no-answer target
        mask = context_mask.clone()
        mask[:, 0] = True

        neg_inf = torch.full_like(start_logits, -1e9)
        start_logits = torch.where(mask, start_logits, neg_inf)
        end_logits   = torch.where(mask, end_logits,   neg_inf)

        return start_logits, end_logits, has_answer


def predict_span(
    model: DebertaQAModel,
    tokenizer,
    question: str,
    context: str,
    device: torch.device,
    max_answer_len: int = MAX_ANSWER_LEN,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    no_answer_threshold: float = 0.5,
) -> dict:
    """Sliding-window inference for long contexts."""
    model.eval()
    model.to(device)

    q_enc = tokenizer(question, add_special_tokens=False, max_length=64, truncation=True)
    q_ids = q_enc["input_ids"]

    ctx_enc    = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    ctx_ids    = ctx_enc["input_ids"]
    ctx_offset = ctx_enc["offset_mapping"]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    best = {
        "answer": "", "score": float("-inf"),
        "has_answer_prob": 0.0, "window_idx": -1, "start_char": 0,
    }

    window_idx = 0
    ctx_start  = 0

    while ctx_start < len(ctx_ids) or window_idx == 0:
        ctx_window = ctx_ids[ctx_start: ctx_start + window_size]

        # DeBERTa format: [CLS] question [SEP] context [SEP]
        input_ids      = [cls_id] + q_ids + [sep_id] + ctx_window + [sep_id]
        attention_mask = [1] * len(input_ids)

        # Build context mask: True where input_ids comes from the context window
        ctx_offset_in_seq = 1 + len(q_ids) + 1
        ctx_mask_list = [False] * len(input_ids)
        for i in range(ctx_offset_in_seq, ctx_offset_in_seq + len(ctx_window)):
            ctx_mask_list[i] = True

        ids_t   = torch.tensor([input_ids],      device=device)
        mask_t  = torch.tensor([attention_mask], device=device)
        ctx_t   = torch.tensor([ctx_mask_list],  device=device, dtype=torch.bool)

        with torch.no_grad():
            start_logits, end_logits, has_answer_logit = model(ids_t, mask_t, ctx_t)

        start_logits = start_logits[0]
        end_logits   = end_logits[0]
        has_prob     = torch.sigmoid(has_answer_logit[0]).item()

        best_score   = float("-inf")
        best_start_t = ctx_offset_in_seq
        best_end_t   = ctx_offset_in_seq

        for s in range(ctx_offset_in_seq, len(input_ids) - 1):
            for e in range(s, min(s + max_answer_len, len(input_ids) - 1)):
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score   = score
                    best_start_t = s
                    best_end_t   = e

        ctx_tok_start = best_start_t - ctx_offset_in_seq + ctx_start
        ctx_tok_end   = best_end_t   - ctx_offset_in_seq + ctx_start

        if best_score > best["score"]:
            if ctx_offset and ctx_tok_start < len(ctx_offset):
                char_start = ctx_offset[ctx_tok_start][0]
                char_end   = ctx_offset[min(ctx_tok_end, len(ctx_offset) - 1)][1]
                answer_text = context[char_start:char_end].strip()
                start_char  = char_start
            else:
                answer_ids  = ctx_ids[ctx_tok_start: ctx_tok_end + 1]
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                start_char  = 0

            best.update({
                "answer":          answer_text,
                "score":           best_score,
                "has_answer_prob": has_prob,
                "window_idx":      window_idx,
                "start_char":      start_char,
            })

        ctx_start  += stride
        window_idx += 1
        if ctx_start >= len(ctx_ids):
            break

    if best["has_answer_prob"] < no_answer_threshold:
        best["answer"] = ""

    return best


def build_deberta_qa(
    freeze_layers: int = 8,
    dropout: float = 0.1,
    model_name: str = DEBERTA_MODEL_NAME,
):
    """Factory. Returns (model, tokenizer)."""
    if not _HF_AVAILABLE:
        raise ImportError("pip install transformers sentencepiece protobuf")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = DebertaQAModel(
        model_name=model_name,
        freeze_layers=freeze_layers,
        dropout=dropout,
    )
    return model, tokenizer
