"""
QAModel = summarizer encoder (reused) + segment embedding (new)
                              + 3 QA heads (new, random init).

Design choices:
  - No external pretrained weights. The encoder comes from the
    summarizer checkpoint (stage3_best.pt).
  - Token embedding is resized from 32000 -> 32002 to hold [CLS]/[SEP];
    the first 32000 rows are copied from the summarizer, the last 2
    are randomly initialized.
  - Positional encoding is sinusoidal and reused unchanged; max_len
    is raised to MAX_TOTAL_LEN since summaries were shorter.
  - All encoder layers are frozen by default. A gradual unfreezing
    schedule will be applied later during QA fine-tuning.
"""

import math
import torch
import torch.nn as nn

from model.encoder import Encoder
from model.positional import PositionalEncoding

from QA.qa_config import (
    ORIG_VOCAB_SIZE, D_MODEL, N_HEADS, N_ENC_LAYERS, D_FF, DROPOUT,
    MAX_TOTAL_LEN, PAD_ID,
)
from QA.model.segment_embedding import SegmentEmbedding
from QA.model.heads import SpanHead, HasAnswerHead


class QAModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        cls_id: int,
        sep_id: int,
        pad_id: int = PAD_ID,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_ENC_LAYERS,
        d_ff: int = D_FF,
        max_seq_len: int = MAX_TOTAL_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.d_model = d_model

        # --- Embedding stack ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.segment_embedding = SegmentEmbedding(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # --- Reused summarizer encoder (same class, random init for now) ---
        self.encoder = Encoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # --- QA heads ---
        self.start_head = SpanHead(d_model)
        self.end_head   = SpanHead(d_model)
        self.has_answer_head = HasAnswerHead(d_model)

        self._init_new_weights()

    # -------------------------------------------------------
    # Weight init — only fresh modules. Encoder/embedding will be
    # overwritten by load_encoder() from the summarizer checkpoint.
    # -------------------------------------------------------
    def _init_new_weights(self):
        for p in self.start_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.end_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.has_answer_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------------------------------------------------
    # Load encoder weights from the summarizer stage3 checkpoint.
    # -------------------------------------------------------
    def load_encoder(self, ckpt_path: str, freeze: bool = True) -> dict:
        """
        Copy encoder.* and embedding.* from the summarizer state dict
        into this model. Return a report dict describing what was copied.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ("model_state", "state_dict", "model"):
                if k in ckpt:
                    state = ckpt[k]
                    break
            else:
                state = ckpt
        else:
            state = ckpt

        report = {"encoder_keys": 0, "embedding_copied_rows": 0, "skipped": []}

        # --- Embedding: copy first ORIG_VOCAB_SIZE rows ---
        emb_key = "embedding.weight"
        if emb_key in state:
            src = state[emb_key]
            n_copy = min(src.size(0), self.embedding.weight.size(0))
            with torch.no_grad():
                self.embedding.weight[:n_copy].copy_(src[:n_copy])
            report["embedding_copied_rows"] = n_copy
        else:
            report["skipped"].append(emb_key)

        # --- Encoder layers + final norm ---
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        report["encoder_keys"] = len(encoder_state)
        report["encoder_missing"] = list(missing)
        report["encoder_unexpected"] = list(unexpected)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Embedding stays trainable so the 2 new rows ([CLS], [SEP])
            # can learn meaningful vectors from scratch.

        return report

    # -------------------------------------------------------
    # Mask helpers
    # -------------------------------------------------------
    def make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Padding mask for the encoder self-attention.
        Returns (batch, 1, 1, seq_len) bool — True for real tokens.
        """
        return (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_context_mask(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Boolean mask of valid answer positions.

        Includes:
          - every context-segment token that is not padding
          - position 0 ([CLS]), which is the SQuAD 2.0 convention for
            'no answer'. Unanswerable examples use start=end=0 as the
            gold target, so [CLS] must not be masked out — otherwise
            cross-entropy on a -1e9 target produces an astronomical
            loss and no useful gradient.
        """
        context = (segment_ids == 1) & (input_ids != self.pad_id)
        cls_pos = torch.zeros_like(context)
        cls_pos[:, 0] = True
        return context | cls_pos

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        segment_ids: torch.Tensor,      # (B, L)
        attention_mask: torch.Tensor = None,  # unused placeholder — we rebuild from pad
    ):
        # Step 1: embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = x + self.segment_embedding(segment_ids)
        x = self.embed_dropout(x)

        # Step 2: encoder
        src_mask = self.make_attention_mask(input_ids)
        encoder_output = self.encoder(x, src_mask)   # (B, L, d)

        # Step 3: [CLS] is always position 0
        cls_hidden = encoder_output[:, 0, :]         # (B, d)

        # Step 4: logits
        start_logits = self.start_head(encoder_output)   # (B, L)
        end_logits   = self.end_head(encoder_output)     # (B, L)
        has_answer   = self.has_answer_head(cls_hidden)  # (B,)

        # Step 5: mask invalid answer positions to -1e9
        context_mask = self.make_context_mask(input_ids, segment_ids)  # (B, L) bool
        neg_inf = torch.full_like(start_logits, -1e9)
        start_logits = torch.where(context_mask, start_logits, neg_inf)
        end_logits   = torch.where(context_mask, end_logits,   neg_inf)

        return start_logits, end_logits, has_answer


def build_qa_model(tokenizer_meta: dict, load_ckpt: str = None, freeze: bool = True) -> QAModel:
    """
    Factory. tokenizer_meta must contain cls_id, sep_id, pad_id, new_vocab.
    """
    model = QAModel(
        vocab_size=tokenizer_meta["new_vocab"],
        cls_id=tokenizer_meta["cls_id"],
        sep_id=tokenizer_meta["sep_id"],
        pad_id=tokenizer_meta["pad_id"],
    )
    if load_ckpt is not None:
        report = model.load_encoder(load_ckpt, freeze=freeze)
        print(f"Encoder load report: {report}")
    return model
