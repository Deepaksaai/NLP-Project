"""
LegalQAModel — DeBERTa-v3-large encoder + span extraction heads.

Architecture:
    ┌─────────────────────────────────────────────┐
    │  DeBERTa-v3-large Encoder                   │
    │  (microsoft/deberta-v3-large, hidden=1024)  │
    └──────────────────┬──────────────────────────┘
                       │  last_hidden_state (B, L, 1024)
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
      ┌─────────┐ ┌─────────┐ ┌────────────┐
      │ Start   │ │  End    │ │ HasAnswer  │
      │ Head    │ │  Head   │ │ Head       │
      │ 1024→1  │ │ 1024→1  │ │ 1024→1     │
      └────┬────┘ └────┬────┘ └─────┬──────┘
           │           │            │
      start_logits end_logits  has_answer_logit
       (B, L)      (B, L)       (B,)

    • Logits in the question segment (token_type_ids==0) and padding
      (attention_mask==0) are masked to -10000.0 so the answer can only
      come from the context (segment 1).
    • Joint span selection at inference: enumerate valid (i,j) pairs in
      the context with a small length penalty.
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model

from legal_qa.model.heads import SpanHead, HasAnswerHead

# Hidden dimension of DeBERTa-v3-large
_HIDDEN_SIZE = 1024

# Masking value — large enough to dominate softmax / argmax
_MASK_VALUE = -10000.0


class LegalQAModel(nn.Module):
    """
    Span-extraction QA model built on DeBERTa-v3-large.

    Three task heads:
        start_head      — per-token start-logit
        end_head        — per-token end-logit
        has_answer_head — binary "does the context contain an answer?"
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        super().__init__()

        # ---- Encoder ----
        self.encoder = DebertaV2Model.from_pretrained(model_name, torch_dtype=torch.float32)

        # ---- Dropout applied to encoder output during training ----
        self.dropout = nn.Dropout(0.1)

        # ---- Task heads ----
        self.start_head = SpanHead(hidden_size=_HIDDEN_SIZE)
        self.end_head = SpanHead(hidden_size=_HIDDEN_SIZE)
        self.has_answer_head = HasAnswerHead(hidden_size=_HIDDEN_SIZE)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,    # (B, L)
        token_type_ids: torch.Tensor,    # (B, L)  — 0=question, 1=context
        training: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:      (B, L) token ids.
            attention_mask: (B, L) 1 for real tokens, 0 for padding.
            token_type_ids: (B, L) 0 for question / special, 1 for context.
            training:       If True, apply dropout to encoder output.

        Returns:
            dict with keys:
                start_logits:      (B, L) — masked so only context positions
                                   can be selected.
                end_logits:        (B, L) — same masking.
                has_answer_logit:  (B,)   — raw logit (sigmoid → probability).
        """
        # ---- Encoder ----
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = encoder_output.last_hidden_state  # (B, L, 1024)

        # ---- Dropout (training only) ----
        if training:
            hidden_states = self.dropout(hidden_states)

        # ---- Heads ----
        start_logits = self.start_head(hidden_states)      # (B, L)
        end_logits = self.end_head(hidden_states)          # (B, L)

        cls_hidden = hidden_states[:, 0, :]                # (B, 1024)
        has_answer_logit = self.has_answer_head(cls_hidden)  # (B,)

        # ---- Masking: only context positions can be the answer ----
        # Invalid positions: question segment OR padding
        invalid_mask = (token_type_ids == 0) | (attention_mask == 0)  # (B, L)
        # CLS (position 0) must stay unmasked — unanswerable examples target it
        invalid_mask[:, 0] = False

        start_logits = start_logits.masked_fill(invalid_mask, _MASK_VALUE)
        end_logits = end_logits.masked_fill(invalid_mask, _MASK_VALUE)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "has_answer_logit": has_answer_logit,
        }

    # -----------------------------------------------------------------
    # Joint span selection (inference)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def get_answer_span(
        self,
        start_logits: torch.Tensor,   # (L,) — single example
        end_logits: torch.Tensor,      # (L,)
        input_ids: torch.Tensor,       # (L,)
        token_type_ids: torch.Tensor,  # (L,)
        max_answer_length: int = 150,
    ) -> tuple:
        """
        Joint span selection at inference time.

        Enumerates all valid (i, j) pairs in the context segment where
        i ≤ j and j-i ≤ max_answer_length.  For each pair:
            score = start_logits[i] + end_logits[j] - 0.05 * (j - i)
        The small length penalty discourages the model from selecting
        extremely long spans.

        This is fundamentally better than independent argmax on start/end
        because it guarantees end ≥ start by construction.

        Args:
            start_logits:      (L,) start logits for one example.
            end_logits:        (L,) end logits for one example.
            input_ids:         (L,) token ids (needed for decoding later).
            token_type_ids:    (L,) segment ids.
            max_answer_length: Maximum number of tokens in the answer span.

        Returns:
            (best_start, best_end, best_score)
        """
        # Valid context positions (token_type_ids == 1)
        context_positions = (token_type_ids == 1).nonzero(as_tuple=False).squeeze(-1)

        if context_positions.numel() == 0:
            return 0, 0, float("-inf")

        ctx_start = context_positions[0].item()
        ctx_end = context_positions[-1].item()

        best_score = float("-inf")
        best_start = ctx_start
        best_end = ctx_start

        # Convert to Python for the loop (small enough — at most ~400 context tokens)
        s_logits = start_logits.cpu().float()
        e_logits = end_logits.cpu().float()

        for i in range(ctx_start, ctx_end + 1):
            if token_type_ids[i].item() != 1:
                continue
            s_i = s_logits[i].item()
            j_max = min(i + max_answer_length, ctx_end)
            for j in range(i, j_max + 1):
                if token_type_ids[j].item() != 1:
                    continue
                score = s_i + e_logits[j].item() - 0.05 * (j - i)
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j

        return best_start, best_end, best_score

    # -----------------------------------------------------------------
    # Alternative constructor: from pretrained encoder only
    # -----------------------------------------------------------------
    @classmethod
    def from_pretrained_encoder(
        cls,
        checkpoint_path: str,
        model_name: str = "microsoft/deberta-v3-large",
    ) -> "LegalQAModel":
        """
        Load only the encoder weights from a saved checkpoint and randomly
        initialise the three task heads.

        Use this when you have a BERT / DeBERTa checkpoint (e.g. further
        pre-trained on legal corpora) but want fresh QA heads.

        Args:
            checkpoint_path: Path to a .pt / .bin checkpoint containing
                             encoder weights (with or without head weights).
            model_name:      HuggingFace model name to use for the encoder
                             config / tokenizer compatibility.

        Returns:
            LegalQAModel with encoder weights loaded and heads freshly
            initialised.
        """
        # Build model with random heads
        model = cls(model_name=model_name)

        # Load checkpoint
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Only load encoder keys; ignore head keys
        encoder_state = {
            k: v for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

        if encoder_state:
            missing, unexpected = model.load_state_dict(
                encoder_state, strict=False,
            )
            print(f"from_pretrained_encoder: loaded {len(encoder_state)} "
                  f"encoder keys, {len(missing)} missing, "
                  f"{len(unexpected)} unexpected")
        else:
            print("Warning: no encoder.* keys found in checkpoint — "
                  "loading full state dict with strict=False")
            model.load_state_dict(state_dict, strict=False)

        # Re-initialise heads to ensure they are fresh
        for head in (model.start_head, model.end_head, model.has_answer_head):
            nn.init.xavier_uniform_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

        return model


# =====================================================================
# Validation
# =====================================================================
def validate_model():
    """
    Create a random batch and run a full forward + backward pass,
    verifying shapes, masking, and gradient flow.
    """
    import sys

    print("=" * 60)
    print("LegalQAModel — architecture validation")
    print("=" * 60)

    device = "cpu"  # validation is always on CPU
    batch_size = 2
    seq_len = 512

    # ---- Build model ----
    print("Loading DeBERTa-v3-large encoder (this may take a moment) …")
    model = LegalQAModel()
    model.to(device)
    model.train()

    # ---- Synthetic inputs ----
    # Question occupies tokens 0-63, context occupies 64-400, rest is padding
    input_ids = torch.randint(1, 128000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    token_type_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    # Question region: positions 0..63 → segment 0
    token_type_ids[:, :64] = 0

    # Padding region: positions 400..511
    attention_mask[:, 400:] = 0
    input_ids[:, 400:] = 0  # PAD tokens

    # ---- Forward ----
    out = model(input_ids, attention_mask, token_type_ids, training=True)

    start_logits = out["start_logits"]
    end_logits = out["end_logits"]
    has_answer_logit = out["has_answer_logit"]

    # ---- Shape checks ----
    assert start_logits.shape == (batch_size, seq_len), (
        f"start_logits shape: expected {(batch_size, seq_len)}, "
        f"got {start_logits.shape}"
    )
    assert end_logits.shape == (batch_size, seq_len), (
        f"end_logits shape: expected {(batch_size, seq_len)}, "
        f"got {end_logits.shape}"
    )
    assert has_answer_logit.shape == (batch_size,), (
        f"has_answer_logit shape: expected {(batch_size,)}, "
        f"got {has_answer_logit.shape}"
    )
    print(f"  ✓ start_logits.shape  = {start_logits.shape}")
    print(f"  ✓ end_logits.shape    = {end_logits.shape}")
    print(f"  ✓ has_answer_logit.shape = {has_answer_logit.shape}")

    # ---- Masking checks ----
    # Question region (token_type_ids == 0) should be masked to -10000
    question_mask = token_type_ids == 0
    padding_mask = attention_mask == 0
    invalid_mask = question_mask | padding_mask

    masked_start = start_logits[invalid_mask]
    masked_end = end_logits[invalid_mask]

    assert torch.all(masked_start == _MASK_VALUE), (
        "start_logits: some invalid positions are NOT masked to -10000.0"
    )
    assert torch.all(masked_end == _MASK_VALUE), (
        "end_logits: some invalid positions are NOT masked to -10000.0"
    )
    print("  ✓ All question-segment logits are -10000.0 (masking works)")
    print("  ✓ All padding-region logits are -10000.0 (masking works)")

    # Context positions should NOT be masked
    context_mask = (token_type_ids == 1) & (attention_mask == 1)
    assert torch.any(start_logits[context_mask] != _MASK_VALUE), (
        "All context logits are masked — something is wrong"
    )
    print("  ✓ Context-segment logits are NOT masked (real values)")

    # ---- Gradient flow ----
    # Dummy loss — sum of all outputs
    loss = start_logits.sum() + end_logits.sum() + has_answer_logit.sum()
    loss.backward()

    all_have_grad = True
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            all_have_grad = False
            no_grad_params.append(name)

    if all_have_grad:
        print("  ✓ Gradients flow to ALL parameters")
    else:
        print(f"  ✗ {len(no_grad_params)} parameters have no gradient:")
        for p in no_grad_params[:10]:
            print(f"      - {p}")

    # ---- Joint span selection test ----
    model.eval()
    with torch.no_grad():
        out_eval = model(input_ids, attention_mask, token_type_ids, training=False)

    s, e, score = model.get_answer_span(
        out_eval["start_logits"][0],
        out_eval["end_logits"][0],
        input_ids[0],
        token_type_ids[0],
        max_answer_length=150,
    )
    assert 64 <= s <= e, (
        f"get_answer_span returned start={s} in non-context region"
    )
    assert e < 400, (
        f"get_answer_span returned end={e} in padding region"
    )
    print(f"  ✓ get_answer_span: start={s}, end={e}, score={score:.4f}")

    # ---- Param count ----
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    print("\n" + "=" * 60)
    print("Model validation passed ✓")
    print("=" * 60)


# =====================================================================
# CLI entry point
# =====================================================================
if __name__ == "__main__":
    validate_model()
