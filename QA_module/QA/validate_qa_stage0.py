"""
QA Stage 0 validation — runs all 8 checks from the spec.

Note on special-token ids: the spec wrote [CLS]=5, [SEP]=6. The real
BPE tokenizer already has ids 5/6 occupied, so we append [CLS]/[SEP]
at 32000/32001 and read the true ids from qa_special_tokens.json.
Check #1 below is adapted accordingly — it asserts that the ids in the
tokenizer match the ids recorded in the metadata file.
"""

import os
import sys
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QA.qa_config import (
    QA_TOKENIZER_PATH, SUMMARIZER_CKPT_PATH, MAX_TOTAL_LEN,
    load_qa_special_tokens,
)
from QA.model.qa_model import build_qa_model
from QA.training.loss import compute_loss
from QA.training.span_select import joint_span_select


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = load_qa_special_tokens()
    print(f"Special token meta: {meta}")

    # -------------------------------------------------
    # 1. Tokenizer special tokens present + consistent
    # -------------------------------------------------
    section("Check 1: tokenizer special tokens")
    tok = Tokenizer.from_file(QA_TOKENIZER_PATH)
    cls_id_tok = tok.token_to_id("[CLS]")
    sep_id_tok = tok.token_to_id("[SEP]")
    assert cls_id_tok == meta["cls_id"], (cls_id_tok, meta["cls_id"])
    assert sep_id_tok == meta["sep_id"], (sep_id_tok, meta["sep_id"])
    assert tok.get_vocab_size() == meta["new_vocab"]
    print(f"  [CLS]={cls_id_tok}  [SEP]={sep_id_tok}  vocab={tok.get_vocab_size()}  PASS")

    # -------------------------------------------------
    # 2. Model loads encoder weights correctly
    # -------------------------------------------------
    section("Check 2: model + encoder load")
    ckpt_path = SUMMARIZER_CKPT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Summarizer checkpoint not found: {ckpt_path}")

    # Cross-check: build a fresh random model and compare a few tensors
    # against the model with checkpoint loaded. If load succeeded they MUST differ.
    fresh = build_qa_model(meta, load_ckpt=None, freeze=False)
    model = build_qa_model(meta, load_ckpt=ckpt_path, freeze=True).to(device)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Encoder params: {encoder_params:,}")
    assert 5_000_000 < encoder_params < 32_000_000, encoder_params

    def _neq(a, b):
        return (a.detach().cpu() - b.detach().cpu()).abs().sum().item() > 1e-6

    loaded_w = model.encoder.layers[0].self_attn.W_q.weight
    fresh_w  = fresh.encoder.layers[0].self_attn.W_q.weight
    assert _neq(loaded_w, fresh_w), "encoder.layers.0.self_attn.W_q.weight was NOT overwritten by load"

    loaded_emb = model.embedding.weight[:32000]
    fresh_emb  = fresh.embedding.weight[:32000]
    assert _neq(loaded_emb, fresh_emb), "embedding rows [:32000] were NOT copied from checkpoint"
    print("  Weights differ from fresh random init -> load succeeded  PASS")

    # -------------------------------------------------
    # 3. Forward pass shape check
    # -------------------------------------------------
    section("Check 3: forward pass shapes")
    B, L = 2, MAX_TOTAL_LEN
    input_ids = torch.randint(10, 32000, (B, L), device=device)
    # Stamp [CLS] at position 0 and segment split at position 61 (1 + 60)
    input_ids[:, 0] = meta["cls_id"]
    input_ids[:, 61] = meta["sep_id"]
    input_ids[:, -1] = meta["sep_id"]
    segment_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    segment_ids[:, 62:] = 1
    attn = torch.ones(B, L, dtype=torch.long, device=device)

    start_logits, end_logits, has_answer = model(input_ids, segment_ids, attn)
    assert start_logits.shape == (B, L), start_logits.shape
    assert end_logits.shape   == (B, L), end_logits.shape
    assert has_answer.shape   == (B,),   has_answer.shape
    print(f"  start={tuple(start_logits.shape)} end={tuple(end_logits.shape)} "
          f"has_answer={tuple(has_answer.shape)}  PASS")

    # -------------------------------------------------
    # 4. Masking — question positions must be -1e9
    # -------------------------------------------------
    section("Check 4: answer-position masking")
    # [CLS] (position 0) must remain a valid target — it's the SQuAD 2.0
    # convention for 'no answer'. Question positions (1..61) must be masked.
    assert start_logits[0, 0].item()   > -1e8, "[CLS] must NOT be masked (no-answer target)"
    assert start_logits[0, 30].item()  < -1e8, "question position should be masked"
    assert start_logits[0, 100].item() > -1e8, "context position should NOT be masked"
    print("  PASS")

    # -------------------------------------------------
    # 5. Joint span selection: end >= start, within max length
    # -------------------------------------------------
    section("Check 5: joint span selection")
    fails = 0
    for _ in range(100):
        fs = torch.randn(1, 200)
        fe = torch.randn(1, 200)
        s, e = joint_span_select(fs, fe, max_answer_len=50)
        s = s.item(); e = e.item()
        if not (e >= s and (e - s + 1) <= 50):
            fails += 1
    assert fails == 0, f"{fails}/100 invalid spans"
    print("  100/100 spans valid  PASS")

    # -------------------------------------------------
    # 6. Loss computation check
    # -------------------------------------------------
    section("Check 6: combined loss")
    start_pos = torch.tensor([70, 80], device=device)
    end_pos   = torch.tensor([75, 90], device=device)
    has_label = torch.tensor([1.0, 1.0], device=device)
    total, comps = compute_loss(
        start_logits, end_logits, has_answer,
        start_pos, end_pos, has_label,
    )
    print(f"  total={total.item():.4f}  comps={ {k: float(v) for k, v in comps.items()} }")
    assert not torch.isnan(total)
    assert total.item() > 0
    print("  PASS")

    # -------------------------------------------------
    # 7. Gradient flow check — every trainable param gets a grad
    # -------------------------------------------------
    section("Check 7: gradient flow")
    model.zero_grad()
    total.backward()
    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad.append(name)
    if no_grad:
        print("  Missing gradients:", no_grad[:5], "...")
    assert not no_grad, f"{len(no_grad)} params with no grad"
    # Encoder is frozen, so ensure its params are excluded from trainable list
    enc_trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert len(enc_trainable) == 0, f"encoder should be frozen, got {len(enc_trainable)} trainable"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Encoder frozen, trainable params (embeddings + heads): {trainable:,}")
    print("  PASS")

    # -------------------------------------------------
    # 8. Data format contract check (synthetic example)
    # -------------------------------------------------
    section("Check 8: data format contract")
    from QA.data.qa_dataset import build_features, REQUIRED_RAW_FIELDS, REQUIRED_TOKENIZED_FIELDS

    raw = {
        "question":      "What are the termination conditions?",
        "context":       "Either party may terminate this agreement upon 30 days written notice to the other party.",
        "answer_text":   "upon 30 days written notice",
        "answer_start":  None,   # fill in below
        "is_answerable": True,
        "domain":        "legal",
    }
    raw["answer_start"] = raw["context"].find(raw["answer_text"])
    assert raw["answer_start"] >= 0

    feat = build_features(
        raw, tok,
        cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
    )
    assert feat is not None, "build_features returned None for a clean example"
    for f in REQUIRED_RAW_FIELDS + REQUIRED_TOKENIZED_FIELDS:
        assert f in feat, f"missing {f}"
    span_ids = feat["input_ids"][feat["answer_start_tok"]: feat["answer_end_tok"] + 1]
    decoded = tok.decode(span_ids)
    print(f"  answer span decodes to: {decoded!r}")
    assert raw["answer_text"].lower() in decoded.lower()
    print("  PASS")

    print("\n" + "=" * 50)
    print("QA Stage 0 COMPLETE — 8/8 checks passed")


if __name__ == "__main__":
    main()
