"""
Stage 0 Validation — runs all 5 checks to confirm the foundation is solid.

1. Parameter count in expected range (25M-35M)
2. Forward pass produces correct output shape
3. Tokenizer round-trip (encode → decode preserves text)
4. Causal mask correctness (can see self, cannot see future)
5. Gradient flows through every parameter
"""

import sys
import torch
from model.transformer import TransformerSummarizer, build_model


def check_1_param_count(model):
    """Parameter count should be ~30M (between 25M and 35M)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total:,}")
    assert 25_000_000 < total < 35_000_000, \
        f"Parameter count {total:,} outside expected range (25M-35M)"
    print("  PASS")


def check_2_forward_shape(model, device):
    """Forward pass should produce (batch, tgt_len, vocab_size) logits."""
    src = torch.randint(1, 32000, (2, 100)).to(device)
    tgt = torch.randint(1, 32000, (2, 50)).to(device)
    logits = model(src, tgt)
    print(f"  src:    {tuple(src.shape)}")
    print(f"  tgt:    {tuple(tgt.shape)}")
    print(f"  logits: {tuple(logits.shape)}")
    assert logits.shape == (2, 50, 32000), \
        f"Output shape {logits.shape} != expected (2, 50, 32000)"
    print("  PASS")


def check_3_tokenizer_roundtrip(tokenizer_path):
    """Encode → decode should approximately preserve text."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    texts = [
        "The plaintiff filed a motion for summary judgment.",
        "We propose a novel method for text summarization.",
        "The bill amends Section 42 of the United States Code.",
    ]
    for text in texts:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        # ByteLevel BPE preserves text exactly (case-sensitive)
        print(f"  Original : {text}")
        print(f"  Decoded  : {decoded}")
        # Check semantic equivalence (strip whitespace differences)
        assert text.replace(" ", "") == decoded.replace(" ", ""), \
            f"Round-trip mismatch:\n  '{text}'\n  '{decoded}'"
    print("  PASS")


def check_4_causal_mask(model):
    """Causal mask: position i can see 0..i, not i+1..n."""
    causal = model.make_causal_mask(10, device="cpu")
    print(f"  Causal mask shape: {tuple(causal.shape)}")

    # Can see itself
    assert causal[0, 0, 0, 0] == True, "Position 0 cannot see itself"
    assert causal[0, 0, 5, 5] == True, "Position 5 cannot see itself"

    # Can see past
    assert causal[0, 0, 5, 3] == True, "Position 5 cannot see position 3"

    # Cannot see future
    assert causal[0, 0, 0, 5] == False, "Position 0 can see position 5 (future!)"
    assert causal[0, 0, 3, 7] == False, "Position 3 can see position 7 (future!)"

    print("  PASS")


def check_5_gradient_flow(model, device):
    """Every parameter receives a gradient after backward pass."""
    model.train()
    src = torch.randint(1, 32000, (2, 30)).to(device)
    tgt = torch.randint(1, 32000, (2, 15)).to(device)

    logits = model(src, tgt)
    loss = logits.mean()
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad.append(name)

    if no_grad:
        print(f"  FAIL — no gradient for: {no_grad}")
        assert False, f"No gradient for {len(no_grad)} parameters"

    total = sum(1 for _ in model.parameters() if _.requires_grad)
    print(f"  All {total} parameters received gradients")
    print("  PASS")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer_path = "tokenizer/tokenizer.json"

    # Build model
    print("Building model...")
    model = build_model(
        device=device,
        vocab_size=32000,
        d_model=384,
        n_heads=6,
        n_encoder_layers=6,
        n_decoder_layers=4,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
        pad_idx=0,
    )

    # Run checks
    checks = [
        ("Check 1: Parameter count", lambda: check_1_param_count(model)),
        ("Check 2: Forward pass shape", lambda: check_2_forward_shape(model, device)),
        ("Check 3: Tokenizer round-trip", lambda: check_3_tokenizer_roundtrip(tokenizer_path)),
        ("Check 4: Causal mask correctness", lambda: check_4_causal_mask(model)),
        ("Check 5: Gradient flow", lambda: check_5_gradient_flow(model, device)),
    ]

    passed = 0
    failed = 0

    for name, fn in checks:
        print(f"\n{name}")
        print("-" * 40)
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL — {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/5 passed, {failed}/5 failed")

    if failed > 0:
        print("\nStage 0 NOT complete — fix failures before proceeding")
        sys.exit(1)
    else:
        print("\nStage 0 COMPLETE — ready for Stage 1 training")


if __name__ == "__main__":
    main()
