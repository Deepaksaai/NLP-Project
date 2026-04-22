"""
Replicates full_eval.py's predict_one step by step on the specific
examples that were reported as empty predictions. Finds where things
go wrong.
"""
import json
import torch
from QA_deberta_old.model import build_deberta_qa

device = torch.device("cuda")
model, tokenizer = build_deberta_qa(freeze_layers=0)
ckpt = torch.load("QA_deberta/checkpoints/deberta_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.to(device).eval()

with open("QA_deberta/data/val.json", encoding="utf-8") as f:
    val = json.load(f)

# Find examples where the model probably said empty but shouldn't have
# Pick 3 answerable "Effective Date" or "Governing Law" or "Expiration Date" questions
target_cats = ["Effective Date", "Expiration Date", "Governing Law", "Parties"]
examples = []
for ex in val:
    if not ex["has_answer"]:
        continue
    for cat in target_cats:
        if cat in ex["question"]:
            examples.append((cat, ex))
            break
    if len(examples) >= 5:
        break

max_len = 451
max_answer_len = 150

for cat, ex in examples:
    print(f"\n{'='*70}")
    print(f"Category: {cat}")
    print(f"Question: {ex['question'][:80]}")
    print(f"Gold: {ex['answer_text'][:100]}")

    enc = tokenizer(
        ex["question"], ex["context"],
        max_length=max_len,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    ).to(device)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets_cpu    = enc["offset_mapping"][0].cpu().tolist()

    sep_id = tokenizer.sep_token_id
    seps = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0].tolist()
    context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if len(seps) >= 2:
        context_mask[0, seps[0] + 1: seps[1]] = True

    with torch.no_grad():
        s_logits, e_logits, ha_logit = model(input_ids, attention_mask, context_mask)
    s_logits = s_logits[0]
    e_logits = e_logits[0]
    L = s_logits.size(0)

    # Paired argmax (same as full_eval)
    span_scores = s_logits.unsqueeze(1) + e_logits.unsqueeze(0)
    idx_t = torch.arange(L, device=device)
    valid = (idx_t.unsqueeze(0) >= idx_t.unsqueeze(1)) & (idx_t.unsqueeze(0) - idx_t.unsqueeze(1) < max_answer_len)
    span_scores = span_scores.masked_fill(~valid, float("-inf"))
    flat = span_scores.view(-1).argmax().item()
    best_s = flat // L
    best_e = flat % L
    best_score = span_scores.view(-1)[flat].item()

    print(f"\nPredicted span: start={best_s}  end={best_e}  score={best_score:.3f}")
    print(f"Position 0 score: start_logit={s_logits[0].item():.3f}  end_logit={e_logits[0].item():.3f}  pair={s_logits[0].item()+e_logits[0].item():.3f}")

    # What's at those positions?
    print(f"\nToken at start ({best_s}): {tokenizer.decode([input_ids[0, best_s].item()])!r}")
    print(f"Token at end ({best_e}): {tokenizer.decode([input_ids[0, best_e].item()])!r}")
    print(f"Offset at start: {offsets_cpu[best_s]}")
    print(f"Offset at end: {offsets_cpu[best_e]}")
    print(f"Is start in context_mask? {context_mask[0, best_s].item()}")
    print(f"Is end in context_mask? {context_mask[0, best_e].item()}")

    # Now replicate full_eval's decode logic
    if best_s == 0 and best_e == 0:
        pred_text = "(said no answer: both 0)"
    elif offsets_cpu and best_s < len(offsets_cpu) and offsets_cpu[best_s][1] > 0:
        char_s = offsets_cpu[best_s][0]
        char_e = offsets_cpu[best_e][1]
        pred_text = ex["context"][char_s:char_e].strip()
        print(f"\n>> Using char-offset path: [{char_s}:{char_e}]")
    else:
        pred_ids = input_ids[0, best_s: best_e + 1].cpu().tolist()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        print(f"\n>> Using fallback decode path (offsets[{best_s}]={offsets_cpu[best_s]})")

    print(f"FINAL PRED: {pred_text!r}")
