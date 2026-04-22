"""
Debug: inspect the actual logit values the model produces.
Reveals whether position 0 is attracting high scores unfairly.
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

# Look at 3 answerable examples
answerable = [ex for ex in val if ex["has_answer"]][:3]

for idx, ex in enumerate(answerable):
    print(f"\n{'='*70}")
    print(f"Example {idx+1}: {ex['question'][:80]}")
    print(f"Gold answer: {ex['answer_text'][:80]}")

    enc = tokenizer(ex["question"], ex["context"], max_length=451,
                    truncation="only_second", padding="max_length",
                    return_tensors="pt").to(device)
    sep_id = tokenizer.sep_token_id
    seps = (enc["input_ids"][0] == sep_id).nonzero(as_tuple=True)[0].tolist()
    ctx_mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
    if len(seps) >= 2:
        ctx_mask[0, seps[0] + 1: seps[1]] = True

    with torch.no_grad():
        s, e, h = model(enc["input_ids"], enc["attention_mask"], ctx_mask)
    s = s[0]
    e = e[0]

    # Top-5 start positions by logit
    top_s = torch.topk(s, 5)
    top_e = torch.topk(e, 5)
    print(f"\nTop-5 start logits:")
    for val_s, pos in zip(top_s.values.tolist(), top_s.indices.tolist()):
        tok = tokenizer.decode([enc["input_ids"][0, pos].item()])
        print(f"  pos {pos:3d}  logit={val_s:+.3f}  token={tok!r}")
    print(f"\nTop-5 end logits:")
    for val_e, pos in zip(top_e.values.tolist(), top_e.indices.tolist()):
        tok = tokenizer.decode([enc["input_ids"][0, pos].item()])
        print(f"  pos {pos:3d}  logit={val_e:+.3f}  token={tok!r}")

    print(f"\nPosition 0 (CLS) logits: start={s[0].item():+.3f}  end={e[0].item():+.3f}")
    print(f"has_ans_prob: {torch.sigmoid(h[0]).item():.3f}")

    # Independent argmax (what quick_eval does)
    indep_s = s.argmax().item()
    indep_e = e.argmax().item()
    print(f"\nIndependent argmax:  start={indep_s}  end={indep_e}")

    # Paired argmax (what full_eval does)
    L = s.size(0)
    span_scores = s.unsqueeze(1) + e.unsqueeze(0)
    idx_t = torch.arange(L, device=device)
    valid = (idx_t.unsqueeze(0) >= idx_t.unsqueeze(1)) & (idx_t.unsqueeze(0) - idx_t.unsqueeze(1) < 150)
    span_scores = span_scores.masked_fill(~valid, float("-inf"))
    flat = span_scores.view(-1).argmax().item()
    print(f"Paired argmax:       start={flat//L}  end={flat%L}  score={span_scores.view(-1)[flat].item():.3f}")
