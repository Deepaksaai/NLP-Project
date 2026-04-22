import torch
import json
from QA_deberta_old.model import build_deberta_qa

device = torch.device("cuda")
model, tokenizer = build_deberta_qa(freeze_layers=0)
ckpt = torch.load("QA_deberta/checkpoints/deberta_best.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

with open("QA_deberta/data/val.json", encoding="utf-8") as f:
    val = json.load(f)

# Look at 5 ANSWERABLE examples
answerable = [ex for ex in val if ex["has_answer"]][:5]
for ex in answerable:
    enc = tokenizer(ex["question"], ex["context"], max_length=451,
                    truncation="only_second", padding="max_length",
                    return_tensors="pt").to(device)
    sep_id = tokenizer.sep_token_id
    seps = (enc["input_ids"][0] == sep_id).nonzero(as_tuple=True)[0].tolist()
    ctx_mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
    if len(seps) >= 2:
        ctx_mask[0, seps[0]+1:seps[1]] = True
    with torch.no_grad():
        s, e, h = model(enc["input_ids"], enc["attention_mask"], ctx_mask)
    ps, pe = s.argmax().item(), e.argmax().item()
    pred = tokenizer.decode(enc["input_ids"][0, ps:pe+1], skip_special_tokens=True)
    print(f"Q: {ex['question'][:60]}")
    print(f"  GOLD: {ex['answer_text'][:80]}")
    print(f"  PRED: {pred[:80]}  (positions {ps}-{pe})")
    print()