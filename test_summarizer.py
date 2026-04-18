"""
Quick test: load the trained model and summarize sample legal texts.
Works on CPU — no GPU needed.
"""

import sys
import torch
from tokenizers import Tokenizer

from model.transformer import build_model
from inference.hierarchical import HierarchicalSummarizer

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file("checkpoints/tokenizer.json")
print(f"  Vocab: {tokenizer.get_vocab_size()}")

device = "cpu"
print(f"\nBuilding model on {device}...")
model = build_model(
    device=device,
    vocab_size=32000,
    d_model=384,
    n_heads=6,
    n_encoder_layers=6,
    n_decoder_layers=4,
    d_ff=1536,
    max_seq_len=512,
    dropout=0.0,
    pad_idx=0,
)

print("Loading checkpoint...")
ckpt = torch.load("checkpoints/stage3_best.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  Loaded (val_loss: {ckpt.get('val_loss', '?'):.4f}, epoch: {ckpt.get('epoch', '?')})")

# Use greedy decoding on CPU for speed (beam search is very slow on CPU)
summarizer = HierarchicalSummarizer(
    model=model,
    tokenizer=tokenizer,
    device=device,
    beam_width=1,       # greedy — fast on CPU
    max_gen_len=150,
    min_gen_len=40,
    length_penalty=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)

# -------------------------------------------------------
# Test documents
# -------------------------------------------------------
tests = [
    {
        "name": "Contract Dispute",
        "text": """On January 15, 2023, plaintiff ABC Corporation filed a breach of contract
        lawsuit against XYZ Industries in the United States District Court for the Southern
        District of New York. The complaint alleges that XYZ Industries failed to deliver
        manufacturing equipment as specified in a purchase agreement dated March 1, 2022,
        valued at approximately $2.4 million. ABC Corporation claims that despite making
        full payment, XYZ Industries delivered defective equipment that did not meet the
        contractual specifications. The plaintiff seeks compensatory damages of $2.4 million,
        consequential damages for lost production revenue estimated at $800,000, and
        attorney's fees. XYZ Industries filed a motion to dismiss arguing that the equipment
        met industry standards and that ABC Corporation failed to provide adequate notice of
        the alleged defects within the contractually required 30-day inspection period. The
        court denied the motion to dismiss, finding that ABC Corporation had sufficiently
        pleaded its claims and that factual disputes regarding the inspection period remained
        for trial. Discovery is ongoing.""",
    },
    {
        "name": "Employment Discrimination",
        "text": """The Equal Employment Opportunity Commission brought this Title VII action
        against Riverside Healthcare System on behalf of Dr. Sarah Chen, alleging sex
        discrimination and retaliation. Dr. Chen, a cardiologist employed by Riverside from
        2015 to 2021, claimed she was paid significantly less than her male counterparts
        performing substantially similar work, was denied promotion to department chair
        despite superior qualifications, and was terminated after filing an internal
        complaint. The evidence showed that male cardiologists at Riverside earned between
        $420,000 and $480,000 annually, while Dr. Chen's salary was $340,000. Hospital
        records indicated Dr. Chen had higher patient satisfaction scores and more published
        research than the male candidate who was promoted. Riverside argued that compensation
        differences were based on years of experience and patient volume. The jury found in
        favor of Dr. Chen on both the discrimination and retaliation claims, awarding
        $580,000 in back pay, $200,000 in compensatory damages for emotional distress, and
        $1.2 million in punitive damages. The court also granted injunctive relief requiring
        Riverside to implement pay equity audits.""",
    },
    {
        "name": "Legislative Bill",
        "text": """This Act amends the Clean Water Act to establish new standards for
        industrial wastewater discharge into navigable waters of the United States. Section
        201 requires all industrial facilities discharging more than 10,000 gallons per day
        to install advanced filtration systems meeting EPA-specified standards within 36
        months of enactment. Section 202 establishes a grant program administered by the
        Environmental Protection Agency to assist small businesses with compliance costs,
        authorizing $500 million in appropriations for fiscal years 2024 through 2028.
        Section 203 increases civil penalties for violations from $25,000 to $75,000 per
        day per violation and introduces criminal penalties for knowing violations resulting
        in serious bodily injury. Section 204 requires the EPA Administrator to submit an
        annual report to Congress on compliance rates, enforcement actions, and water quality
        improvements in affected waterways. The Congressional Budget Office estimates the
        total cost at $3.2 billion over five years, offset by projected reductions in
        healthcare costs associated with waterborne illness.""",
    },
]

print("\n" + "=" * 70)
print("SUMMARIZATION TESTS")
print("=" * 70)

for i, test in enumerate(tests, 1):
    print(f"\n{'-' * 70}")
    print(f"Test {i}: {test['name']}")
    print(f"{'-' * 70}")
    print(f"\n[INPUT — {len(test['text'].split())} words]")
    print(test["text"].strip()[:500] + ("..." if len(test["text"]) > 500 else ""))

    print(f"\n[GENERATING SUMMARY...]")
    summary = summarizer.summarize(test["text"])

    print(f"\n[SUMMARY — {len(summary.split())} words]")
    print(summary)

print("\n" + "=" * 70)
print("TESTS COMPLETE")
print("=" * 70)
