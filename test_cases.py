import sys
sys.path.append(".")

import torch
import json
from tokenizers import Tokenizer

from QA.model.qa_model import QAModel
from QA.inference.legal_pipeline import LegalQAPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TOKENIZER ----------------
tokenizer = Tokenizer.from_file("tokenizer/qa_tokenizer.json")

# ---------------- META ----------------
with open("tokenizer/qa_special_tokens.json") as f:
    qa_meta = json.load(f)

# ---------------- MODEL ----------------
model = QAModel(
    vocab_size=qa_meta["new_vocab"],
    cls_id=qa_meta["cls_id"],
    sep_id=qa_meta["sep_id"]
)

ckpt = torch.load("checkpoints/qa_stage3_best.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)

# ---------------- PIPELINE ----------------
pipeline = LegalQAPipeline(model, tokenizer, qa_meta, device)

# ---------------- TEST CASES ----------------
test_cases = [

    # ================= HARD LEGAL =================
    {
        "context": "The contract starts in 2020. The payment is 500 dollars. The agreement ends in 2025.",
        "question": "What is the payment amount?",
        "expected": "500"
    },
    {
        "context": "The license fee is 2000 rupees per month. A penalty of 500 rupees applies if late.",
        "question": "What is the license fee?",
        "expected": "2000"
    },
    {
        "context": "The agreement is valid for 3 years and may be extended by 1 additional year.",
        "question": "What is the duration of the agreement?",
        "expected": "3 years"
    },

    # ================= DISTRACTOR =================
    {
        "context": "The company was founded in 2000. The CEO earns 1 million dollars. The office is in Berlin.",
        "question": "Where is the office located?",
        "expected": "Berlin"
    },
    {
        "context": "The rent is 1000 USD per month. Electricity costs 200 USD. Water costs 100 USD.",
        "question": "What is the rent?",
        "expected": "1000"
    },

    # ================= PARAPHRASE =================
    {
        "context": "The contract shall terminate upon 30 days written notice.",
        "question": "After how many days can the contract be terminated?",
        "expected": "30"
    },
    {
        "context": "Water boils at 100 degrees Celsius.",
        "question": "What temperature is required for water to boil?",
        "expected": "100"
    },

    # ================= MULTIPLE ANSWERS =================
    {
        "context": "The agreement starts in 2020 and ends in 2025. The payment is 300 dollars.",
        "question": "When does the agreement end?",
        "expected": "2025"
    },

    # ================= TRICKY =================
    {
        "context": "John signed the contract on January 1, 2022. It becomes effective from February 1, 2022.",
        "question": "When was the contract signed?",
        "expected": "January 1, 2022"
    },

    # ================= LONG CONTEXT =================
    {
        "context": "This agreement outlines various responsibilities. The contract duration is set to 5 years. Payments must be made monthly. Termination requires notice.",
        "question": "What is the duration of the contract?",
        "expected": "5 years"
    },

    # ================= HARD NO ANSWER =================
    {
        "context": "The contract starts in 2020 and ends in 2025.",
        "question": "What is the payment amount?",
        "expected": "No Answer"
    },
    {
        "context": "The company operates in Europe and Asia.",
        "question": "What is the company's revenue?",
        "expected": "No Answer"
    },

    # ================= CONFUSING =================
    {
        "context": "The penalty is 200 dollars. The fee is 1000 dollars. The tax is 50 dollars.",
        "question": "What is the fee?",
        "expected": "1000"
    },
]

print("\n================ HARD TEST RESULTS ================\n")


# No-answer detection
def is_no_answer(text):
    text = text.lower()
    return (
        "not found" in text or
        "no answer" in text or
        "does not contain" in text or
        "not present" in text
    )


# ---------------- RUN TESTS ----------------
correct = 0

for i, case in enumerate(test_cases):

    result = pipeline.answer(
        document=case["context"],
        question=case["question"]
    )

    predicted = result.get("answer", "")

    print(f"Test Case {i+1}")
    print("-" * 40)
    print("Question :", case["question"])
    print("Expected :", case["expected"])
    print("Predicted:", predicted)

    if case["expected"].lower() == "no answer":
        if is_no_answer(predicted):
            print("Result   : PASS - PASS")
            correct += 1
        else:
            print("Result   : FAIL - FAIL")
    else:
        if case["expected"].lower() in predicted.lower():
            print("Result   : PASS - PASS")
            correct += 1
        else:
            print("Result   : FAIL - FAIL")

    print()


# ---------------- SUMMARY ----------------
total = len(test_cases)
accuracy = correct / total

print("============================================")
print(f"Total: {total}")
print(f"Passed: {correct}")
print(f"Accuracy: {accuracy:.2f}")
print("============================================")
