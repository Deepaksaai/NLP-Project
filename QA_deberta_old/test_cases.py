from QA_module.inference.pipeline import QAPipeline


def run_tests():

    # 🔥 FULL NDA DOCUMENT (from your file)
    document = """
NON-DISCLOSURE AGREEMENT (NDA)

This Nondisclosure Agreement or ("Agreement") has been entered into on the date of 
______________________________ and is by and between:

Party Disclosing Information: ______________________________ (“Disclosing Party”).

Party Receiving Information: ______________________________ (“Receiving Party”).

For the purpose of preventing the unauthorized disclosure of Confidential Information.

1. Definition of Confidential Information:
Confidential Information shall include all information or material that has or could have commercial value or other utility.

If written, it must be labeled "Confidential".
If oral, written confirmation must be provided.

2. Exclusions:
(a) publicly known information  
(b) information known before disclosure  
(c) information learned through legitimate means  
(d) information disclosed with prior written approval  

3. Obligations:
Receiving Party must keep information confidential for the benefit of Disclosing Party.

Access limited to employees, contractors, third parties.
Those persons must sign nondisclosure agreements.

Receiving Party cannot use, publish, copy, or disclose information without approval.

All materials must be returned immediately upon written request.

4. Time Period:
Obligations continue until:
- information is no longer a trade secret OR
- Disclosing Party releases in writing

5. Relationships:
No partnership or joint venture is created.

6. Severability:
If part is invalid, rest still applies.

7. Integration:
This agreement supersedes all prior agreements.
Can only be modified in writing signed by both parties.

8. Waiver:
Failure to exercise a right is not a waiver.

9. Notice of Immunity:
No liability if disclosure is made to government or attorney for reporting violations.
Trade secrets may be used in court if filed under seal.

This agreement binds representatives, assigns, and successors.
"""

    # 🔥 HARD QUESTIONS
    test_cases = [
        ("What is the purpose of this agreement?", "unauthorized disclosure"),
        ("What type of relationship do the parties enter?", "confidential relationship"),
        ("What does confidential information include?", "commercial value"),
        ("What information is excluded if it becomes public?", "publicly known"),
        ("What information is excluded if known before disclosure?", "before disclosure"),
        ("Can approved disclosures be excluded?", "written approval"),
        ("Who benefits from confidentiality obligations?", "Disclosing Party"),
        ("Who can access confidential information?", "employees"),
        ("What must those people sign?", "nondisclosure"),
        ("Can the receiving party use the information freely?", "not"),
        ("When must materials be returned?", "immediately"),
        ("What materials must be returned?", "records"),
        ("When do obligations end?", "trade secret"),
        ("Do obligations survive termination?", "continue"),
        ("Does this agreement create a partnership?", "no"),
        ("What happens if part is invalid?", "rest"),
        ("Can the agreement be modified?", "writing"),
        ("When is disclosure not punishable?", "government"),
        ("Can trade secrets be used in court?", "under seal"),
        ("Who is bound by the agreement?", "successors"),
        ("What must be done for oral confidential information?", "written"),
        ("What happens if a right is not exercised?", "not a waiver"),
        ("What supersedes prior agreements?", "this agreement"),
    ]

    print("\n================ NDA TEST RESULTS ================\n")

    pipeline = QAPipeline()

    passed = 0

    for i, (question, expected) in enumerate(test_cases, 1):

        result = pipeline.answer(document, question)
        prediction = result["answer"]

        is_correct = expected.lower() in prediction.lower()

        if is_correct:
            passed += 1

        print(f"Test Case {i}")
        print("----------------------------------------")
        print("Question :", question)
        print("Expected :", expected)
        print("Predicted:", prediction)
        print("Result   :", "✅ PASS" if is_correct else "❌ FAIL")
        print()

    total = len(test_cases)
    accuracy = passed / total

    print("============================================")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {accuracy:.2f}")
    print("============================================")


if __name__ == "__main__":
    run_tests()
