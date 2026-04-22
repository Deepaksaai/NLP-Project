"""
Runs both retrieval systems (TF-IDF and Hybrid) on the 23 NDA test cases
and produces a side-by-side comparison table.

Usage (from NLP-Project/ root):
    python -m QA_deberta_old.compare_retrievers
"""

import sys
import os
import re
import torch
import numpy as np

# ─── The NDA document from test_cases.py ─────────────────────────────────────

DOCUMENT = """
NON-DISCLOSURE AGREEMENT (NDA)

This Nondisclosure Agreement or ("Agreement") has been entered into on the date of 
______________________________ and is by and between:

Party Disclosing Information: ______________________________ ("Disclosing Party").

Party Receiving Information: ______________________________ ("Receiving Party").

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

# ─── Test cases from test_cases.py ───────────────────────────────────────────

TEST_CASES = [
    ("What is the purpose of this agreement?",                    "unauthorized disclosure"),
    ("What type of relationship do the parties enter?",           "confidential relationship"),
    ("What does confidential information include?",               "commercial value"),
    ("What information is excluded if it becomes public?",        "publicly known"),
    ("What information is excluded if known before disclosure?",  "before disclosure"),
    ("Can approved disclosures be excluded?",                     "written approval"),
    ("Who benefits from confidentiality obligations?",            "Disclosing Party"),
    ("Who can access confidential information?",                  "employees"),
    ("What must those people sign?",                              "nondisclosure"),
    ("Can the receiving party use the information freely?",       "not"),
    ("When must materials be returned?",                          "immediately"),
    ("What materials must be returned?",                          "records"),
    ("When do obligations end?",                                  "trade secret"),
    ("Do obligations survive termination?",                       "continue"),
    ("Does this agreement create a partnership?",                 "no"),
    ("What happens if part is invalid?",                          "rest"),
    ("Can the agreement be modified?",                            "writing"),
    ("When is disclosure not punishable?",                        "government"),
    ("Can trade secrets be used in court?",                       "under seal"),
    ("Who is bound by the agreement?",                            "successors"),
    ("What must be done for oral confidential information?",      "written"),
    ("What happens if a right is not exercised?",                 "not a waiver"),
    ("What supersedes prior agreements?",                         "this agreement"),
]


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_words: int = 80, overlap_words: int = 20):
    """
    Smaller chunks for short document — each clause gets its own chunk.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


# ─── TF-IDF Retriever ─────────────────────────────────────────────────────────

class TFIDFRetriever:
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.chunks = chunks
        self.cosine_similarity = cosine_similarity
        self.vec = TfidfVectorizer(lowercase=True, stop_words=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(chunks)

    def retrieve(self, question, k=3):
        q_vec = self.vec.transform([question])
        scores = self.cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[-k:][::-1]
        return [(int(i), self.chunks[i], float(scores[i])) for i in top_idx]


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    RRF_K = 60

    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.cosine_similarity = cosine_similarity

        self.tfidf_vec = TfidfVectorizer(lowercase=True, stop_words=None,
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf_vec.fit_transform(chunks)

        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dense_matrix = self.encoder.encode(
            chunks, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def _rrf(self, a, b):
        ra = np.argsort(np.argsort(-a))
        rb = np.argsort(np.argsort(-b))
        return 1.0 / (self.RRF_K + ra) + 1.0 / (self.RRF_K + rb)

    def retrieve(self, question, k=3):
        tfidf_scores = self.cosine_similarity(
            self.tfidf_vec.transform([question]), self.tfidf_matrix
        )[0]
        q_emb = self.encoder.encode(
            question, convert_to_numpy=True, normalize_embeddings=True
        )
        dense_scores = self.dense_matrix @ q_emb
        final_scores = self._rrf(tfidf_scores, dense_scores)
        top_k = min(k, len(self.chunks))
        top_idx = np.argsort(final_scores)[-top_k:][::-1]
        return [(int(i), self.chunks[i], float(final_scores[i])) for i in top_idx]


# ─── Run QA with a retriever ──────────────────────────────────────────────────

def run_qa(model, tokenizer, question, retriever, device, predict_span_fn, top_k=5):
    top_chunks = retriever.retrieve(question, k=top_k)
    best_answer = ""
    best_score  = float("-inf")

    for _, chunk_text, retrieval_score in top_chunks:
        result = predict_span_fn(
            model=model, tokenizer=tokenizer,
            question=question, context=chunk_text, device=device,
        )
        if not result["answer"] or len(result["answer"].strip()) < 2:
            continue
        combined = result["score"] + retrieval_score * 0.5
        if combined > best_score:
            best_score  = combined
            best_answer = result["answer"]

    return best_answer


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load model
    print("Loading model...")
    from QA_deberta_old.model import build_deberta_qa, predict_span
    model, tokenizer = build_deberta_qa(freeze_layers=0)
    ckpt = torch.load(
        "QA_deberta_old/checkpoints/deberta_best.pt",
        map_location=device, weights_only=False
    )
    state = ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("Model loaded.")

    # Build chunks and retrievers
    chunks = chunk_text(DOCUMENT)
    print(f"Document split into {len(chunks)} chunks")

    print("Building TF-IDF retriever...")
    tfidf_ret = TFIDFRetriever(chunks)

    print("Building Hybrid retriever...")
    hybrid_ret = HybridRetriever(chunks)

    # Run tests
    tfidf_pass  = 0
    hybrid_pass = 0
    results     = []

    print("\nRunning 23 test cases on both systems...\n")

    for i, (question, expected) in enumerate(TEST_CASES, 1):
        tfidf_ans  = run_qa(model, tokenizer, question, tfidf_ret,  device, predict_span)
        hybrid_ans = run_qa(model, tokenizer, question, hybrid_ret, device, predict_span)

        tfidf_ok  = expected.lower() in tfidf_ans.lower()
        hybrid_ok = expected.lower() in hybrid_ans.lower()

        if tfidf_ok:  tfidf_pass  += 1
        if hybrid_ok: hybrid_pass += 1

        results.append((question, expected, tfidf_ans, tfidf_ok, hybrid_ans, hybrid_ok))

    # ─── Print results table ──────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON: TF-IDF vs Hybrid (TF-IDF + Dense)")
    print("=" * 100)
    print(f"{'#':<3} {'Question':<45} {'Expected':<22} {'TF-IDF':^6} {'Hybrid':^6}")
    print(f"{'-'*3} {'-'*45} {'-'*22} {'-'*6} {'-'*6}")

    for i, (q, exp, tans, tok, hans, hok) in enumerate(results, 1):
        q_short   = q[:44]
        exp_short = exp[:21]
        t_icon = "✅" if tok  else "❌"
        h_icon = "✅" if hok  else "❌"
        print(f"{i:<3} {q_short:<45} {exp_short:<22} {t_icon:^6} {h_icon:^6}")

    print("=" * 100)
    print(f"{'TOTAL PASSED':<50} {tfidf_pass}/23        {hybrid_pass}/23")
    print(f"{'ACCURACY':<50} {tfidf_pass/23:.1%}        {hybrid_pass/23:.1%}")
    print("=" * 100)

    # ─── Detailed diff: cases where they disagree ─────────────────────────────
    print("\n--- CASES WHERE SYSTEMS DISAGREE ---\n")
    diff_found = False
    for i, (q, exp, tans, tok, hans, hok) in enumerate(results, 1):
        if tok != hok:
            diff_found = True
            winner = "TF-IDF" if tok else "Hybrid"
            print(f"Q{i}: {q}")
            print(f"  Expected : {exp}")
            print(f"  TF-IDF   : {tans[:100]}  {'✅' if tok else '❌'}")
            print(f"  Hybrid   : {hans[:100]}  {'✅' if hok else '❌'}")
            print(f"  Winner   : {winner}")
            print()
    if not diff_found:
        print("Both systems agree on all test cases.")

    # ─── Cases where BOTH fail ───────────────────────────────────────────────
    print("\n--- CASES WHERE BOTH FAIL ---\n")
    both_fail = False
    for i, (q, exp, tans, tok, hans, hok) in enumerate(results, 1):
        if not tok and not hok:
            both_fail = True
            print(f"Q{i}: {q}")
            print(f"  Expected : {exp}")
            print(f"  TF-IDF   : {tans[:100]}")
            print(f"  Hybrid   : {hans[:100]}")
            print()
    if not both_fail:
        print("No cases where both systems fail.")


if __name__ == "__main__":
    main()
