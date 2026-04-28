"""
Interactive QA on a legal PDF document.
Hybrid TF-IDF + dense retrieval (RRF) + targeted question expansion.
Pure span extraction — no generation layer.

Usage:
    python -m QA_deberta_old.ask_pdf --pdf path/to/contract.pdf --use_old
    python -m QA_deberta_old.ask_pdf --pdf contract.pdf --question "What is the governing law?" --use_old

Requires: pip install pypdf sentence-transformers scikit-learn
"""

import os, sys, argparse, re, torch
import numpy as np

try:
    from pypdf import PdfReader
    _PYPDF = True
except ImportError:
    _PYPDF = False

try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except ImportError:
    _ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    _SK = True
except ImportError:
    _SK = False


def extract_pdf_text(pdf_path):
    if not _PYPDF:
        raise ImportError("pip install pypdf")
    reader = PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(text, chunk_words=150, overlap_words=30):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


class HybridRetriever:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K = 60

    def __init__(self, chunks):
        self.chunks = chunks
        if _SK:
            self.tfidf = TfidfVectorizer(lowercase=True, stop_words=None,
                                          ngram_range=(1,2), sublinear_tf=True)
            self.tmat = self.tfidf.fit_transform(chunks)
            print(f"[retriever] TF-IDF built ({len(chunks)} chunks)")
        else:
            self.tfidf = None

        if _ST:
            print(f"[retriever] Loading dense encoder...")
            self.enc = SentenceTransformer(self.DENSE_MODEL)
            self.dmat = self.enc.encode(chunks, convert_to_numpy=True,
                                         normalize_embeddings=True,
                                         show_progress_bar=False)
            print(f"[retriever] Dense built")
        else:
            self.enc = None

        print(f"[retriever] Hybrid ready ({len(chunks)} chunks)")

    def _rrf(self, a, b):
        ra = np.argsort(np.argsort(-a))
        rb = np.argsort(np.argsort(-b))
        return 1.0/(self.RRF_K+ra) + 1.0/(self.RRF_K+rb)

    def retrieve(self, question, k=3):
        if self.tfidf and self.enc:
            ts = cos_sim(self.tfidf.transform([question]), self.tmat)[0]
            qe = self.enc.encode(question, convert_to_numpy=True, normalize_embeddings=True)
            ds = self.dmat @ qe
            scores = self._rrf(ts, ds)
        elif self.tfidf:
            scores = cos_sim(self.tfidf.transform([question]), self.tmat)[0]
        else:
            qe = self.enc.encode(question, convert_to_numpy=True, normalize_embeddings=True)
            scores = self.dmat @ qe
        top = min(k, len(self.chunks))
        idx = np.argsort(scores)[-top:][::-1]
        return [(int(i), self.chunks[i], float(scores[i])) for i in idx]


EXPANSIONS = [
    (
        re.compile(r'\b(court|venue|jurisdict|disput|settl)\w*\b', re.IGNORECASE),
        ["What is the legal venue for disputes?",
         "Which court has jurisdiction?",
         "Where will disputes be resolved?"]
    ),
    (
        re.compile(r'(how long|how many years).*(confidential|secret|nda|agreement ends|expires)',
                   re.IGNORECASE),
        ["How long do confidentiality obligations last after termination?",
         "What is the post-termination confidentiality period?",
         "For how many years does confidentiality apply after the NDA ends?"]
    ),
    (
        re.compile(r'\b(contact\s+person|authorized\s+contact|representative|point\s+of\s+contact)\b',
                   re.IGNORECASE),
        ["What are the names of the authorized contact persons?",
         "Who are the named representatives in Clause 10?",
         "What are the contact details listed in the agreement?"]
    ),
]


def get_variants(question):
    variants = [question]
    for pat, alts in EXPANSIONS:
        if pat.search(question):
            for a in alts:
                if a.lower() != question.lower() and a not in variants:
                    variants.append(a)
            break
    return variants[:3]


def answer_question(model, tokenizer, question, retriever, device, predict_span_fn, top_k=5):
    variants = get_variants(question)
    if len(variants) > 1:
        print(f"   [expanding — {len(variants)} variants]")

    best, best_score = None, float("-inf")

    for variant in variants:
        for chunk_idx, chunk_text, ret_score in retriever.retrieve(variant, k=top_k):
            res = predict_span_fn(model=model, tokenizer=tokenizer,
                                  question=variant, context=chunk_text, device=device)
            if not res["answer"] or len(res["answer"].strip()) < 2:
                continue
            score = res["score"] + ret_score * 0.5
            if score > best_score:
                best_score = score
                best = {**res, "chunk_idx": chunk_idx,
                        "retrieval_score": ret_score,
                        "matched_variant": variant,
                        "chunk_preview": chunk_text[:200] + "..."}

    if best is None:
        return {"answer": "No answer found in the document.",
                "has_answer_prob": 0.0, "score": 0.0,
                "chunk_idx": -1, "retrieval_score": 0.0,
                "matched_variant": question}
    return best


def print_answer(question, result):
    print("\n" + "─"*70)
    print(f"Q: {question}")
    if result.get("matched_variant") and result["matched_variant"] != question:
        print(f"   (answered via: \"{result['matched_variant']}\")")
    print(f"A: {result['answer']}")
    print(f"   (confidence: {result['has_answer_prob']:.2f}  "
          f"retrieval: {result.get('retrieval_score',0):.3f}  "
          f"chunk #{result.get('chunk_idx',-1)})")
    if result.get("chunk_preview"):
        print(f"\n   Source: {result['chunk_preview']}")
    print("─"*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",      type=str, required=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--use_old",  action="store_true")
    parser.add_argument("--top_k",    type=int, default=5)
    args = parser.parse_args()

    if args.use_old:
        print("[ask_pdf] Using OLD model (DeBERTa-v3-base) from QA_deberta_old/")
        from QA_deberta_old.model import build_deberta_qa, predict_span
        ckpt_path = "QA_deberta_old/checkpoints/deberta_best.pt"
    else:
        print("[ask_pdf] Using NEW model (DeBERTa-v3-large) from QA_deberta/")
        from QA_deberta.model import build_deberta_qa, predict_span
        ckpt_path = "QA_deberta/checkpoints/deberta_large_best.pt"

    if not os.path.exists(ckpt_path):
        print(f"[ask_pdf] ERROR: checkpoint not found at {ckpt_path}"); sys.exit(1)
    if not os.path.exists(args.pdf):
        print(f"[ask_pdf] ERROR: PDF not found at {args.pdf}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ask_pdf] device={device}")

    print("[ask_pdf] Loading QA model + checkpoint...")
    model, tokenizer = build_deberta_qa(freeze_layers=0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    print(f"[ask_pdf] Extracting text from {args.pdf}...")
    doc_text = extract_pdf_text(args.pdf)
    print(f"[ask_pdf] Document has {len(doc_text.split())} words")

    chunks = chunk_document(doc_text)
    print(f"[ask_pdf] Split into {len(chunks)} chunks")
    retriever = HybridRetriever(chunks)

    if args.question:
        result = answer_question(model, tokenizer, args.question, retriever,
                                  device, predict_span, top_k=args.top_k)
        print_answer(args.question, result)
        return

    print("\n" + "="*70)
    print("INTERACTIVE MODE  (hybrid retrieval + targeted question expansion)")
    print("="*70)
    print("Ask questions about the document. Type 'quit' to exit.\n")

    try:
        while True:
            q = input("Question: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            result = answer_question(model, tokenizer, q, retriever,
                                      device, predict_span, top_k=args.top_k)
            print_answer(q, result)
    except (KeyboardInterrupt, EOFError):
        print("\n[ask_pdf] Exiting.")


if __name__ == "__main__":
    main()
