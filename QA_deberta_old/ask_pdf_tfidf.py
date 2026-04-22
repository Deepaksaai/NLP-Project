"""
Interactive QA on a legal PDF document.

Usage (from NLP-Project/ root):
    # Use the new LARGE model
    python -m QA_deberta.ask_pdf --pdf path/to/contract.pdf

    # Use the OLD base model
    python -m QA_deberta.ask_pdf --pdf path/to/contract.pdf --use_old

    # Ask a single question non-interactively
    python -m QA_deberta.ask_pdf --pdf contract.pdf --question "What is the governing law?"

Requires: pip install pypdf
"""

import os
import sys
import argparse
import re
import torch

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


# ─── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF and join pages. Basic cleanup of whitespace."""
    if not _PYPDF_AVAILABLE:
        raise ImportError("pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n\n".join(pages)

    # Basic cleanup: collapse excessive whitespace but preserve structure
    full_text = re.sub(r" +", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = full_text.strip()

    return full_text


# ─── TF-IDF chunk retriever (lightweight) ──────────────────────────────────────

def chunk_document(text: str, chunk_words: int = 400, overlap_words: int = 50):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def retrieve_top_chunks(question: str, chunks: list, k: int = 3):
    """Return the k chunks most relevant to the question using TF-IDF cosine."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # Fallback: just use first k chunks
        return [(i, c, 0.0) for i, c in enumerate(chunks[:k])]

    vec = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2),
                          sublinear_tf=True)
    doc_matrix = vec.fit_transform(chunks)
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, doc_matrix)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(int(i), chunks[i], float(sims[i])) for i in top_idx]


# ─── Run QA on a single question ───────────────────────────────────────────────

def answer_question(model, tokenizer, question: str, document_text: str,
                    device, predict_span_fn, top_k: int = 3) -> dict:
    """
    Retrieve top-k chunks, run QA on each, return the best answer.
    """
    chunks = chunk_document(document_text)
    top_chunks = retrieve_top_chunks(question, chunks, k=top_k)

    best_result = None
    best_combined_score = float("-inf")

    for chunk_idx, chunk_text, retrieval_score in top_chunks:
        result = predict_span_fn(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=chunk_text,
            device=device,
        )
        # Combined score: QA confidence × retrieval score
        combined = result["score"] * (0.5 + 0.5 * retrieval_score)

        if combined > best_combined_score and result["answer"]:
            best_combined_score = combined
            best_result = {
                **result,
                "chunk_idx":       chunk_idx,
                "retrieval_score": retrieval_score,
                "chunk_preview":   chunk_text[:200] + "...",
            }

    if best_result is None:
        return {
            "answer":          "No answer found in the document.",
            "has_answer_prob": 0.0,
            "score":           0.0,
            "chunk_idx":       -1,
            "retrieval_score": 0.0,
        }
    return best_result


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",      type=str, required=True, help="Path to PDF file")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question. If omitted, runs interactive mode.")
    parser.add_argument("--use_old",  action="store_true",
                        help="Use the old DeBERTa-base model from QA_deberta_old/")
    parser.add_argument("--top_k",    type=int, default=3,
                        help="Number of chunks to retrieve before QA.")
    args = parser.parse_args()

    # Pick which model to load
    if args.use_old:
        print("[ask_pdf] Using OLD model (DeBERTa-v3-base) from QA_deberta_old/")
        from QA_deberta_old.model import build_deberta_qa, predict_span
        default_ckpt = "QA_deberta_old/checkpoints/deberta_best.pt"
    else:
        print("[ask_pdf] Using NEW model (DeBERTa-v3-large) from QA_deberta/")
        from QA_deberta.model import build_deberta_qa, predict_span
        default_ckpt = "QA_deberta/checkpoints/deberta_large_best.pt"

    if not os.path.exists(default_ckpt):
        print(f"[ask_pdf] ERROR: checkpoint not found at {default_ckpt}")
        sys.exit(1)

    if not os.path.exists(args.pdf):
        print(f"[ask_pdf] ERROR: PDF not found at {args.pdf}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ask_pdf] device={device}")

    # Load model
    print(f"[ask_pdf] Loading model + checkpoint ...")
    model, tokenizer = build_deberta_qa(freeze_layers=0)
    ckpt = torch.load(default_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Extract PDF text
    print(f"[ask_pdf] Extracting text from {args.pdf} ...")
    document_text = extract_pdf_text(args.pdf)
    words = len(document_text.split())
    print(f"[ask_pdf] Document has {words} words")

    # Single question mode
    if args.question:
        result = answer_question(
            model, tokenizer, args.question, document_text,
            device, predict_span, top_k=args.top_k,
        )
        print_answer(args.question, result)
        return

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Ask questions about the document. Type 'quit' or Ctrl+C to exit.\n")

    try:
        while True:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            result = answer_question(
                model, tokenizer, question, document_text,
                device, predict_span, top_k=args.top_k,
            )
            print_answer(question, result)
    except (KeyboardInterrupt, EOFError):
        print("\n[ask_pdf] Exiting.")


def print_answer(question: str, result: dict):
    print("\n" + "─" * 70)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"   (confidence: {result['has_answer_prob']:.2f}, "
          f"span score: {result.get('score', 0):.2f}, "
          f"retrieved chunk #{result.get('chunk_idx', -1)})")
    if result.get("chunk_preview"):
        print(f"\n   Source chunk preview:")
        print(f"   {result['chunk_preview']}")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    main()
