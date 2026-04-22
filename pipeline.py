"""
Master pipeline — Legal Document Assistant.

Connects:
    baseline/preprocessing.py      (PDF extraction, cleaning, chunking,
                                    TF-IDF index, document store)
    baseline/inference/
        summarizer_inference.py    (hierarchical beam-search summarizer)
    QA_module/QA/inference/
        qa_inference.py            (span-extraction QA on pre-retrieved
                                    chunks + plain-English rewriter)

Usage:
    from pipeline import LegalDocumentPipeline
    pipe = LegalDocumentPipeline()
    info = pipe.process_document("sample_contract.pdf")
    print(info["summary"])
    ans = pipe.answer("What is the termination notice period?")
    print(ans["plain_answer"])
"""

import os
import sys
import json
import torch

# -------------------------------------------------------
# Path setup — put the project root and QA_module on sys.path
# before any local imports so both package trees resolve.
# -------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_QA_ROOT = os.path.join(_HERE, "QA_module")

for _p in (_HERE, _QA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizers import Tokenizer

from baseline.preprocessing import (
    preprocess_document,
    load_document_store,
    retrieve_chunks,
    generate_doc_id,
)
from baseline.inference.summarizer_inference import (
    load_summarizer,
    run_summarization,
)
from QA.inference.qa_inference import (
    load_qa_model,
    load_generation_model,
    run_qa,
    run_generation,
)


# -------------------------------------------------------
# Fixed paths — all resolved against this file's location
# -------------------------------------------------------
_SUMM_TOKENIZER  = os.path.join(_HERE,    "checkpoints", "tokenizer.json")
_SUMM_CKPT       = os.path.join(_HERE,    "checkpoints", "stage3_best.pt")
_QA_TOKENIZER    = os.path.join(_QA_ROOT, "tokenizer",   "qa_tokenizer.json")
_QA_SPECIAL_TOK  = os.path.join(_QA_ROOT, "tokenizer",   "qa_special_tokens.json")
_QA_CKPT         = os.path.join(_QA_ROOT, "checkpoints", "qa_stage3_best.pt")
_STORE_ROOT      = os.path.join(_HERE,    "document_store")

# Threshold for "did we find an answer?"
_HAS_ANSWER_THRESHOLD = 0.5


# =========================================================
# PIPELINE
# =========================================================
class LegalDocumentPipeline:
    """
    End-to-end legal-document assistant.

    Loads the summarizer and QA models once at construction time, then
    answers any number of process_document() / answer() calls without
    reloading weights.
    """

    # -----------------------------------------------------
    # INIT — load models, tokenizers, and state
    # -----------------------------------------------------
    def __init__(self):
        # --- device ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on: {self.device}")

        # --- tokenizers ---
        # The summarizer's BPE (32 000 vocab) and the QA tokenizer
        # (32 002 vocab — adds [CLS] and [SEP]) share the first 32 000
        # rows, so either can tokenize legal text. We load both and hand
        # each model the tokenizer it was trained with.
        self.summ_tokenizer = Tokenizer.from_file(_SUMM_TOKENIZER)
        self.qa_tokenizer   = Tokenizer.from_file(_QA_TOKENIZER)

        with open(_QA_SPECIAL_TOK) as f:
            self.qa_meta = json.load(f)

        print(f"Tokenizers loaded — summ vocab: "
              f"{self.summ_tokenizer.get_vocab_size()}, "
              f"qa vocab: {self.qa_tokenizer.get_vocab_size()}")

        # --- summarizer ---
        self.summarizer = load_summarizer(_SUMM_CKPT, self.device)
        print("Summarizer loaded")

        # --- QA model ---
        self.qa_model = load_qa_model(_QA_CKPT, self.qa_meta, self.device)
        print("QA model loaded")

        # --- (optional) generation model — none trained, so None ---
        self.gen_model = load_generation_model()
        print("Generation rewriter ready (using summarizer as backbone)")

        # --- runtime state ---
        self.current_doc_id       = None
        self.current_metadata     = None
        self.current_chunks       = None
        self.current_tfidf_index  = None
        self.conversation_history = []
        self.current_summary      = None

        print("Pipeline ready\n")

    # -----------------------------------------------------
    # DOCUMENT INGEST + SUMMARY
    # -----------------------------------------------------
    def process_document(self, pdf_path: str) -> dict:
        """
        Preprocess a PDF (with caching) and generate a summary.

        Args:
            pdf_path: Path to a legal PDF.

        Returns:
            {
              "doc_id":        str,
              "doc_type":      str,
              "parties":       list[str],
              "dates":         list[str],
              "jurisdiction":  str,
              "total_chunks":  int,
              "summary":       str,
              "word_count":    int,
              "message":       str,
            }
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = generate_doc_id(pdf_path)
        store_path = os.path.join(_STORE_ROOT, doc_id)

        if os.path.exists(store_path):
            print(f"Document {doc_id} already indexed — loading from cache")
            metadata, chunks, tfidf_index = load_document_store(
                doc_id, store_root=_STORE_ROOT,
            )
        else:
            (_cleaned_text, _sections, _sentences, chunks,
             metadata, tfidf_index) = preprocess_document(
                pdf_path, store_root=_STORE_ROOT,
            )

        # Cache on instance
        self.current_doc_id       = doc_id
        self.current_metadata     = metadata
        self.current_chunks       = chunks
        self.current_tfidf_index  = tfidf_index
        self.conversation_history = []

        # Summarize
        print("\nSummarizing...")
        summary_result = run_summarization(
            chunks=self.current_chunks,
            model=self.summarizer,
            tokenizer=self.summ_tokenizer,
            device=self.device,
        )
        self.current_summary = summary_result["summary"]
        print(f"Summary generated — {summary_result['word_count']} words "
              f"(path: {summary_result['path']})")

        return {
            "doc_id":       doc_id,
            "doc_type":     metadata["doc_type"],
            "parties":      metadata["parties"],
            "dates":        metadata["dates"],
            "jurisdiction": metadata["jurisdiction"],
            "total_chunks": len(chunks),
            "summary":      self.current_summary,
            "word_count":   summary_result["word_count"],
            "message":      f"Document processed. "
                            f"{len(chunks)} sections indexed.",
        }

    # -----------------------------------------------------
    # QA
    # -----------------------------------------------------
    def answer(self, question: str) -> dict:
        """
        Answer a natural-language question about the currently loaded
        document.

        Args:
            question: User question.

        Returns:
            {
              "found":          bool,
              "plain_answer":   str,
              "raw_span":       str | None,
              "section":        str | None,
              "page_start":     int | None,
              "page_end":       int | None,
              "source_display": str | None,
              "confidence":     "High" | "Medium" | "Low" | None,
            }
        """
        if self.current_doc_id is None:
            raise RuntimeError(
                "No document loaded. Call process_document() first."
            )

        # --- build query augmented with recent chat history ---
        if self.conversation_history:
            history_pairs = self.conversation_history[-2:]
            history_text = " ".join(
                f"Q: {q} A: {a}" for q, a in history_pairs
            )
            augmented_query = f"{history_text} Q: {question}"
        else:
            augmented_query = question

        # --- retrieve top-5 chunks ---
        top_chunks = retrieve_chunks(
            question=augmented_query,
            tfidf_index=self.current_tfidf_index,
            chunks=self.current_chunks,
            k=5,
        )

        # --- run span extraction per chunk ---
        per_chunk = run_qa(
            question=question,
            top_chunks=top_chunks,
            model=self.qa_model,
            tokenizer=self.qa_tokenizer,
            device=self.device,
        )

        if not per_chunk:
            return self._not_found_response(question)

        # pick the best chunk by combined evidence
        winning = max(per_chunk, key=lambda r: r["combined_score"])

        # --- has-answer threshold ---
        if (winning["has_answer_prob"] < _HAS_ANSWER_THRESHOLD
                or not winning["raw_span"]):
            return self._not_found_response(question)

        # --- plain-English generation (summarizer as rewriter) ---
        plain = run_generation(
            question=question,
            raw_span=winning["raw_span"],
            model=self.summarizer,
            tokenizer=self.summ_tokenizer,
            device=self.device,
        )

        prob = winning["has_answer_prob"]
        if prob > 0.80:
            confidence = "High"
        elif prob > 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        chunk = winning["chunk"]
        response = {
            "found":         True,
            "plain_answer":  plain,
            "raw_span":      winning["raw_span"],
            "section":       chunk.get("section"),
            "page_start":    chunk.get("page_start"),
            "page_end":      chunk.get("page_end"),
            "source_display": (
                f"Found in {chunk.get('section', 'document')}, "
                f"Pages {chunk.get('page_start', '?')}"
                f"-{chunk.get('page_end', '?')}"
            ),
            "confidence":    confidence,
        }

        # keep last 3 turns
        self.conversation_history.append((question, plain))
        self.conversation_history = self.conversation_history[-3:]

        return response

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------
    def reset(self):
        """Clear per-document state; keep loaded models."""
        self.conversation_history = []
        self.current_doc_id       = None
        self.current_metadata     = None
        self.current_chunks       = None
        self.current_tfidf_index  = None
        self.current_summary      = None
        print("Pipeline reset — ready for new document")

    # -----------------------------------------------------
    # INTERNAL
    # -----------------------------------------------------
    def _not_found_response(self, question: str) -> dict:
        msg = ("This information was not found in the document you "
               "provided. The document may not contain a clause "
               "addressing this question.")
        # still log the turn so subsequent retrieval is aware
        self.conversation_history.append((question, msg))
        self.conversation_history = self.conversation_history[-3:]
        return {
            "found":          False,
            "plain_answer":   msg,
            "raw_span":       None,
            "section":        None,
            "page_start":     None,
            "page_end":       None,
            "source_display": None,
            "confidence":     None,
        }


# =========================================================
# Quick smoke test
# =========================================================
if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.pdf"
    pipe = LegalDocumentPipeline()
    info = pipe.process_document(pdf)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(info["summary"])

    print("\n" + "=" * 60)
    print("QA")
    print("=" * 60)
    questions = [
        "What is the termination notice period?",
        "Who are the parties to this agreement?",
        "What is the governing law?",
    ]
    for q in questions:
        r = pipe.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {r['plain_answer']}")
        if r["found"]:
            print(f"   Source: {r['source_display']}  | "
                  f"Confidence: {r['confidence']}")
