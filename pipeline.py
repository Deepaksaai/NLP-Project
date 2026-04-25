"""
Master pipeline — Legal Document Assistant.

Connects:
    baseline/preprocessing.py      (PDF extraction, cleaning, chunking,
                                    TF-IDF index, document store)
    baseline/inference/
        summarizer_inference.py    (hierarchical beam-search summarizer)
    QA_deberta/
        model.py                   (DeBERTa span-extraction QA)

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
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizers import Tokenizer
from transformers import AutoTokenizer

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
from QA_deberta.model import DebertaQAModel, predict_span, DEBERTA_MODEL_NAME
from text_rank_summarizer import extractive_summarize


# -------------------------------------------------------
# Paths
# -------------------------------------------------------
_SUMM_DIR       = os.path.join(_HERE, "CHECKPOINTS", "summarizer_checkpoints")
_SUMM_TOKENIZER = os.path.join(_SUMM_DIR, "tokenizer.json")
_SUMM_CKPT      = os.path.join(_SUMM_DIR, "stage3_best.pt")

_QA_CKPT        = os.path.join(_HERE, "QA_deberta", "deberta_best.pt")
_STORE_ROOT     = os.path.join(_HERE, "document_store")

_HAS_ANSWER_THRESHOLD = 0.5


# =========================================================
# PIPELINE
# =========================================================
class LegalDocumentPipeline:
    """
    End-to-end legal-document assistant.

    Loads the summarizer and QA models once at construction time, then
    handles any number of process_document() / answer() calls without
    reloading weights.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on: {self.device}")

        # --- summarizer tokenizer (custom BPE 32k vocab) ---
        self.summ_tokenizer = Tokenizer.from_file(_SUMM_TOKENIZER)
        print(f"Summarizer tokenizer loaded — vocab: {self.summ_tokenizer.get_vocab_size()}")

        # --- summarizer ---
        self.summarizer = load_summarizer(_SUMM_CKPT, self.device)
        print("Summarizer loaded")

        # --- QA model (DeBERTa) ---
        print(f"Loading QA model from {_QA_CKPT} ...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
        self.qa_model = DebertaQAModel(model_name=DEBERTA_MODEL_NAME)
        ckpt = torch.load(_QA_CKPT, map_location="cpu", weights_only=False)
        self.qa_model.load_state_dict(ckpt["model_state"])
        self.qa_model.to(self.device)
        self.qa_model.eval()
        print(f"QA model loaded (epoch {ckpt.get('epoch', '?')}, "
              f"val F1={ckpt.get('val_f1', 0):.4f})")

        # --- runtime state ---
        self.current_doc_id         = None
        self.current_metadata       = None
        self.current_chunks         = None
        self.current_tfidf_index    = None
        self.conversation_history   = []
        self.current_summary        = None
        self.current_textrank_summary = None

        print("Pipeline ready\n")

    # --------------------------------------------------
    # DOCUMENT INGEST + SUMMARY
    # --------------------------------------------------
    def process_document(self, pdf_path: str) -> dict:
        """
        Preprocess a PDF (with caching) and generate a summary.

        Returns a dict with doc_id, doc_type, parties, dates,
        jurisdiction, total_chunks, summary, word_count, message.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = generate_doc_id(pdf_path)
        store_path = os.path.join(_STORE_ROOT, doc_id)

        sentences = sections = cleaned_text = None

        if os.path.exists(store_path):
            print(f"Document {doc_id} already indexed — loading from cache")
            metadata, chunks, tfidf_index, sentences, sections, cleaned_text = \
                load_document_store(doc_id, store_root=_STORE_ROOT)
        else:
            (cleaned_text, sections, sentences, chunks,
             metadata, tfidf_index) = preprocess_document(
                pdf_path, store_root=_STORE_ROOT,
            )

        self.current_doc_id       = doc_id
        self.current_metadata     = metadata
        self.current_chunks       = chunks
        self.current_tfidf_index  = tfidf_index
        self.conversation_history = []

        # --- Transformer summarizer ---
        print("\nRunning transformer summarizer...")
        summary_result = run_summarization(
            chunks=self.current_chunks,
            model=self.summarizer,
            tokenizer=self.summ_tokenizer,
            device=self.device,
        )
        self.current_summary = summary_result["summary"]
        print(f"Transformer summary — {summary_result['word_count']} words "
              f"(path: {summary_result['path']})")

        # --- TextRank baseline summarizer ---
        print("Running TextRank summarizer...")
        textrank_summary = ""
        if sentences:
            try:
                textrank_summary, _ = extractive_summarize(
                    sentences,
                    sections=sections,
                    cleaned_text=cleaned_text,
                    num_sentences=8,
                    min_words=12,
                    max_words=80,
                    position_weight=0.05,
                    redundancy_threshold=0.35,
                )
                print(f"TextRank summary — {len(textrank_summary.split())} words")
            except Exception as e:
                print(f"TextRank failed: {e}")
                textrank_summary = "TextRank summarization unavailable."
        else:
            textrank_summary = "TextRank requires sentence data (re-process the document)."
        self.current_textrank_summary = textrank_summary

        return {
            "doc_id":            doc_id,
            "doc_type":          metadata["doc_type"],
            "parties":           metadata["parties"],
            "dates":             metadata["dates"],
            "jurisdiction":      metadata["jurisdiction"],
            "total_chunks":      len(chunks),
            "summary":           self.current_summary,
            "textrank_summary":  self.current_textrank_summary,
            "word_count":        summary_result["word_count"],
            "textrank_word_count": len(textrank_summary.split()),
            "message":           f"Document processed. {len(chunks)} sections indexed.",
        }

    # --------------------------------------------------
    # QA
    # --------------------------------------------------
    def answer(self, question: str) -> dict:
        """
        Answer a natural-language question about the currently loaded document.

        Returns a dict with found, plain_answer, raw_span, section,
        page_start, page_end, source_display, confidence.
        """
        if self.current_doc_id is None:
            raise RuntimeError("No document loaded. Call process_document() first.")

        # augment query with recent history
        if self.conversation_history:
            history_pairs = self.conversation_history[-2:]
            history_text = " ".join(f"Q: {q} A: {a}" for q, a in history_pairs)
            augmented_query = f"{history_text} Q: {question}"
        else:
            augmented_query = question

        # retrieve top-5 chunks
        top_chunks = retrieve_chunks(
            question=augmented_query,
            tfidf_index=self.current_tfidf_index,
            chunks=self.current_chunks,
            k=5,
        )

        # run span extraction on each chunk
        results = []
        for chunk in top_chunks:
            context = chunk.get("text", "")
            if not context.strip():
                continue
            pred = predict_span(
                model=self.qa_model,
                tokenizer=self.qa_tokenizer,
                question=question,
                context=context,
                device=self.device,
                no_answer_threshold=_HAS_ANSWER_THRESHOLD,
            )
            results.append({
                "chunk":           chunk,
                "raw_span":        pred["answer"],
                "score":           pred["score"],
                "has_answer_prob": pred["has_answer_prob"],
            })

        if not results:
            return self._not_found_response(question)

        winning = max(results, key=lambda r: r["score"])

        if winning["has_answer_prob"] < _HAS_ANSWER_THRESHOLD or not winning["raw_span"]:
            return self._not_found_response(question)

        prob = winning["has_answer_prob"]
        if prob > 0.80:
            confidence = "High"
        elif prob > 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        chunk = winning["chunk"]
        response = {
            "found":          True,
            "plain_answer":   winning["raw_span"],
            "raw_span":       winning["raw_span"],
            "section":        chunk.get("section"),
            "page_start":     chunk.get("page_start"),
            "page_end":       chunk.get("page_end"),
            "source_display": (
                f"Found in {chunk.get('section', 'document')}, "
                f"Pages {chunk.get('page_start', '?')}"
                f"-{chunk.get('page_end', '?')}"
            ),
            "confidence":     confidence,
        }

        self.conversation_history.append((question, winning["raw_span"]))
        self.conversation_history = self.conversation_history[-3:]

        return response

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset(self):
        """Clear per-document state; keep loaded models."""
        self.conversation_history     = []
        self.current_doc_id           = None
        self.current_metadata         = None
        self.current_chunks           = None
        self.current_tfidf_index      = None
        self.current_summary          = None
        self.current_textrank_summary = None
        print("Pipeline reset — ready for new document")

    # --------------------------------------------------
    # INTERNAL
    # --------------------------------------------------
    def _not_found_response(self, question: str) -> dict:
        msg = ("This information was not found in the document you "
               "provided. The document may not contain a clause "
               "addressing this question.")
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
    for q in [
        "What is the termination notice period?",
        "Who are the parties to this agreement?",
        "What is the governing law?",
    ]:
        r = pipe.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {r['plain_answer']}")
        if r["found"]:
            print(f"   Source: {r['source_display']}  |  Confidence: {r['confidence']}")
