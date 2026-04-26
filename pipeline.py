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
import re
import sys
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tokenizers import Tokenizer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sentence_transformers import SentenceTransformer

from baseline.preprocessing import (
    preprocess_document,
    load_document_store,
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
_DENSE_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
_RRF_K                = 60


# =========================================================
# Hybrid retriever — TF-IDF + dense (RRF fusion)
# =========================================================
class _HybridRetriever:
    def __init__(self, chunks: list[str], encoder: SentenceTransformer):
        self.chunks = chunks
        self.encoder = encoder

        self.tfidf = TfidfVectorizer(
            lowercase=True, stop_words=None,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.tmat = self.tfidf.fit_transform(chunks)

        self.dmat = encoder.encode(
            chunks, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def retrieve(self, question: str, k: int = 5):
        ts = cos_sim(self.tfidf.transform([question]), self.tmat)[0]
        qe = self.encoder.encode(
            question, convert_to_numpy=True, normalize_embeddings=True,
        )
        ds = self.dmat @ qe

        # Reciprocal Rank Fusion
        ra = np.argsort(np.argsort(-ts))
        rb = np.argsort(np.argsort(-ds))
        scores = 1.0 / (_RRF_K + ra) + 1.0 / (_RRF_K + rb)

        top = min(k, len(self.chunks))
        idx = np.argsort(scores)[-top:][::-1]
        return [(int(i), self.chunks[i], float(scores[i])) for i in idx]


# =========================================================
# Question expansion
# =========================================================
_EXPANSIONS = [
    (
        re.compile(r'\b(court|venue|jurisdict|disput|settl)\w*\b', re.IGNORECASE),
        ["What is the legal venue for disputes?",
         "Which court has jurisdiction?",
         "Where will disputes be resolved?"],
    ),
    (
        re.compile(
            r'(how long|how many years).*(confidential|secret|nda|agreement ends|expires)',
            re.IGNORECASE,
        ),
        ["How long do confidentiality obligations last after termination?",
         "What is the post-termination confidentiality period?",
         "For how many years does confidentiality apply after the NDA ends?"],
    ),
    (
        re.compile(
            r'\b(contact\s+person|authorized\s+contact|representative|point\s+of\s+contact)\b',
            re.IGNORECASE,
        ),
        ["What are the names of the authorized contact persons?",
         "Who are the named representatives in the agreement?",
         "What are the contact details listed in the agreement?"],
    ),
    (
        re.compile(r'\b(terminat|cancel|end|expir)\w*\b', re.IGNORECASE),
        ["What is the termination notice period?",
         "How can either party end the agreement?",
         "What are the conditions for termination?"],
    ),
    (
        re.compile(r'\b(payment|fee|invoice|compens|remunerat)\w*\b', re.IGNORECASE),
        ["When must invoices be paid?",
         "What are the payment terms?",
         "How much is the fee?"],
    ),
]


def _get_variants(question: str) -> list[str]:
    variants = [question]
    for pat, alts in _EXPANSIONS:
        if pat.search(question):
            for a in alts:
                if a.lower() != question.lower() and a not in variants:
                    variants.append(a)
            break
    return variants[:3]


def _make_qa_chunks(text: str, chunk_words: int = 150, overlap_words: int = 30) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


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

        # --- dense encoder for hybrid retrieval ---
        print(f"Loading dense encoder ({_DENSE_MODEL})...")
        self.encoder = SentenceTransformer(_DENSE_MODEL)
        print("Dense encoder loaded")

        # --- runtime state ---
        self.current_doc_id           = None
        self.current_metadata         = None
        self.current_chunks           = None      # section chunks (for summarizer)
        self.current_retriever        = None      # HybridRetriever on fine QA chunks
        self.conversation_history     = []
        self.current_summary          = None
        self.current_textrank_summary = None

        print("Pipeline ready\n")

    # --------------------------------------------------
    # DOCUMENT INGEST + SUMMARY
    # --------------------------------------------------
    def process_document(self, pdf_path: str) -> dict:
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

        self.current_doc_id   = doc_id
        self.current_metadata = metadata
        self.current_chunks   = chunks
        self.conversation_history = []

        # Build fine-grained QA chunks (150 words, 30-word overlap) + hybrid retriever
        print("Building hybrid retriever (TF-IDF + dense)...")
        qa_text = cleaned_text or " ".join(c.get("text", "") for c in chunks)
        qa_chunks = _make_qa_chunks(qa_text, chunk_words=150, overlap_words=30)
        self.current_retriever = _HybridRetriever(qa_chunks, self.encoder)
        print(f"Retriever ready — {len(qa_chunks)} fine chunks")

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
            textrank_summary = "TextRank summarization unavailable."
        self.current_textrank_summary = textrank_summary

        return {
            "doc_id":              doc_id,
            "doc_type":            metadata["doc_type"],
            "parties":             metadata["parties"],
            "dates":               metadata["dates"],
            "jurisdiction":        metadata["jurisdiction"],
            "total_chunks":        len(chunks),
            "summary":             self.current_summary,
            "textrank_summary":    self.current_textrank_summary,
            "word_count":          summary_result["word_count"],
            "textrank_word_count": len(textrank_summary.split()),
            "message":             f"Document processed. {len(chunks)} sections indexed.",
        }

    # --------------------------------------------------
    # QA
    # --------------------------------------------------
    def answer(self, question: str) -> dict:
        if self.current_doc_id is None:
            raise RuntimeError("No document loaded. Call process_document() first.")

        variants = _get_variants(question)

        best_result  = None
        best_score   = float("-inf")

        for variant in variants:
            retrieved = self.current_retriever.retrieve(variant, k=5)

            for chunk_idx, chunk_text, ret_score in retrieved:
                if not chunk_text.strip():
                    continue

                pred = predict_span(
                    model=self.qa_model,
                    tokenizer=self.qa_tokenizer,
                    question=variant,
                    context=chunk_text,
                    device=self.device,
                    no_answer_threshold=_HAS_ANSWER_THRESHOLD,
                )

                if not pred["answer"] or len(pred["answer"].strip()) < 2:
                    continue

                # Combined score: span score + retrieval score weighted by 0.5
                combined = pred["score"] + ret_score * 0.5

                if combined > best_score:
                    best_score = combined
                    best_result = {
                        "raw_span":        pred["answer"],
                        "has_answer_prob": pred["has_answer_prob"],
                        "score":           pred["score"],
                        "ret_score":       ret_score,
                        "chunk_text":      chunk_text,
                    }

        if best_result is None:
            return self._not_found_response(question)

        if best_result["has_answer_prob"] < _HAS_ANSWER_THRESHOLD:
            return self._not_found_response(question)

        prob = best_result["has_answer_prob"]
        if prob > 0.80:
            confidence = "High"
        elif prob > 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        response = {
            "found":          True,
            "plain_answer":   best_result["raw_span"],
            "raw_span":       best_result["raw_span"],
            "section":        None,
            "page_start":     None,
            "page_end":       None,
            "source_display": "Found in document",
            "confidence":     confidence,
        }

        self.conversation_history.append((question, best_result["raw_span"]))
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
        self.current_retriever        = None
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
            print(f"   Confidence: {r['confidence']}")
