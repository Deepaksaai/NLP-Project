"""
Complete inference pipeline for Legal Document QA.
"""

import logging
from typing import List, Dict, Any

import torch
from legal_qa.inference.retriever import LegalRetriever
from legal_qa.inference.predictor import QAPredictor
from legal_qa.inference.generator import PlainEnglishGenerator

logger = logging.getLogger(__name__)

class LegalQAPipeline:
    """
    End-to-End Legal QA Inference Pipeline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initializes the entire QA stack.
        """
        self.device = device
        
        logger.info("Initializing Predictor...")
        self.predictor = QAPredictor(checkpoint_path=checkpoint_path, device=self.device)
        
        logger.info("Initializing Generator...")
        # If device string is cuda, try to use device 0
        gen_device_id = 0 if "cuda" in self.device and torch.cuda.is_available() else -1
        self.generator = PlainEnglishGenerator(device_id=gen_device_id)
        
        # Pipeline State
        self.retriever = None
        self.conversation_history = []
        
        logger.info("Pipeline ready.")

    def load_document(self, chunks: List[Dict[str, Any]]):
        """
        Takes preprocessed document chunks and initializes the retriever.
        """
        logger.info(f"Loading document with {len(chunks)} chunks into pipeline.")
        self.retriever = LegalRetriever(chunks=chunks)
        self.reset()
        logger.info("Document loaded and retriever configured.")

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Executes the exact end-to-end flow:
         1. Retrieval with Query Expansion & History Augmentation
         2. BERT-based Span Extraction (Top K Chunks evaluated)
         3. BART-based Plain English Generation & Faithfulness Check
        """
        if self.retriever is None:
            raise ValueError("Document not loaded. Call load_document() first.")
            
        logger.info(f"\n--- Processing QA Trace ---\nInput Question: {question}")
        
        # 1. Retrieve
        top_chunks = self.retriever.retrieve(
            question=question,
            conversation_history=self.conversation_history,
            k=5
        )
        
        for i, c in enumerate(top_chunks):
            logger.debug(f"Rank {i+1} Chunk (Score {c['retrieval_score']:.4f}): {c.get('body', c.get('text'))[:100]}...")
            
        # 2. Extract
        result = self.predictor.predict(question, top_chunks)
        
        if not result["found"]:
            logger.info("Answer not found in document.")
            msg = "I could not find an answer to your question in the provided document."
            # We still add to conversation history
            self.conversation_history.append((question, msg))
            return {
                "found": False,
                "plain_answer": msg,
                "raw_span": "",
                "section": None,
                "page_start": None,
                "page_end": None,
                "has_answer_prob": result["has_answer_prob"],
                "confidence": "Low",
                "source_display": "Not found in document."
            }
            
        logger.info(f"Span Found! Prob: {result['has_answer_prob']:.4f} | Raw Span: {result['raw_span']}")
        
        # 3. Generate Plain English
        plain_english = self.generator.generate(question, result["raw_span"])
        
        # Format the final dictionary
        final_response = {
            "found": True,
            "plain_answer": plain_english,
            "raw_span": result["raw_span"],
            "section": result["section"],
            "page_start": result["page_start"],
            "page_end": result["page_end"],
            "has_answer_prob": result["has_answer_prob"],
            "confidence": result["confidence"],
            "source_display": result["source_display"]
        }
        
        logger.info(f"Generated Plain English: {plain_english}")
        
        # 4. History Maintenance
        self.conversation_history.append((question, plain_english))
        
        # Optional safeguard: limit history to last 5 turns to prevent massive prompt runaway
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
            
        return final_response

    def reset(self):
        """Clears conversational context."""
        self.conversation_history = []
        logger.info("Conversation history reset.")
