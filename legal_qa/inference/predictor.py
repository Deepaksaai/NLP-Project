"""
QA Predictor for exact span extraction.
"""

import logging
import torch
import string

from legal_qa.model.qa_model import LegalQAModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class QAPredictor:
    """
    Handles inference span extraction logic.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Loads the DeBERTa-v3-large tokeniser and LegalQAModel from a checkpoint.
        """
        self.device = device
        
        logger.info(f"Loading DeBERTa Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        
        logger.info(f"Loading QA Model from {checkpoint_path} to {device}...")
        self.model = LegalQAModel(model_name="microsoft/deberta-v3-large")
        
        # Load the checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Predictor initialized.")

    def _fix_entity_boundaries(self, raw_span: str, full_context: str, extract_end_idx: int) -> str:
        """
        If the decoded span ends mid-word, extends it to the natural word boundary.
        """
        if not raw_span or extract_end_idx >= len(full_context):
            return raw_span
            
        next_char = full_context[extract_end_idx:extract_end_idx+1]
        # If the next character is part of a word (not space or punctuation), we likely cut it off
        if next_char and next_char not in string.whitespace and next_char not in string.punctuation:
            # Find the next word boundary
            boundary_offset = 0
            while (extract_end_idx + boundary_offset < len(full_context) and 
                   full_context[extract_end_idx + boundary_offset] not in string.whitespace and 
                   full_context[extract_end_idx + boundary_offset] not in string.punctuation):
                boundary_offset += 1
            
            extended_span = raw_span + full_context[extract_end_idx:extract_end_idx + boundary_offset]
            logger.debug(f"Span boundary fixed: '{raw_span}' -> '{extended_span}'")
            return extended_span
            
        return raw_span

    @torch.no_grad()
    def predict_single_chunk(self, question: str, chunk_text: str) -> dict:
        """
        Runs the QA model forward pass on a single question-chunk pair.
        """
        # Always prepend the legal string during Stage 3 / Inference
        domain_context = "legal: " + chunk_text
        
        encoding = self.tokenizer(
            question,
            domain_context,
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        seq_ids = encoding.sequence_ids(batch_index=0)
        token_type_list = [0 if s is None or s == 0 else 1 for s in seq_ids]
        token_type_ids = torch.tensor([token_type_list], dtype=torch.long, device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=False
        )

        has_answer_prob = torch.sigmoid(outputs["has_answer_logit"]).item()

        start_idx, end_idx, best_span_score = self.model.get_answer_span(
            outputs["start_logits"][0],
            outputs["end_logits"][0],
            input_ids[0],
            token_type_ids[0],
            max_answer_length=150
        )

        raw_span = ""
        if start_idx > 0 and end_idx >= start_idx:
            span_ids = input_ids[0, start_idx:end_idx+1].tolist()
            raw_span = self.tokenizer.decode(span_ids, skip_special_tokens=True).strip()

        combined_score = has_answer_prob * best_span_score

        return {
            "raw_span": raw_span,
            "start_position": start_idx,
            "end_position": end_idx,
            "has_answer_prob": has_answer_prob,
            "best_span_score": best_span_score,
            "combined_score": combined_score,
            "input_ids_list": input_ids[0].tolist(),
            "domain_context": domain_context
        }

    def predict(self, question: str, retrieved_chunks: list) -> dict:
        """
        Evaluates extracted spans across all retrieved chunks and picks the winner.
        """
        best_result = None
        best_combined = float("-inf")
        best_chunk = None

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("body", chunk.get("text", ""))
            
            result = self.predict_single_chunk(question, chunk_text)
            
            if result["combined_score"] > best_combined:
                best_combined = result["combined_score"]
                best_result = result
                best_chunk = chunk

        if not best_result or best_result["has_answer_prob"] < 0.5 or not best_result["raw_span"]:
            return {
                "found": False,
                "raw_span": "",
                "section": None,
                "page_start": None,
                "page_end": None,
                "has_answer_prob": best_result["has_answer_prob"] if best_result else 0.0,
                "confidence": "Low",
                "source_display": "Not found in document."
            }

        # Decode, Clean, and Entity Boundary Fix
        raw_span = best_result["raw_span"]
        
        # We need the character index where the span ends in the context to do the boundary fix
        pos = best_result["domain_context"].find(raw_span)
        if pos >= 0:
            extract_end_idx = pos + len(raw_span)
            raw_span = self._fix_entity_boundaries(raw_span, best_result["domain_context"], extract_end_idx)

        # Confidence banding
        prob = best_result["has_answer_prob"]
        confidence = "Low"
        if prob > 0.80:
            confidence = "High"
        elif prob > 0.65:
            confidence = "Medium"

        section = best_chunk.get("section", "Unknown Section")
        page_start = best_chunk.get("page_start", "?")
        page_end = best_chunk.get("page_end", "?")

        return {
            "found": True,
            "raw_span": raw_span,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
            "has_answer_prob": prob,
            "confidence": confidence,
            "source_display": f"Found in {section}, Pages {page_start}-{page_end}"
        }
