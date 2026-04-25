"""
Generator for translating legal spans into Plain English.
"""

import logging
import re
from transformers import pipeline

logger = logging.getLogger(__name__)

class PlainEnglishGenerator:
    """
    Seq2Seq model wrap to rephrase legal extracts into non-lawyer terms.
    """

    def __init__(self, device_id: int = -1):
        """
        Loads facebook/bart-base (139M params).
        """
        logger.info("Loading BART-base for Plain English generation...")
        
        # device_id = -1 means CPU, >= 0 means GPU
        self.generator = pipeline(
            "text2text-generation", 
            model="facebook/bart-base",
            device=device_id
        )
        logger.info("BART-base loaded.")

    def _extract_critical_entities(self, text: str) -> set:
        """
        Extracts numbers, dates, and capitalized terms for faithfulness checking.
        """
        entities = set()
        
        # 1. Numbers / Dates (digits, potentially mixed with slashes or dashes)
        numbers = re.findall(r'\b\d+(?:[.,/-]\d+)*\b', text)
        entities.update(numbers)
        
        # 2. Capitalized terms (ignoring the first word of the sentence to reduce false positives)
        words = text.split()
        if len(words) > 1:
            for w in words[1:]:
                # Check if starts with a capital letter and isn't just punctuation
                clean_w = ''.join(c for c in w if c.isalnum())
                if clean_w and clean_w[0].isupper():
                    entities.add(clean_w)
                    
        return entities

    def generate(self, question: str, raw_span: str) -> str:
        """
        Generates translation and enforces faithfulness checks.
        """
        if not raw_span:
            return ""

        prompt = f"Explain this legal text in simple terms that a non-lawyer can understand. Question: {question} Legal text: {raw_span}"
        
        logger.debug("Running generation...")
        results = self.generator(
            prompt,
            max_new_tokens=150,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        
        generated_text = results[0]["generated_text"].strip()
        logger.debug(f"Generated text: {generated_text}")

        # Faithfulness check: A simplified text must not omit numbers or party names
        entities = self._extract_critical_entities(raw_span)
        missing_entities = []
        for e in entities:
            # Simple check if the core alphanumeric part exists in the generated output
            if e.lower() not in generated_text.lower():
                missing_entities.append(e)
                
        if missing_entities:
            logger.warning(f"Faithfulness check failed. Missing critical terms: {missing_entities}")
            logger.warning("Falling back strictly to raw_span to prevent hallucinations.")
            return raw_span
            
        return generated_text
