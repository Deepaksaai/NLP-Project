"""
TF-IDF Retrieval system for Legal QA.
"""

import logging
from typing import List, Dict, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

# Hardcoded synonym dictionary for expanding plain English questions to legal vocabulary
LEGAL_SYNONYMS = {
    "cancel": "terminate",
    "canceling": "termination",
    "fine": "penalty",
    "sign": "execute",
    "signed": "executed",
    "start": "effective date",
    "end": "expiration",
    "ending": "expiration",
    "rule": "provision",
    "rules": "provisions",
    "pay": "compensate",
    "payment": "compensation",
    "law": "governing law",
    "court": "jurisdiction",
    "promise": "covenant",
    "promises": "covenants",
    "lie": "misrepresentation",
    "change": "amend",
    "changes": "amendments",
    "force": "enforce",
    "stop": "cease",
    "responsibility": "liability",
    "responsible": "liable"
}

class LegalRetriever:
    """
    TF-IDF based semantic retrieval system augmented with legal synonyms.
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        """
        Initializes and fits the TF-IDF index over all document chunks.

        Args:
            chunks: A list of chunk dictionaries. Each chunk must have at least a 'text' or 'body'
                    and potentially a 'section' heading.
        """
        self.chunks = chunks
        self.chunk_texts = [c.get("body", c.get("text", "")) for c in self.chunks]
        
        logger.info(f"Building TF-IDF over {len(self.chunk_texts)} chunks...")
        
        # We deliberately do not remove stopwords because in legal text, words like "shall", "not", "or" 
        # completely change the meaning. Sublinear tf helps manage repeating boilerplate text.
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,
            sublinear_tf=True
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        logger.info("TF-IDF Vectorizer fitted successfully.")

    def _expand_query(self, query: str) -> str:
        """Expands plain English words into legal synonyms for better retrieval."""
        words = query.lower().split()
        expanded_words = []
        for w in words:
            # Strip punctuation for lookup
            clean_w = ''.join(c for c in w if c.isalnum())
            if clean_w in LEGAL_SYNONYMS:
                # Add both the original word and the synonym
                expanded_words.append(w)
                expanded_words.append(LEGAL_SYNONYMS[clean_w])
            else:
                expanded_words.append(w)
        return " ".join(expanded_words)

    def retrieve(self, question: str, conversation_history: List[Tuple[str, str]] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top k most relevant chunks for the given question.
        
        Args:
            question: The user's question.
            conversation_history: List of (past_question, past_answer) tuples.
            k: Top k results to return.
        
        Returns:
            List of chunk dicts including their 'retrieval_score'.
        """
        # 1. Augment Query with history (Coreference handling)
        augmented_query = ""
        if conversation_history:
            last_two_turns = conversation_history[-2:]
            history_str = " ".join([f"Q: {q} A: {a}" for q, a in last_two_turns])
            augmented_query = f"{history_str} Q: {question}"
        else:
            augmented_query = question

        # 2. Expand Query with Legal Synonyms
        expanded_query = self._expand_query(augmented_query)
        logger.debug(f"Retrieval Query Expanded: {expanded_query}")

        # 3. Compute Cosine Similarity
        query_vec = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 4. Section-Weighted Boost
        query_words = set(expanded_query.lower().split())
        boosted_scores = []
        for i, chunk in enumerate(self.chunks):
            score = similarities[i]
            section_heading = chunk.get("section", "").lower()
            if section_heading:
                section_words = set(section_heading.split())
                # If any word from the section heading (longer than 3 chars to avoid "of", "the", etc.) appears in the query
                overlap = [w for w in section_words if w in query_words and len(w) > 3]
                if overlap:
                    score *= 1.3
            boosted_scores.append((i, score))

        # 5. Sort and return top K
        boosted_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = boosted_scores[:k]
        
        results = []
        for idx, score in top_k:
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["retrieval_score"] = float(score)
            results.append(chunk_copy)
            
        logger.info(f"Retrieved {len(results)} chunks. Top score: {results[0]['retrieval_score']:.4f} if results else 0.")
        return results
