"""
Model architecture for the Legal QA system.

Core class: LegalQAModel  (DeBERTa-v3-large + span heads + has-answer head)
"""

from legal_qa.model.qa_model import LegalQAModel
from legal_qa.model.heads import SpanHead, HasAnswerHead

__all__ = ["LegalQAModel", "SpanHead", "HasAnswerHead"]
