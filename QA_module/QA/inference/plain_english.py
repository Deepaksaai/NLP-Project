"""
Plain English answer generation — skeleton.

The spec treats this as an optional post-processor trained separately
from the QA model. This module exposes a minimal PlainEnglishRewriter
interface with two backends:

    NoopRewriter:
        Returns the extracted span unchanged. Safe default until a
        real generator is trained.

    SummarizerRewriter(model, tokenizer):
        Uses the existing summarizer TransformerSummarizer as a silver
        rewrite engine: feeds "Question: ... Answer: ..." through
        greedy decoding and returns the generated text. Intended as
        the fallback the spec describes ("use your summarizer model
        to generate plain English versions").

Training a dedicated seq2seq plain-English generator is left as a
separate exercise — no training loop lives in this file.
"""

import torch


class PlainEnglishRewriter:
    """Base interface — override `rewrite`."""
    def rewrite(self, question: str, extracted_span: str) -> str:
        raise NotImplementedError


class NoopRewriter(PlainEnglishRewriter):
    def rewrite(self, question: str, extracted_span: str) -> str:
        return extracted_span


class SummarizerRewriter(PlainEnglishRewriter):
    """
    Wraps the existing summarizer model. `summarizer` must be a
    TransformerSummarizer instance from model/transformer.py.
    """
    def __init__(self, summarizer, tokenizer, device, max_len: int = 120):
        self.summarizer = summarizer
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def rewrite(self, question: str, extracted_span: str) -> str:
        self.summarizer.eval()
        prompt = f"Question: {question} Answer: {extracted_span}"
        ids = self.tokenizer.encode(prompt).ids[: 400]
        src = torch.tensor([ids], dtype=torch.long, device=self.device)

        out_ids = self.summarizer.greedy_generate(
            src, max_len=self.max_len, sos_idx=1, eos_idx=2,
        )
        # Strip SOS/EOS
        toks = out_ids[0].tolist()
        if toks and toks[0] == 1:
            toks = toks[1:]
        if toks and toks[-1] == 2:
            toks = toks[:-1]
        return self.tokenizer.decode(toks).strip()
