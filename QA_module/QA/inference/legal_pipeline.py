import torch
from dataclasses import dataclass, field
from typing import List, Optional

from QA.qa_config import (
    MAX_QUESTION_LEN, MAX_CONTEXT_LEN, MAX_TOTAL_LEN, PAD_ID,
    LEGAL_MAX_ANSWER_LEN, LEGAL_LENGTH_PENALTY, LEGAL_TOKEN,
)
from QA.inference.retriever import TfidfChunkRetriever
from QA.inference.legal_entity import extend_span_to_entities
from QA.inference.plain_english import PlainEnglishRewriter, NoopRewriter
from QA.training.span_select import joint_span_select


# =========================
# CONSTANT
# =========================
NOT_FOUND = (
    "This information was not found in the document you provided. "
    "The document may not contain a clause addressing this question."
)


# =========================
# MEMORY
# =========================
@dataclass
class ConversationMemory:
    history: List[tuple] = field(default_factory=list)
    max_turns: int = 3

    def append(self, question: str, answer: str):
        self.history.append((question, answer))
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def recent_questions(self) -> List[str]:
        return [q for (q, _) in self.history]


# =========================
# PIPELINE
# =========================
class LegalQAPipeline:

    def __init__(
        self,
        model,
        tokenizer,
        qa_meta: dict,
        device,
        rewriter: Optional[PlainEnglishRewriter] = None,
        top_k: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.qa_meta = qa_meta
        self.device = device
        self.rewriter = rewriter or NoopRewriter()
        self.top_k = top_k
        self.memory = ConversationMemory()

        self.legal_id = tokenizer.token_to_id(LEGAL_TOKEN)
        if self.legal_id is None:
            raise RuntimeError(f"{LEGAL_TOKEN} token missing from tokenizer")

    # =========================
    # INPUT BUILDER
    # =========================
    def _build_input(self, question: str, chunk_text: str):

        q_ids = self.tokenizer.encode(question).ids[:MAX_QUESTION_LEN]
        c_ids = self.tokenizer.encode(chunk_text).ids[:MAX_CONTEXT_LEN]

        input_ids = (
            [self.qa_meta["cls_id"]] + q_ids + [self.qa_meta["sep_id"]]
            + [self.legal_id] + c_ids + [self.qa_meta["sep_id"]]
        )

        seg = [0] * (1 + len(q_ids) + 1) + [1] * (1 + len(c_ids) + 1)
        attn = [1] * len(input_ids)

        pad = MAX_TOTAL_LEN - len(input_ids)
        if pad > 0:
            input_ids += [PAD_ID] * pad
            seg += [1] * pad
            attn += [0] * pad
        else:
            input_ids = input_ids[:MAX_TOTAL_LEN]
            seg = seg[:MAX_TOTAL_LEN]
            attn = attn[:MAX_TOTAL_LEN]

        ctx_base = 1 + len(q_ids) + 1 + 1
        return input_ids, seg, attn, ctx_base

    # =========================
    # STRONG VALIDATION (KEY FIX)
    # =========================
    def _validate_answer(self, question: str, context: str):

        q = question.lower()
        ctx = context.lower()

        # 🔥 STRICT PRICE CHECK
        if "price" in q or "cost" in q or "amount" in q:
            if any(x in ctx for x in ["$", "usd", "rs", "rupees", "price", "amount"]):
                return True
            return False

        # 🔥 STRICT TIME CHECK
        if "when" in q:
            if any(x in ctx for x in ["year", "month", "day", "202", "203"]):
                return True
            return False

        return True

    # =========================
    # MAIN ANSWER FUNCTION
    # =========================
    @torch.no_grad()
    def answer(self, document: str, question: str) -> dict:

        self.model.eval()

        # ---------- RETRIEVAL ----------
        retriever = TfidfChunkRetriever(document)
        retrieved = retriever.top_k(
            question,
            k=self.top_k,
            history=self.memory.recent_questions()
        )

        if not retrieved:
            return {"answer": NOT_FOUND, "confidence": 0.0, "source_chunk": None}

        # ---------- BUILD INPUT ----------
        inputs, segs, attns, bases = [], [], [], []

        for r in retrieved:
            ii, ss, aa, cb = self._build_input(question, r.chunk.text)
            inputs.append(ii)
            segs.append(ss)
            attns.append(aa)
            bases.append(cb)

        input_ids = torch.tensor(inputs, dtype=torch.long, device=self.device)
        segment_ids = torch.tensor(segs, dtype=torch.long, device=self.device)
        attn_mask = torch.tensor(attns, dtype=torch.long, device=self.device)

        # ---------- MODEL ----------
        start_logits, end_logits, has_logits = self.model(
            input_ids, segment_ids, attn_mask
        )

        # ---------- SPAN ----------
        s_idx, e_idx = joint_span_select(
            start_logits,
            end_logits,
            max_answer_len=LEGAL_MAX_ANSWER_LEN,
            length_penalty=LEGAL_LENGTH_PENALTY,
        )

        best = None

        for k in range(len(retrieved)):

            i = int(s_idx[k].item())
            j = int(e_idx[k].item())

            if i == 0 and j == 0:
                continue

            span_ids = input_ids[k, i:j+1].tolist()
            span_text = self.tokenizer.decode(span_ids).strip()

            if best is None or len(span_text) > len(best["span_text"]):
                best = {
                    "chunk": retrieved[k].chunk,
                    "span_text": span_text
                }

        if best is None:
            return {"answer": NOT_FOUND, "confidence": 0.0}

        # 🔥 DEBUG (to confirm working)
        print("DEBUG QUESTION:", question)
        print("DEBUG CONTEXT:", best["chunk"].text)
        print("DEBUG VALIDATION:", self._validate_answer(question, best["chunk"].text))

        # ---------- VALIDATION ----------
        if not self._validate_answer(question, best["chunk"].text):
            self.memory.append(question, NOT_FOUND)
            return {"answer": NOT_FOUND, "confidence": 0.0}

        # ---------- ENTITY FIX ----------
        chunk_text = best["chunk"].text
        idx = chunk_text.find(best["span_text"])

        if idx >= 0:
            new_s, new_e = extend_span_to_entities(
                chunk_text,
                idx,
                idx + len(best["span_text"])
            )
            extracted = chunk_text[new_s:new_e]
        else:
            extracted = best["span_text"]

        # ---------- REWRITE ----------
        plain = self.rewriter.rewrite(question, extracted)

        self.memory.append(question, plain)

        return {
            "answer": plain,
            "raw_span": extracted,
            "confidence": 1.0,
            "source_chunk": best["chunk"],
        }

    def reset_memory(self):
        self.memory = ConversationMemory()
