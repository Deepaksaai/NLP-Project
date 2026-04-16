"""
Full Encoder-Decoder Transformer for abstractive summarization.

Assembles all components:
  - Shared embedding (encoder + decoder + output projection via weight tying)
  - Sinusoidal positional encoding
  - 6-layer encoder, 4-layer decoder
  - Mask construction (padding + causal)

All weights trained from scratch. No pretrained components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


class TransformerSummarizer(nn.Module):

    def __init__(
        self,
        vocab_size=32000,
        d_model=384,
        n_heads=6,
        n_encoder_layers=6,
        n_decoder_layers=4,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

        # Shared embedding for encoder and decoder input
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Encoder: 6 layers
        self.encoder = Encoder(
            n_layers=n_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Decoder: 4 layers
        self.decoder = Decoder(
            n_layers=n_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Output projection — tied with embedding weights
        # logits = decoder_output @ embedding.weight.T + out_bias
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all multi-dim parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------------- Masks --------------------
    def make_src_mask(self, src):
        """
        Padding mask for encoder.
        src: (batch, src_len)
        Returns: (batch, 1, 1, src_len) bool — True for real tokens
        """
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """
        Combined padding + causal mask for decoder self-attention.
        tgt: (batch, tgt_len)
        Returns: (batch, 1, tgt_len, tgt_len) bool
        """
        tgt_len = tgt.size(1)

        # Padding mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: (1, 1, tgt_len, tgt_len) — lower triangular
        causal = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

        return pad_mask & causal

    @staticmethod
    def make_causal_mask(seq_len, device=None):
        """
        Standalone causal mask for validation checks.
        Returns: (1, 1, seq_len, seq_len) lower-triangular bool
        """
        return torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

    # -------------------- Encode / Decode --------------------
    def encode(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        return self.decoder(x, memory, tgt_mask, src_mask)

    def project(self, decoder_output):
        """Tied output projection: decoder_output @ embedding.T + bias."""
        return decoder_output @ self.embedding.weight.t() + self.out_bias

    # -------------------- Forward --------------------
    def forward(self, src, tgt):
        """
        Args:
            src: (batch, src_len) — source token IDs
            tgt: (batch, tgt_len) — target token IDs (decoder input)
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.project(out)

    # -------------------- Inference --------------------
    @torch.no_grad()
    def greedy_generate(self, src, max_len=120, sos_idx=1, eos_idx=2):
        """Greedy decoding — works for any batch size."""
        self.eval()
        device = src.device
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(tgt)
            out = self.decode(tgt, memory, tgt_mask, src_mask)
            logits = self.project(out[:, -1])
            next_token = logits.argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == eos_idx).all():
                break

        return tgt

    @torch.no_grad()
    def beam_search(self, src, beam_width=4, max_len=120, sos_idx=1,
                    eos_idx=2, length_penalty=0.7):
        """Beam search for batch_size=1."""
        self.eval()
        assert src.size(0) == 1, "beam_search expects batch_size=1"

        device = src.device
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)

        beams = [(torch.tensor([[sos_idx]], device=device), 0.0, False)]
        completed = []

        for _ in range(max_len - 1):
            candidates = []

            for tokens, score, done in beams:
                if done:
                    completed.append((tokens, score))
                    continue

                tgt_mask = self.make_tgt_mask(tokens)
                out = self.decode(tokens, memory, tgt_mask, src_mask)
                logits = self.project(out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_logp, top_idx = log_probs.topk(beam_width)

                for k in range(beam_width):
                    tok = top_idx[k].item()
                    new_score = score + top_logp[k].item()
                    new_tokens = torch.cat(
                        [tokens, torch.tensor([[tok]], device=device)], dim=1
                    )
                    candidates.append((new_tokens, new_score, tok == eos_idx))

            def lp_score(item):
                toks, sc, _ = item
                lp = ((5 + toks.size(1)) / 6) ** length_penalty
                return sc / lp

            candidates.sort(key=lp_score, reverse=True)
            beams = candidates[:beam_width]

            if all(b[2] for b in beams):
                break

        completed.extend(beams)

        def final_score(item):
            toks, sc = item[0], item[1]
            lp = ((5 + toks.size(1)) / 6) ** length_penalty
            return sc / lp

        best = max(completed, key=final_score)
        return best[0].squeeze(0).tolist()


def build_model(device="cpu", **kwargs):
    """Factory: build model, move to device, report parameter count."""
    model = TransformerSummarizer(**kwargs).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built — {total:,} trainable parameters")
    return model
