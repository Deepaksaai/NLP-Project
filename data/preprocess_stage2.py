"""
Stage 2 Data Preprocessing — Chunk-Target Alignment.

This script runs ONCE before Stage 2 training. It:
  1. Loads arXiv + PubMed papers
  2. Chunks each paper into overlapping segments
  3. Splits each abstract into sentences
  4. Uses MiniLM to compute cosine similarity between chunks and sentences
  5. Assigns each chunk its top-3 most relevant abstract sentences
  6. Filters out low-quality alignments (cosine < 0.25)
  7. Generates second-pass examples (concatenated chunk targets → full abstract)
  8. Saves everything to disk as JSON

MiniLM is used ONLY here as a preprocessing tool. It is discarded
after this script runs and never loaded during training.

Usage:
    python -m data.preprocess_stage2
"""

import os
import sys
import json
import re
import math
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------------------------------------
# Config
# -------------------------------------------------------
CHUNK_WORDS = 350
OVERLAP_WORDS = 50
MAX_CHUNKS = 8
MIN_COSINE_SIM = 0.25
TOP_K_SENTENCES = 3

OUTPUT_DIR = "data/stage2_aligned"
ARXIV_OUTPUT = os.path.join(OUTPUT_DIR, "arxiv_aligned.jsonl")
PUBMED_OUTPUT = os.path.join(OUTPUT_DIR, "pubmed_aligned.jsonl")
SECOND_PASS_OUTPUT = os.path.join(OUTPUT_DIR, "second_pass.jsonl")
STATS_OUTPUT = os.path.join(OUTPUT_DIR, "alignment_stats.json")


# -------------------------------------------------------
# Text utilities
# -------------------------------------------------------
def chunk_document(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    stride = chunk_words - overlap_words
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start:start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def split_sentences(text):
    """Simple regex sentence splitter for abstracts."""
    # Protect common abbreviations
    protected = text
    abbrevs = ["Dr.", "Mr.", "Mrs.", "Prof.", "Fig.", "Eq.", "et al.",
               "i.e.", "e.g.", "vs.", "No.", "Vol."]
    for a in abbrevs:
        protected = protected.replace(a, a.replace(".", "<DOT>"))

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return [s for s in sentences if len(s.split()) >= 5]


# -------------------------------------------------------
# MiniLM embedding
# -------------------------------------------------------
class MiniLMEmbedder:
    """
    Thin wrapper around sentence-transformers MiniLM.
    Used ONLY for preprocessing, discarded after.
    """

    def __init__(self, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        self.device = device
        print("MiniLM loaded for chunk-target alignment")

    def encode(self, texts, batch_size=64):
        """Encode a list of strings into normalized embeddings."""
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        )


def align_chunks_to_abstract(chunks, abstract_sentences, embedder):
    """
    For each chunk, find the top-K most similar abstract sentences.

    Returns list of dicts:
      {
        "chunk_idx": int,
        "chunk_text": str,
        "target_sentences": [str, ...],
        "max_similarity": float,
      }
    """
    if not chunks or not abstract_sentences:
        return []

    chunk_embs = embedder.encode(chunks)
    sent_embs = embedder.encode(abstract_sentences)

    # Cosine similarity matrix: (n_chunks, n_sentences)
    # Embeddings are already normalized, so dot product = cosine sim
    sim_matrix = chunk_embs @ sent_embs.T

    aligned = []
    for i, chunk in enumerate(chunks):
        sims = sim_matrix[i]
        max_sim = float(sims.max())

        # Skip low-quality chunks
        if max_sim < MIN_COSINE_SIM:
            continue

        # Top-K sentence indices, preserving original abstract order
        top_k_idx = sims.argsort()[-TOP_K_SENTENCES:][::-1]
        top_k_idx_sorted = sorted(top_k_idx)

        target_sents = [abstract_sentences[j] for j in top_k_idx_sorted]

        aligned.append({
            "chunk_idx": i,
            "chunk_text": chunk,
            "target_sentences": target_sents,
            "target": " ".join(target_sents),
            "max_similarity": round(max_sim, 4),
        })

    return aligned


# -------------------------------------------------------
# Process a dataset
# -------------------------------------------------------
def process_dataset(dataset_name, split_data, embedder, output_path,
                    text_key="article", abstract_key="abstract"):
    """
    Process an entire dataset split: chunk, align, save.
    Returns stats dict and list of per-paper aligned chunks.
    """
    print(f"\nProcessing {dataset_name} ({len(split_data)} papers)...")

    total_papers = 0
    total_chunks = 0
    total_aligned = 0
    total_filtered = 0
    all_second_pass = []

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, paper in enumerate(tqdm(split_data, desc=dataset_name)):
            text = paper[text_key]
            abstract = paper[abstract_key]

            if not text or not abstract:
                continue
            if len(text.split()) < 50 or len(abstract.split()) < 10:
                continue

            total_papers += 1

            # Chunk the paper
            chunks = chunk_document(text)[:MAX_CHUNKS]
            total_chunks += len(chunks)

            # Split abstract into sentences
            abstract_sents = split_sentences(abstract)
            if not abstract_sents:
                continue

            # Align chunks to abstract sentences
            aligned = align_chunks_to_abstract(chunks, abstract_sents, embedder)
            total_aligned += len(aligned)
            total_filtered += (len(chunks) - len(aligned))

            # Write aligned examples
            for item in aligned:
                record = {
                    "input": item["chunk_text"],
                    "target": item["target"],
                    "score": item["max_similarity"],
                    "doc_id": f"{dataset_name}_{idx}",
                    "chunk_i": item["chunk_idx"],
                    "type": "chunk",
                }
                f.write(json.dumps(record) + "\n")

            # Generate second-pass example for this paper
            if len(aligned) >= 2:
                concat_targets = " ".join(
                    item["target"] for item in aligned
                )
                second_pass = {
                    "input": concat_targets,
                    "target": abstract,
                    "doc_id": f"{dataset_name}_{idx}",
                    "type": "second_pass",
                }
                all_second_pass.append(second_pass)

            if (idx + 1) % 5000 == 0:
                pct = total_aligned / max(total_chunks, 1) * 100
                print(f"  [{idx+1}] chunks={total_chunks}, "
                      f"aligned={total_aligned} ({pct:.1f}%), "
                      f"filtered={total_filtered}")

    coverage = total_aligned / max(total_chunks, 1) * 100
    stats = {
        "dataset": dataset_name,
        "papers": total_papers,
        "total_chunks": total_chunks,
        "aligned_chunks": total_aligned,
        "filtered_chunks": total_filtered,
        "coverage_pct": round(coverage, 1),
        "second_pass_examples": len(all_second_pass),
    }

    print(f"  {dataset_name} done: {total_aligned}/{total_chunks} chunks aligned "
          f"({coverage:.1f}%), {len(all_second_pass)} second-pass examples")

    return stats, all_second_pass


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("Stage 2 Preprocessing: Chunk-Target Alignment")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = MiniLMEmbedder(device=device)

    from datasets import load_dataset

    all_stats = []
    all_second_pass = []

    # ---- arXiv ----
    print("\nLoading arXiv...")
    try:
        arxiv = load_dataset("ccdv/arxiv-summarization", split="train")
        text_key = "article" if "article" in arxiv.column_names else "text"
        abstract_key = "abstract" if "abstract" in arxiv.column_names else "summary"
        stats, sp = process_dataset(
            "arxiv", arxiv, embedder, ARXIV_OUTPUT,
            text_key=text_key, abstract_key=abstract_key
        )
        all_stats.append(stats)
        all_second_pass.extend(sp)
        del arxiv
    except Exception as e:
        print(f"  arXiv failed: {e}")

    # ---- PubMed ----
    print("\nLoading PubMed...")
    try:
        pubmed = load_dataset("ccdv/pubmed-summarization", split="train")
        text_key = "article" if "article" in pubmed.column_names else "text"
        abstract_key = "abstract" if "abstract" in pubmed.column_names else "summary"
        stats, sp = process_dataset(
            "pubmed", pubmed, embedder, PUBMED_OUTPUT,
            text_key=text_key, abstract_key=abstract_key
        )
        all_stats.append(stats)
        all_second_pass.extend(sp)
        del pubmed
    except Exception as e:
        print(f"  PubMed failed: {e}")

    # ---- Save second-pass examples ----
    print(f"\nSaving {len(all_second_pass)} second-pass examples...")
    with open(SECOND_PASS_OUTPUT, "w", encoding="utf-8") as f:
        for ex in all_second_pass:
            f.write(json.dumps(ex) + "\n")

    # ---- Save stats ----
    with open(STATS_OUTPUT, "w") as f:
        json.dump(all_stats, f, indent=2)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['dataset']:10s}: {s['aligned_chunks']:6d} aligned chunks, "
              f"{s['second_pass_examples']:5d} second-pass, "
              f"{s['coverage_pct']}% coverage")
    print(f"  Total second-pass examples: {len(all_second_pass)}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"  {os.path.basename(ARXIV_OUTPUT)}")
    print(f"  {os.path.basename(PUBMED_OUTPUT)}")
    print(f"  {os.path.basename(SECOND_PASS_OUTPUT)}")
    print(f"  {os.path.basename(STATS_OUTPUT)}")

    # Discard MiniLM
    del embedder
    print("\nMiniLM discarded — not used during training")
    print("=" * 60)


if __name__ == "__main__":
    main()
