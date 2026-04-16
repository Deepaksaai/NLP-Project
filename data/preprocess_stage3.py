"""
Stage 3 Data Preprocessing — Legal Chunk-Target Alignment + Second-Pass.

Runs ONCE before Stage 3 training. Steps:
  1. Load MultiLexSum (v1 + v2, deduplicated by case_id) and BillSum
  2. Per-document pairing for MultiLexSum (each source doc → case summary)
  3. Chunk legal documents (350w, 50w overlap, max 6 chunks)
  4. MiniLM alignment with legal-specific params:
       - min_cosine_threshold = 0.20 (lower than Stage 2 — legal text is sparse)
       - top-3 sentences per chunk
       - max_chunks = 6 (denser than scientific papers)
  5. Generate second-pass examples by concatenating chunk targets
     (no model inference needed — uses MiniLM-aligned sentences directly)
  6. Case-level train/val split (no case_id leakage between sets)

The <legal> token is NOT prepended here — it's added at training time
in the dataset class so we can reuse the same files for evaluation.

Usage:
    python -m data.preprocess_stage3
"""

import os
import sys
import json
import re
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------------------------------------
# Config
# -------------------------------------------------------
CHUNK_WORDS = 350
OVERLAP_WORDS = 50
MAX_CHUNKS = 6
MIN_COSINE_SIM = 0.20
TOP_K_SENTENCES = 3

# Second-pass quality filters
MIN_CONCAT_WORDS = 50
MIN_CHUNKS_FOR_SECOND_PASS = 3

OUTPUT_DIR = "data/stage3_aligned"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "legal_train.jsonl")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "legal_val.jsonl")
SECOND_PASS_OUTPUT = os.path.join(OUTPUT_DIR, "legal_second_pass.jsonl")
STATS_OUTPUT = os.path.join(OUTPUT_DIR, "alignment_stats.json")

# MultiLexSum local files (v1 + v2)
MULTILEXSUM_FILES = [
    {"train": "datasets/multilexsum/train_1.json", "test": "datasets/multilexsum/test_1.json", "sources": "datasets/multilexsum/sources_1.json"},
    {"train": "datasets/multilexsum/train_2.json", "test": "datasets/multilexsum/test_2.json", "sources": "datasets/multilexsum/sources_2.json"},
]


# -------------------------------------------------------
# Text utilities
# -------------------------------------------------------
def chunk_document(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
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
    protected = text
    abbrevs = ["Dr.", "Mr.", "Mrs.", "Prof.", "Inc.", "Corp.", "Ltd.", "LLC",
               "No.", "v.", "vs.", "et al.", "i.e.", "e.g.", "U.S.", "Sec.",
               "Art.", "Para.", "Co.", "Ch."]
    for a in abbrevs:
        protected = protected.replace(a, a.replace(".", "<DOT>"))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return [s for s in sentences if len(s.split()) >= 5]


# -------------------------------------------------------
# MiniLM alignment
# -------------------------------------------------------
class MiniLMEmbedder:
    def __init__(self, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        print("MiniLM loaded for legal alignment")

    def encode(self, texts, batch_size=64):
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        )


def align_chunks_to_summary(chunks, summary_sentences, embedder):
    """For each chunk, find top-3 most similar summary sentences."""
    if not chunks or not summary_sentences:
        return []

    chunk_embs = embedder.encode(chunks)
    sent_embs = embedder.encode(summary_sentences)
    sim_matrix = chunk_embs @ sent_embs.T

    aligned = []
    for i, chunk in enumerate(chunks):
        sims = sim_matrix[i]
        max_sim = float(sims.max())
        if max_sim < MIN_COSINE_SIM:
            continue
        top_k_idx = sims.argsort()[-TOP_K_SENTENCES:][::-1]
        top_k_idx_sorted = sorted(top_k_idx)
        target_sents = [summary_sentences[j] for j in top_k_idx_sorted]
        aligned.append({
            "chunk_idx": i,
            "chunk_text": chunk,
            "target_sentences": target_sents,
            "target": " ".join(target_sents),
            "max_similarity": round(max_sim, 4),
        })
    return aligned


# -------------------------------------------------------
# MultiLexSum loading
# -------------------------------------------------------
def load_multilexsum_cases():
    """
    Load all MultiLexSum cases from local JSONL files.
    Deduplicates by case_id across v1 and v2.
    Returns: train_cases dict, val_cases dict (case_id -> case data)
    """
    print("Loading MultiLexSum cases...")
    train_cases = {}
    val_cases = {}
    sources_lookup = {}

    for fg in MULTILEXSUM_FILES:
        if not os.path.exists(fg["sources"]):
            print(f"  Skipping {fg['sources']} (not found)")
            continue

        # Load sources
        print(f"  Loading {fg['sources']}...")
        with open(fg["sources"], "r", encoding="utf-8") as f:
            sources = json.load(f)

        # Add to global lookup
        for doc_id, entry in sources.items():
            if doc_id not in sources_lookup:
                if isinstance(entry, dict):
                    text = entry.get("doc_text") or entry.get("text", "")
                elif isinstance(entry, str):
                    text = entry
                else:
                    text = ""
                if text:
                    sources_lookup[doc_id] = text
        del sources

        # Load train cases
        with open(fg["train"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                cid = case.get("case_id")
                if cid and cid not in train_cases and cid not in val_cases:
                    train_cases[cid] = case

        # Load val cases (test split)
        with open(fg["test"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                cid = case.get("case_id")
                if cid and cid not in val_cases and cid not in train_cases:
                    val_cases[cid] = case

    print(f"  Train cases: {len(train_cases)}")
    print(f"  Val cases:   {len(val_cases)}")
    print(f"  Source documents indexed: {len(sources_lookup)}")
    return train_cases, val_cases, sources_lookup


def process_multilexsum_split(cases, sources_lookup, embedder, split_name):
    """
    Process one MultiLexSum split (train or val).
    Per-doc pairing: each source document paired with case summary.
    Returns list of aligned chunk records and list of second-pass records.
    """
    all_aligned = []
    all_second_pass = []

    for case_id, case in tqdm(cases.items(), desc=f"MultiLexSum {split_name}"):
        summary = case.get("summary/long") or case.get("summary/short")
        if not summary:
            continue

        summary_sents = split_sentences(summary)
        if not summary_sents:
            continue

        doc_ids = case.get("case_documents") or []

        # Process each source document separately (per-doc pairing)
        case_chunk_targets = []
        for doc_id in doc_ids:
            doc_text = sources_lookup.get(doc_id)
            if not doc_text or len(doc_text.split()) < 50:
                continue

            chunks = chunk_document(doc_text)[:MAX_CHUNKS]
            aligned = align_chunks_to_summary(chunks, summary_sents, embedder)

            for item in aligned:
                record = {
                    "input": item["chunk_text"],
                    "target": item["target"],
                    "score": item["max_similarity"],
                    "case_id": case_id,
                    "doc_id": doc_id,
                    "chunk_i": item["chunk_idx"],
                    "type": "legal_chunk",
                    "source": "multilexsum",
                }
                all_aligned.append(record)
                case_chunk_targets.append(item["target"])

        # Second-pass: concatenated chunk targets → full summary
        if (
            split_name == "train"
            and len(case_chunk_targets) >= MIN_CHUNKS_FOR_SECOND_PASS
        ):
            concat = " ".join(case_chunk_targets)
            if len(concat.split()) >= MIN_CONCAT_WORDS:
                all_second_pass.append({
                    "input": concat,
                    "target": summary,
                    "case_id": case_id,
                    "type": "second_pass",
                    "source": "multilexsum",
                })

    return all_aligned, all_second_pass


# -------------------------------------------------------
# BillSum loading
# -------------------------------------------------------
def process_billsum(embedder):
    """Load BillSum, chunk-align with MiniLM, generate second-pass for multi-chunk bills."""
    from datasets import load_dataset

    print("Loading BillSum...")
    try:
        bs = load_dataset("billsum")
    except Exception as e:
        print(f"  BillSum load failed: {e}")
        return [], [], [], []

    train_aligned = []
    val_aligned = []
    train_second_pass = []

    print("Processing BillSum train...")
    for idx, ex in enumerate(tqdm(bs["train"], desc="BillSum train")):
        text = ex["text"]
        summary = ex["summary"]
        if not text or not summary:
            continue

        summary_sents = split_sentences(summary)
        if not summary_sents:
            continue

        chunks = chunk_document(text)[:MAX_CHUNKS]
        aligned = align_chunks_to_summary(chunks, summary_sents, embedder)

        chunk_targets = []
        for item in aligned:
            record = {
                "input": item["chunk_text"],
                "target": item["target"],
                "score": item["max_similarity"],
                "case_id": f"billsum_{idx}",
                "doc_id": f"billsum_{idx}",
                "chunk_i": item["chunk_idx"],
                "type": "legal_chunk",
                "source": "billsum",
            }
            train_aligned.append(record)
            chunk_targets.append(item["target"])

        # Second-pass only for multi-chunk bills
        if len(chunk_targets) >= MIN_CHUNKS_FOR_SECOND_PASS:
            concat = " ".join(chunk_targets)
            if len(concat.split()) >= MIN_CONCAT_WORDS:
                train_second_pass.append({
                    "input": concat,
                    "target": summary,
                    "case_id": f"billsum_{idx}",
                    "type": "second_pass",
                    "source": "billsum",
                })

    print("Processing BillSum test...")
    for idx, ex in enumerate(tqdm(bs["test"], desc="BillSum test")):
        text = ex["text"]
        summary = ex["summary"]
        if not text or not summary:
            continue
        summary_sents = split_sentences(summary)
        if not summary_sents:
            continue

        chunks = chunk_document(text)[:MAX_CHUNKS]
        aligned = align_chunks_to_summary(chunks, summary_sents, embedder)
        for item in aligned:
            record = {
                "input": item["chunk_text"],
                "target": item["target"],
                "score": item["max_similarity"],
                "case_id": f"billsum_test_{idx}",
                "doc_id": f"billsum_test_{idx}",
                "chunk_i": item["chunk_idx"],
                "type": "legal_chunk",
                "source": "billsum",
            }
            val_aligned.append(record)

    return train_aligned, val_aligned, train_second_pass


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("Stage 3 Preprocessing: Legal Alignment + Second-Pass")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = MiniLMEmbedder(device=device)

    all_train = []
    all_val = []
    all_second_pass = []
    stats = {}

    # ---- MultiLexSum ----
    train_cases, val_cases, sources_lookup = load_multilexsum_cases()

    print("\nProcessing MultiLexSum train cases...")
    mlx_train_aligned, mlx_second_pass = process_multilexsum_split(
        train_cases, sources_lookup, embedder, "train"
    )
    all_train.extend(mlx_train_aligned)
    all_second_pass.extend(mlx_second_pass)

    print("\nProcessing MultiLexSum val cases...")
    mlx_val_aligned, _ = process_multilexsum_split(
        val_cases, sources_lookup, embedder, "val"
    )
    all_val.extend(mlx_val_aligned)

    stats["multilexsum"] = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_chunks": len(mlx_train_aligned),
        "val_chunks": len(mlx_val_aligned),
        "second_pass": len(mlx_second_pass),
    }
    print(f"\nMultiLexSum: {len(mlx_train_aligned)} train chunks, "
          f"{len(mlx_val_aligned)} val chunks, {len(mlx_second_pass)} second-pass")

    # Free sources
    del sources_lookup, train_cases, val_cases

    # ---- BillSum ----
    bs_train_aligned, bs_val_aligned, bs_second_pass = process_billsum(embedder)
    all_train.extend(bs_train_aligned)
    all_val.extend(bs_val_aligned)
    all_second_pass.extend(bs_second_pass)

    stats["billsum"] = {
        "train_chunks": len(bs_train_aligned),
        "val_chunks": len(bs_val_aligned),
        "second_pass": len(bs_second_pass),
    }
    print(f"\nBillSum: {len(bs_train_aligned)} train chunks, "
          f"{len(bs_val_aligned)} val chunks, {len(bs_second_pass)} second-pass")

    # ---- Verify case-level disjoint split ----
    train_cids = {ex["case_id"] for ex in all_train}
    val_cids = {ex["case_id"] for ex in all_val}
    overlap = train_cids & val_cids
    if overlap:
        print(f"\nWARNING: {len(overlap)} case_ids appear in both train and val!")
    else:
        print(f"\nCase-level split verified: train={len(train_cids)}, "
              f"val={len(val_cids)}, no overlap")

    stats["disjoint_split"] = len(overlap) == 0
    stats["unique_train_cases"] = len(train_cids)
    stats["unique_val_cases"] = len(val_cids)

    # ---- Save ----
    print(f"\nSaving train ({len(all_train)} examples)...")
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for ex in all_train:
            f.write(json.dumps(ex) + "\n")

    print(f"Saving val ({len(all_val)} examples)...")
    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for ex in all_val:
            f.write(json.dumps(ex) + "\n")

    print(f"Saving second-pass ({len(all_second_pass)} examples)...")
    with open(SECOND_PASS_OUTPUT, "w", encoding="utf-8") as f:
        for ex in all_second_pass:
            f.write(json.dumps(ex) + "\n")

    with open(STATS_OUTPUT, "w") as f:
        json.dump(stats, f, indent=2)

    del embedder
    print("\nMiniLM discarded — not used during training")

    print("\n" + "=" * 60)
    print("Stage 3 Preprocessing Complete")
    print("=" * 60)
    print(f"  Total train chunks   : {len(all_train)}")
    print(f"  Total val chunks     : {len(all_val)}")
    print(f"  Total second-pass    : {len(all_second_pass)}")
    print(f"  Output dir           : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
