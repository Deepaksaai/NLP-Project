"""
Stage 4 Step 1 — Generate test set outputs from the trained model.

Loads stage3_best.pt, runs the hierarchical inference pipeline on every
test example, and saves the results to disk so that all downstream
metrics can be computed without re-running inference.

Test set composition:
  - MultiLexSum test cases (case-level, never seen during training)
  - BillSum test bills

For MultiLexSum, we pick the longest source document per case as the
representative input. This avoids the multi-doc complexity while still
exercising the hierarchical pipeline.

Outputs:
  evaluation/generated_summaries.json   {test_id: generated_summary}
  evaluation/reference_summaries.json   {test_id: reference}
  evaluation/source_documents.json      {test_id: full source text}
  evaluation/test_meta.json             {test_id: source_dataset, ...}

Usage:
    python -m evaluation.generate_outputs              # default 500 examples
    python -m evaluation.generate_outputs --n 100      # 100 examples (faster)
    python -m evaluation.generate_outputs --n 5000     # full eval (very slow)
"""

import os
import sys
import json
import argparse
import time
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import build_model
from data.preprocess import load_tokenizer, PAD_ID
from inference.hierarchical import HierarchicalSummarizer


# -------------------------------------------------------
# Test set assembly
# -------------------------------------------------------
def load_multilexsum_test(n_max=100):
    """
    Memory-efficient MultiLexSum test loader.

    Strategy: first collect needed doc_ids from test cases (small files),
    then stream the huge sources.json file and only keep needed entries.
    Caps to n_max cases to keep memory/time bounded.
    """
    print(f"Loading MultiLexSum test (max {n_max} cases, memory-efficient)...")

    files = [
        {"test": "datasets/multilexsum/test_1.json", "sources": "datasets/multilexsum/sources_1.json"},
        {"test": "datasets/multilexsum/test_2.json", "sources": "datasets/multilexsum/sources_2.json"},
    ]

    # Step 1: collect test cases and their needed doc_ids from the small files
    cases_info = []
    seen_cases = set()
    needed_doc_ids = set()

    for fg in files:
        if not os.path.exists(fg["test"]):
            continue
        with open(fg["test"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                cid = case.get("case_id")
                if not cid or cid in seen_cases:
                    continue
                summary = case.get("summary/long") or case.get("summary/short")
                if not summary:
                    continue
                doc_ids = case.get("case_documents") or []
                if not doc_ids:
                    continue
                seen_cases.add(cid)
                cases_info.append({
                    "case_id": cid,
                    "summary": summary,
                    "doc_ids": doc_ids,
                    "source_file": fg["sources"],
                })
                needed_doc_ids.update(doc_ids)
                if len(cases_info) >= n_max:
                    break
        if len(cases_info) >= n_max:
            break

    print(f"  Collected {len(cases_info)} cases needing {len(needed_doc_ids)} docs")

    # Step 2: try ijson streaming first — if not available, fall back to
    # bulk loading but ONLY keep the doc_ids we actually need.
    doc_texts = {}

    try:
        import ijson  # streaming JSON parser
        streaming = True
        print("  Using ijson streaming parser")
    except ImportError:
        streaming = False
        print("  ijson not available, falling back to bulk load with filter")

    for fg in files:
        if not os.path.exists(fg["sources"]):
            continue

        if streaming:
            with open(fg["sources"], "rb") as f:
                for doc_id, entry in ijson.kvitems(f, ""):
                    if doc_id not in needed_doc_ids:
                        continue
                    if isinstance(entry, dict):
                        text = entry.get("doc_text") or entry.get("text", "")
                    elif isinstance(entry, str):
                        text = entry
                    else:
                        text = ""
                    if text:
                        doc_texts[doc_id] = text
                    if len(doc_texts) >= len(needed_doc_ids):
                        break
        else:
            # Bulk load but filter aggressively — can be slow and memory-heavy
            print(f"  Bulk loading {fg['sources']}...")
            with open(fg["sources"], "r", encoding="utf-8") as f:
                data = json.load(f)
            for doc_id in list(needed_doc_ids):
                if doc_id in data and doc_id not in doc_texts:
                    entry = data[doc_id]
                    if isinstance(entry, dict):
                        text = entry.get("doc_text") or entry.get("text", "")
                    elif isinstance(entry, str):
                        text = entry
                    else:
                        text = ""
                    if text:
                        doc_texts[doc_id] = text
            del data

    print(f"  Fetched {len(doc_texts)} source document texts")

    # Step 3: assemble test examples
    examples = []
    for info in cases_info:
        best_doc = None
        best_len = 0
        for did in info["doc_ids"]:
            text = doc_texts.get(did, "")
            if len(text) > best_len:
                best_len = len(text)
                best_doc = text
        if best_doc and best_len > 200:
            examples.append({
                "test_id": f"mlx_{info['case_id']}",
                "source": best_doc,
                "reference": info["summary"],
                "dataset": "multilexsum",
            })

    print(f"  {len(examples)} MultiLexSum test examples ready")
    return examples


def load_billsum_test():
    """Load BillSum test split via HuggingFace."""
    print("Loading BillSum test split...")
    from datasets import load_dataset
    bs = load_dataset("billsum", split="test")
    examples = []
    for i, ex in enumerate(bs):
        examples.append({
            "test_id": f"bs_{i}",
            "source": ex["text"],
            "reference": ex["summary"],
            "dataset": "billsum",
        })
    print(f"  {len(examples)} BillSum test examples loaded")
    return examples


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of test examples (default 500)")
    parser.add_argument("--checkpoint", default="checkpoints/stage3_best.pt")
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--min_gen_len", type=int, default=30)
    parser.add_argument("--skip_mlx", action="store_true",
                        help="Skip MultiLexSum (avoids loading 4GB of JSON)")
    parser.add_argument("--mlx_max", type=int, default=100,
                        help="Max MultiLexSum cases to load (default 100)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load test data ----
    if args.skip_mlx:
        print("Skipping MultiLexSum (--skip_mlx)")
        mlx = []
    else:
        mlx = load_multilexsum_test(n_max=args.mlx_max)
    bs = load_billsum_test()

    # Mix and subsample
    if mlx:
        # Aim for ~40/60 mlx/bs split (mlx is capped smaller)
        n_mlx = min(len(mlx), int(args.n * 0.4))
        n_bs = min(len(bs), args.n - n_mlx)
        test_examples = mlx[:n_mlx] + bs[:n_bs]
        print(f"\nTest set size: {len(test_examples)} ({n_mlx} mlx + {n_bs} bs)")
    else:
        n_bs = min(len(bs), args.n)
        test_examples = bs[:n_bs]
        print(f"\nTest set size: {len(test_examples)} ({n_bs} bs only)")

    # ---- Load tokenizer + model ----
    tokenizer = load_tokenizer("tokenizer/tokenizer.json")
    model = build_model(
        device=device,
        vocab_size=32000,
        d_model=384,
        n_heads=6,
        n_encoder_layers=6,
        n_decoder_layers=4,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.0,  # disable for inference
        pad_idx=PAD_ID,
    )
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  val_loss: {ckpt.get('val_loss', '?'):.4f}")
    print(f"  epoch:    {ckpt.get('epoch', '?')}")

    # ---- Build hierarchical summarizer ----
    summarizer = HierarchicalSummarizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beam_width=args.beam_width,
        max_gen_len=args.max_gen_len,
        min_gen_len=args.min_gen_len,
    )
    print(f"\nBeam width: {args.beam_width}, max_gen: {args.max_gen_len}")

    # ---- Generate ----
    print("\nGenerating test outputs...")
    generated = {}
    references = {}
    sources = {}
    meta = {}

    t0 = time.time()
    for i, ex in enumerate(tqdm(test_examples, desc="Generating")):
        try:
            summary = summarizer.summarize(ex["source"])
        except Exception as e:
            print(f"  Error on {ex['test_id']}: {e}")
            summary = ""

        generated[ex["test_id"]] = summary
        references[ex["test_id"]] = ex["reference"]
        sources[ex["test_id"]] = ex["source"]
        meta[ex["test_id"]] = {"dataset": ex["dataset"]}

        # Periodic checkpoint save
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(test_examples)}] "
                  f"elapsed: {elapsed/60:.1f}min, rate: {rate:.2f} ex/s")
            with open(os.path.join(args.output_dir, "generated_summaries.json"), "w") as f:
                json.dump(generated, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed/60:.1f} minutes")
    print(f"  Avg time per example: {elapsed/len(test_examples):.2f} sec")

    # ---- Save all ----
    with open(os.path.join(args.output_dir, "generated_summaries.json"), "w") as f:
        json.dump(generated, f, indent=2)
    with open(os.path.join(args.output_dir, "reference_summaries.json"), "w") as f:
        json.dump(references, f, indent=2)
    with open(os.path.join(args.output_dir, "source_documents.json"), "w") as f:
        json.dump(sources, f, indent=2)
    with open(os.path.join(args.output_dir, "test_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Next: python -m evaluation.run_evaluation")


if __name__ == "__main__":
    main()
