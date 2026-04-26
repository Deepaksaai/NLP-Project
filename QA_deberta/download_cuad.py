"""
Download the ORIGINAL CUAD dataset from the official Atticus Project source,
and optionally filter out the no-answer (unanswerable) examples.

Why filter? In a pipeline with a retriever upstream, the QA model doesn't
need to learn "is there an answer?" — the retriever already filters for
relevance. Training only on positive examples makes the model better at
span extraction (its actual job) and avoids the conservative-to-a-fault
behavior you get when 68% of training is no-answer.

Saves to:
    QA_deberta/data/train.json
    QA_deberta/data/val.json

Usage (from NLP-Project/ root):
    python -m QA_deberta.download_cuad                    # positives only (recommended)
    python -m QA_deberta.download_cuad --keep_negatives   # full CUAD (SQuAD 2.0 style)
"""

import os
import sys
import json
import zipfile
import random
import argparse
import urllib.request

CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"


def download_with_progress(url: str, out_path: str):
    print(f"Downloading {url}")
    print("(~100-170 MB, may take a few minutes)")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_done  = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {percent:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, out_path, reporthook)
    print()


def extract_json_from_zip(zip_path: str, extract_dir: str) -> str:
    print(f"Extracting CUAD_v1.json from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_name = None
        for name in zf.namelist():
            if name.endswith("CUAD_v1.json"):
                json_name = name
                break
        if json_name is None:
            raise RuntimeError("Could not find CUAD_v1.json inside the zip")
        zf.extract(json_name, extract_dir)
        return os.path.join(extract_dir, json_name)


def convert_squad_format(squad_json_path: str, keep_negatives: bool):
    """Convert CUAD_v1.json (SQuAD 2.0 format) into our training format."""
    with open(squad_json_path, encoding="utf-8") as f:
        squad = json.load(f)

    examples = []
    skipped_neg = 0
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question      = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                has_answers   = bool(qa.get("answers"))

                if is_impossible or not has_answers:
                    if not keep_negatives:
                        skipped_neg += 1
                        continue
                    examples.append({
                        "question":     question,
                        "context":      context,
                        "answer_start": -1,
                        "answer_text":  "",
                        "has_answer":   False,
                    })
                else:
                    ans = qa["answers"][0]
                    examples.append({
                        "question":     question,
                        "context":      context,
                        "answer_start": ans["answer_start"],
                        "answer_text":  ans["text"],
                        "has_answer":   True,
                    })
    return examples, skipped_neg


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    zip_path = os.path.join(args.output_dir, "CUAD_v1.zip")
    if os.path.exists(zip_path) and not args.force_download:
        print(f"Found existing {zip_path}, skipping download.")
    else:
        download_with_progress(CUAD_ZIP_URL, zip_path)

    json_path = extract_json_from_zip(zip_path, args.output_dir)

    print(f"Converting {json_path} to training format...")
    print(f"  Mode: {'FULL CUAD (with negatives)' if args.keep_negatives else 'POSITIVES ONLY'}")
    all_examples, skipped = convert_squad_format(json_path, args.keep_negatives)
    if skipped:
        print(f"  Filtered out {skipped} no-answer examples.")

    # Split 90/10 by contract
    by_context = {}
    for ex in all_examples:
        key = ex["context"][:200]
        by_context.setdefault(key, []).append(ex)

    contracts = list(by_context.keys())
    random.seed(42)
    random.shuffle(contracts)
    cut = int(len(contracts) * 0.9)
    train_examples = [ex for c in contracts[:cut] for ex in by_context[c]]
    val_examples   = [ex for c in contracts[cut:] for ex in by_context[c]]

    train_path = os.path.join(args.output_dir, "train.json")
    val_path   = os.path.join(args.output_dir, "val.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False)

    train_has = sum(1 for e in train_examples if e["has_answer"])
    val_has   = sum(1 for e in val_examples if e["has_answer"])
    print()
    print("Done.")
    print(f"  train.json: {len(train_examples)} examples "
          f"({train_has} with answer, {len(train_examples)-train_has} no-answer)")
    print(f"  val.json:   {len(val_examples)} examples "
          f"({val_has} with answer, {len(val_examples)-val_has} no-answer)")
    print(f"Written to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",     type=str, default="QA_deberta/data")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--keep_negatives", action="store_true",
                        help="Include no-answer examples (full SQuAD 2.0 style CUAD)")
    args = parser.parse_args()
    main(args)
