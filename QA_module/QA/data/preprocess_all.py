"""
Orchestrator: preprocesses every QA dataset into data/qa/stage1/.

Running this end-to-end requires network access to HuggingFace for
SQuAD 2.0 / TriviaQA / (optionally) Natural Questions. Use
--limit on first runs to validate the pipeline without pulling the
entire corpora.

Usage:
    python -m QA.data.preprocess_all --limit 5000
    python -m QA.data.preprocess_all            # full datasets
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Cap examples per split (useful for smoke tests).")
    p.add_argument("--skip_trivia", action="store_true")
    p.add_argument("--skip_nq",     action="store_true", default=True,
                   help="NQ preprocessor is a stub; disabled by default.")
    args = p.parse_args()

    from QA.data import preprocess_squad
    print("=" * 50)
    print("SQuAD 2.0")
    print("=" * 50)
    preprocess_squad.preprocess_split("train",      limit=args.limit)
    preprocess_squad.preprocess_split("validation", limit=(args.limit // 10 if args.limit else None))

    if not args.skip_trivia:
        print("=" * 50)
        print("TriviaQA")
        print("=" * 50)
        from QA.data import preprocess_trivia
        preprocess_trivia.preprocess_split("train",      limit=args.limit)
        preprocess_trivia.preprocess_split("validation", limit=(args.limit // 10 if args.limit else None))

    if not args.skip_nq:
        print("=" * 50)
        print("Natural Questions (stub)")
        print("=" * 50)
        from QA.data import preprocess_nq
        preprocess_nq.preprocess_split("train", limit=args.limit)


if __name__ == "__main__":
    main()
