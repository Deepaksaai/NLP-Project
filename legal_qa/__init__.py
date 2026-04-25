"""
legal_qa — Legal Document Question Answering Pipeline.

Uses DeBERTa-v3-large as the backbone with a retrieval-augmented
span-extraction approach, trained across three progressive stages:
    Stage 1: General QA (SQuAD 2.0, TriviaQA, Natural Questions)
    Stage 2: Long-document QA (QuALITY, QASPER)
    Stage 3: Legal domain QA (CUAD, LEDGAR, COLIEE)
"""
