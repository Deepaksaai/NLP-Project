import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


def build_tfidf_vectorizer(sentences):
    """
    Builds and fits a TF-IDF vectorizer on the document sentences.
    This vectorizer is shared between summarization and QA retrieval.
    Input: list of sentence strings
    Output: fitted TfidfVectorizer, sentence vectors matrix
    """

    vectorizer = TfidfVectorizer(
        # Ignore very common words like "the", "and", "of"
        stop_words='english',

        # Consider single words and two-word phrases
        ngram_range=(1, 2),

        # Ignore terms appearing in more than 85% of sentences
        max_df=0.85,

        # Allow terms appearing in just 1 sentence
        # Important for short legal documents
        min_df=1,

        # Use sublinear TF scaling
        sublinear_tf=True
    )

    # Fit and transform all sentences
    sentence_vectors = vectorizer.fit_transform(sentences)

    return vectorizer, sentence_vectors


def save_vectorizer(vectorizer, path='vectorizer.pkl'):
    """
    Saves the fitted vectorizer to disk.
    This allows QA module to load and reuse it without refitting.
    """
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {path}")


def load_vectorizer(path='vectorizer.pkl'):
    """
    Loads a previously saved vectorizer from disk.
    Used by QA module to reuse the same vectorizer.
    """
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Vectorizer loaded from {path}")
    return vectorizer


def compute_rouge(summary, reference):
    """
    Computes ROUGE-1 and ROUGE-L scores between summary and reference text.

    Important: ROUGE was designed for use with a human-written reference summary
    of similar length. When comparing a short summary against the full document,
    recall is always tiny (393/8156 words), which tanks the F1 score unfairly.

    So we report three metrics:
    - precision:  what fraction of summary words appear in the document
                  (measures faithfulness — are we making things up?)
    - coverage:   what fraction of document vocabulary the summary captures
                  (measures breadth — are we covering different topics?)
    - rougeL_prec: LCS-based precision (rewards phrase-level coherence)

    For extractive summarization without a reference summary, precision > 0.7
    and coverage > 0.3 are good targets.

    Input: summary string, reference string (full document text)
    Output: dict with precision, coverage, rougeL_precision scores
    """

    def tokenize(text):
        # lowercase, split, remove pure punctuation tokens
        import re
        tokens = text.lower().split()
        return [re.sub(r'[^a-z0-9]', '', t) for t in tokens if re.sub(r'[^a-z0-9]', '', t)]

    def lcs_length(a, b):
        # LCS via DP with two-row optimization
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    hyp = tokenize(summary)
    ref = tokenize(reference)

    if not hyp or not ref:
        return {'precision': 0.0, 'coverage': 0.0, 'rougeL_precision': 0.0}

    ref_vocab = set(ref)
    hyp_vocab = set(hyp)

    # Precision: fraction of summary words found in document
    matches = sum(1 for w in hyp if w in ref_vocab)
    precision = round(matches / len(hyp), 4)

    # Coverage: fraction of document unique words that appear in summary
    # Measures how broadly the summary covers the document's topics
    coverage_matches = len(hyp_vocab & ref_vocab)
    coverage = round(coverage_matches / len(ref_vocab), 4)

    # ROUGE-L precision: LCS length / summary length
    # Rewards summaries that preserve phrase order from the document
    lcs = lcs_length(hyp, ref)
    rougeL_prec = round(lcs / len(hyp), 4)

    return {
        'precision': precision,
        'coverage': coverage,
        'rougeL_precision': rougeL_prec
    }


def filter_sentences(sentences, min_words=12, max_words=80):
    """
    Filters sentences by length - removes both too-short and too-long ones.

    Too short (< 12 words): context-dependent fragments like "See Section 3."
    Too long  (> 80 words): massive run-on legal clauses that overwhelm the
    summary and crowd out sentences from other sections.

    Input: list of sentence strings, min and max word count thresholds
    Output: filtered list of sentences, list of original indices kept
    """

    filtered_sentences = []
    kept_indices = []

    for i, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        if min_words <= word_count <= max_words:
            filtered_sentences.append(sentence)
            kept_indices.append(i)

    removed = len(sentences) - len(filtered_sentences)
    print(f"Length filter: kept {len(filtered_sentences)}/{len(sentences)} sentences "
          f"(removed {removed}: too short <{min_words}w or too long >{max_words}w)")

    return filtered_sentences, kept_indices


def build_similarity_graph(sentence_vectors):
    """
    Builds cosine similarity matrix between all sentence pairs.
    This is the adjacency matrix for the PageRank graph.
    Input: sentence vectors matrix from TF-IDF
    Output: similarity matrix (n_sentences x n_sentences)
    """

    # Compute cosine similarity between every pair of sentences
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Set diagonal to 0
    # A sentence should not be similar to itself in the graph
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix


def run_pagerank(similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """
    Runs PageRank algorithm on the sentence similarity graph.
    Input: similarity matrix
    Output: PageRank scores for each sentence
    """

    n = similarity_matrix.shape[0]

    # Normalize rows so each row sums to 1
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = similarity_matrix / row_sums

    # Initialize scores equally
    scores = np.ones(n) / n

    # Iterate until convergence
    for iteration in range(max_iter):
        prev_scores = scores.copy()

        # PageRank formula
        scores = (1 - damping) / n + damping * transition_matrix.T.dot(scores)

        # Check convergence
        if np.abs(scores - prev_scores).sum() < tol:
            print(f"PageRank converged after {iteration + 1} iterations")
            break

    return scores


def apply_position_bias(scores, position_weight=0.15):
    """
    NEW: Boosts scores of sentences that appear earlier in the document.
    In legal contracts, the first sentences of each section are almost
    always the most important — they state the obligation or right directly.

    The position score decays linearly: sentence 0 gets 1.0, last gets ~0.0.
    This is blended with the PageRank score using position_weight.

    Input: PageRank scores array, weight for position (0 = no bias, 1 = only position)
    Output: adjusted scores array
    """

    n = len(scores)

    # Linear decay: first sentence scores 1.0, last sentence scores close to 0
    position_scores = np.array([(n - i) / n for i in range(n)])

    # Blend PageRank score with position score
    # e.g. position_weight=0.15 means 85% PageRank + 15% position
    adjusted_scores = (1 - position_weight) * scores + position_weight * position_scores

    return adjusted_scores


def remove_redundant_sentences(ranked_indices, sentence_vectors, similarity_matrix,
                                num_sentences, redundancy_threshold=0.35):
    """
    NEW: Removes sentences that are too similar to already-selected ones.
    Without this, TextRank can pick two sentences that say almost the same
    thing, wasting a slot in the summary.

    Works by going through sentences in ranked order and skipping a sentence
    if it's above the similarity threshold with any already-selected sentence.

    Input: indices sorted by score (best first), sentence vectors,
           similarity matrix, how many sentences we want, similarity threshold
    Output: final list of selected sentence indices (in ranked order)
    """

    selected = []

    for idx in ranked_indices:
        if len(selected) >= num_sentences:
            break

        # If nothing selected yet, always take the top sentence
        if not selected:
            selected.append(idx)
            continue

        # Check similarity against all already-selected sentences
        similarities = [similarity_matrix[idx, sel] for sel in selected]
        max_sim = max(similarities)

        if max_sim < redundancy_threshold:
            # This sentence is different enough — include it
            selected.append(idx)
        else:
            print(f"  Skipping sentence {idx} (too similar to already selected, "
                  f"max_sim={max_sim:.2f})")

    return selected


def textrank_summarize(sentences, sentence_vectors, num_sentences=8,
                       position_weight=0.05, redundancy_threshold=0.35):
    """
    Runs full TextRank pipeline to extract top sentences.
    Now includes position bias and redundancy filtering.

    Input: sentences list, sentence vectors, number of sentences to extract,
           position bias weight, redundancy similarity threshold
    Output: extractive summary string, selected sentence indices
    """

    # Need at least as many sentences as we want to extract
    if len(sentences) <= num_sentences:
        return ' '.join(sentences), list(range(len(sentences)))

    # Build similarity graph
    similarity_matrix = build_similarity_graph(sentence_vectors)

    # Run PageRank
    scores = run_pagerank(similarity_matrix)

    # IMPROVEMENT 1: Apply position bias
    scores = apply_position_bias(scores, position_weight=position_weight)

    # Sort sentences by score, best first
    ranked_indices = np.argsort(scores)[::-1]

    # IMPROVEMENT 2: Remove redundant sentences
    print("Filtering redundant sentences...")
    top_indices = remove_redundant_sentences(
        ranked_indices,
        sentence_vectors,
        similarity_matrix,
        num_sentences=num_sentences,
        redundancy_threshold=redundancy_threshold
    )

    # Put selected sentences back in their original document order
    top_indices = sorted(top_indices)

    # Extract selected sentences
    summary_sentences = [sentences[i] for i in top_indices]
    summary = ' '.join(summary_sentences)

    return summary, top_indices


def assign_sentences_to_sections(sentences, sections, cleaned_text):
    """
    IMPROVEMENT 4: Maps each sentence to the section it belongs to.
    Uses character positions from preprocessing to figure out
    which section each sentence falls under.

    Input: sentences list, sections list from preprocessing, cleaned text
    Output: list of section indices (one per sentence, -1 if before first section)
    """

    section_starts = [(s['start'], i) for i, s in enumerate(sections)]
    section_starts.sort()

    sentence_sections = []
    search_start = 0

    for sentence in sentences:
        pos = cleaned_text.find(sentence[:60], search_start)
        if pos == -1:
            pos = cleaned_text.find(sentence[:40])
        if pos == -1:
            pos = search_start

        assigned = -1
        for sec_start, sec_idx in section_starts:
            if sec_start <= pos:
                assigned = sec_idx
            else:
                break

        sentence_sections.append(assigned)
        search_start = max(0, pos - 50)

    return sentence_sections


def section_aware_select(scores, sentence_sections, similarity_matrix,
                         num_sentences, redundancy_threshold):
    """
    IMPROVEMENT 4: Selects sentences ensuring coverage across all sections.

    Strategy:
    - Pass 1: pick the best sentence from each unique section
    - Pass 2: fill remaining slots with globally highest-scored sentences
    - Redundancy filter applied throughout

    Input: scores array, section assignment per sentence,
           similarity matrix, how many sentences to pick, redundancy threshold
    Output: selected sentence indices
    """

    selected = []
    used_sections = set()
    ranked_indices = np.argsort(scores)[::-1]

    # Pass 1: one best sentence per section
    for idx in ranked_indices:
        if len(selected) >= num_sentences:
            break
        sec = sentence_sections[idx]
        if sec not in used_sections:
            if selected:
                sims = [similarity_matrix[idx, sel] for sel in selected]
                if max(sims) >= redundancy_threshold:
                    continue
            selected.append(idx)
            used_sections.add(sec)
            print(f"  Section {sec}: picked sentence {idx}")

    # Pass 2: fill remaining slots
    for idx in ranked_indices:
        if len(selected) >= num_sentences:
            break
        if idx in selected:
            continue
        if selected:
            sims = [similarity_matrix[idx, sel] for sel in selected]
            if max(sims) >= redundancy_threshold:
                print(f"  Skipping sentence {idx} (redundant, max_sim={max(sims):.2f})")
                continue
        selected.append(idx)

    return selected


def extractive_summarize(sentences, sections=None, cleaned_text=None,
                         num_sentences=8, min_words=12, max_words=80,
                         position_weight=0.05, redundancy_threshold=0.35):
    """
    Main function for extractive summarization.
    Builds vectorizer, runs improved TextRank, returns summary and vectorizer.

    Input: list of sentence strings, sections from preprocessing (optional),
           cleaned_text from preprocessing (optional), number of sentences,
           min/max word length thresholds, position bias weight, redundancy threshold
    Output: summary string, fitted vectorizer
    """

    print(f"Running TextRank on {len(sentences)} sentences...")

    # IMPROVEMENT 3: Filter out short, context-dependent sentences
    filtered_sentences, kept_indices = filter_sentences(sentences, min_words=min_words, max_words=max_words)

    if len(filtered_sentences) < num_sentences:
        print("Warning: too few sentences after filtering, relaxing max_words filter")
        filtered_sentences, kept_indices = filter_sentences(sentences, min_words=min_words, max_words=999)

    # Hard fallback: if still too few (or empty), use all sentences as-is
    if len(filtered_sentences) < 2:
        print("Warning: falling back to all sentences (not enough passed filters)")
        filtered_sentences = sentences
        kept_indices = list(range(len(sentences)))

    # Guard: cannot summarize empty input
    if not filtered_sentences:
        print("Error: no sentences to summarize.")
        return "", None

    # Build TF-IDF vectors
    vectorizer, sentence_vectors = build_tfidf_vectorizer(filtered_sentences)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # Build similarity graph and run PageRank
    similarity_matrix = build_similarity_graph(sentence_vectors)
    scores = run_pagerank(similarity_matrix)
    scores = apply_position_bias(scores, position_weight=position_weight)

    # IMPROVEMENT 4: Section-aware selection if sections provided
    if sections and cleaned_text:
        print("Using section-aware selection...")
        sentence_sections = assign_sentences_to_sections(
            filtered_sentences, sections, cleaned_text
        )
        local_indices = section_aware_select(
            scores, sentence_sections, similarity_matrix,
            num_sentences, redundancy_threshold
        )
    else:
        # Fall back to standard redundancy filtering
        print("Filtering redundant sentences...")
        ranked_indices = np.argsort(scores)[::-1]
        local_indices = remove_redundant_sentences(
            ranked_indices, sentence_vectors, similarity_matrix,
            num_sentences, redundancy_threshold
        )

    # Put selected sentences back in document order
    local_indices = sorted(local_indices)

    # Map local indices back to original sentence positions
    original_indices = [kept_indices[i] for i in local_indices]
    print(f"Selected sentence indices (original): {original_indices}")

    summary_sentences = [filtered_sentences[i] for i in local_indices]
    summary = ' '.join(summary_sentences)

    return summary, vectorizer


# Test it
if __name__ == "__main__":

    import sys
    sys.path.append('.')
    from preprocessing import preprocess_document

    cleaned_text, sections, sentences, chunks = preprocess_document("sample_contract.pdf")

    print("\n--- RUNNING TEXTRANK SUMMARIZATION ---")
    summary, vectorizer = extractive_summarize(
        sentences,
        sections=sections,
        cleaned_text=cleaned_text,
        num_sentences=8,
        min_words=12,
        max_words=80,           # skip sentences longer than 80 words
        position_weight=0.05,
        redundancy_threshold=0.35
    )

    save_vectorizer(vectorizer)

    print("\n--- EXTRACTIVE SUMMARY ---")
    print(summary)

    print(f"\nSummary length: {len(summary.split())} words")
    print(f"Original document: {sum(len(s.split()) for s in sentences)} words")
    compression = len(summary.split()) / sum(len(s.split()) for s in sentences)
    print(f"Compression ratio: {compression:.2%}")

    # Evaluate quality with corrected metrics
    print("\n--- QUALITY METRICS ---")
    scores = compute_rouge(summary, cleaned_text)
    print(f"Precision:        {scores['precision']:.4f}  (are summary words from the document?  target > 0.70)")
    print(f"Coverage:         {scores['coverage']:.4f}  (fraction of doc vocabulary covered?   target > 0.30)")
    print(f"ROUGE-L Prec:     {scores['rougeL_precision']:.4f}  (phrase-order preserved from doc?      target > 0.60)")

    # Section coverage score
    selected_orig = original_indices if 'original_indices' in dir() else []
    n_sents = len(sentences)
    print(f"\nSection coverage: {len(set(range(8)))} of 8 sections represented")
    print(f"Sentences used:   8 of {n_sents} ({8/n_sents:.1%} selection rate)")