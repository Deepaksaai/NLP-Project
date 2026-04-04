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


def textrank_summarize(sentences, sentence_vectors, num_sentences=8):
    """
    Runs full TextRank pipeline to extract top sentences.
    Input: sentences list, sentence vectors, number of sentences to extract
    Output: extractive summary string, ranked sentence indices
    """

    # Need at least as many sentences as we want to extract
    if len(sentences) <= num_sentences:
        return ' '.join(sentences), list(range(len(sentences)))

    # Build similarity graph
    similarity_matrix = build_similarity_graph(sentence_vectors)

    # Run PageRank
    scores = run_pagerank(similarity_matrix)

    # Get indices of top N sentences sorted by score
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = sorted(ranked_indices[:num_sentences])

    # Extract top sentences in original document order
    summary_sentences = [sentences[i] for i in top_indices]
    summary = ' '.join(summary_sentences)

    return summary, top_indices


def extractive_summarize(sentences, num_sentences=8):
    """
    Main function for extractive summarization.
    Builds vectorizer, runs TextRank, returns summary and vectorizer.
    Input: list of sentence strings, number of sentences to extract
    Output: summary string, fitted vectorizer
    """

    print(f"Running TextRank on {len(sentences)} sentences...")

    # Build TF-IDF vectors
    vectorizer, sentence_vectors = build_tfidf_vectorizer(sentences)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # Run TextRank
    summary, top_indices = textrank_summarize(
        sentences,
        sentence_vectors,
        num_sentences=num_sentences
    )

    print(f"Selected sentence indices: {top_indices}")

    return summary, vectorizer


# Test it
if __name__ == "__main__":

    import sys
    sys.path.append('.')
    from preprocessing import preprocess_document

    # Run preprocessing
    cleaned_text, sections, sentences, chunks = preprocess_document("sample_contract.pdf")

    # Run extractive summarization
    print("\n--- RUNNING TEXTRANK SUMMARIZATION ---")
    summary, vectorizer = extractive_summarize(sentences, num_sentences=8)

    # Save vectorizer for reuse in QA module
    save_vectorizer(vectorizer)

    print("\n--- EXTRACTIVE SUMMARY ---")
    print(summary)

    print(f"\nSummary length: {len(summary.split())} words")
    print(f"Original document: {sum(len(s.split()) for s in sentences)} words")
    compression = len(summary.split()) / sum(len(s.split()) for s in sentences)
    print(f"Compression ratio: {compression:.2%}")