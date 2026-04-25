import os
import re
import json
import hashlib
import pickle

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Known section keywords that must NOT be removed during cleaning
# ---------------------------------------------------------------------------
KNOWN_SECTION_KEYWORDS = {
    "DEFINITIONS", "RECITALS", "WHEREAS", "WITNESSETH",
    "GOVERNING LAW", "INDEMNIFICATION", "CONFIDENTIALITY",
    "MISCELLANEOUS", "TERMINATION", "WARRANTIES", "REPRESENTATIONS",
    "TERM AND TERMINATION", "LIMITATION OF LIABILITY",
    "DISPUTE RESOLUTION", "REPRESENTATIONS AND WARRANTIES",
    "INTELLECTUAL PROPERTY", "FORCE MAJEURE", "ASSIGNMENT",
    "NOTICES", "AMENDMENTS", "ENTIRE AGREEMENT", "SEVERABILITY",
    "WAIVER", "COUNTERPARTS", "HEADINGS", "SURVIVAL"
}

# ---------------------------------------------------------------------------
# Legal synonym map for query expansion (used by QA retrieval)
# ---------------------------------------------------------------------------
LEGAL_SYNONYMS = {
    "cancel"     : ["terminate", "termination", "rescind", "void", "dissolution"],
    "cancelled"  : ["terminated", "rescinded", "voided", "dissolved"],
    "fine"       : ["penalty", "damages", "liquidated damages", "sanction"],
    "sign"       : ["execute", "enter into", "execution"],
    "signed"     : ["executed", "entered into"],
    "start"      : ["effective date", "commencement", "commence"],
    "end"        : ["termination", "expiration", "expire", "dissolution"],
    "buy"        : ["purchase", "acquire", "acquisition"],
    "sell"       : ["transfer", "assign", "convey"],
    "agreement"  : ["contract", "arrangement", "understanding"],
    "party"      : ["counterparty", "signatory"],
    "payment"    : ["consideration", "remuneration", "compensation", "fee"],
    "secret"     : ["confidential", "proprietary", "non-disclosure"],
    "break"      : ["breach", "default", "violation"],
    "responsible": ["liable", "liability", "obligated", "obligation"],
    "allow"      : ["permit", "authorize", "consent"],
    "forbid"     : ["prohibit", "restrict", "shall not"],
    "own"        : ["intellectual property", "proprietary rights", "ownership"],
    "rules"      : ["governing law", "jurisdiction", "applicable law"],
    "arbitration": ["dispute resolution", "mediation", "arbitration"],
}


# ===========================================================================
# TASK 1 — PDF TEXT EXTRACTION (printed documents only)
# ===========================================================================

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a printed (digital) PDF file.
    Preserves page boundary markers so chunks can report page numbers.

    Input  : path to PDF file
    Output : raw text string with [PAGE_N] markers embedded
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text is None or text.strip() == "":
                print(f"  Warning: Page {page_num + 1} returned no text — skipping")
                continue

            # Embed page marker so we can track page numbers through chunking
            full_text.append(f"[PAGE_{page_num + 1}]\n{text}")

    raw_text = "\n".join(full_text)

    # Fix hyphenated line breaks  e.g. "indem-\nnification"
    raw_text = re.sub(r'-\n', '', raw_text)

    # Fix irregular mid-sentence line breaks
    raw_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text)

    # Normalize whitespace
    raw_text = re.sub(r' +', ' ', raw_text)

    # Fix unicode artifacts
    raw_text = raw_text.replace('\xa0', ' ')
    raw_text = raw_text.replace('\x0c', '\n')

    return raw_text.strip()


# ===========================================================================
# TASK 2 — TEXT CLEANING
# ===========================================================================

def clean_text(raw_text):
    """
    Cleans raw extracted text from a legal PDF.
    Preserves: punctuation, capitalization, stopwords, section headers.
    Removes : headers, footers, page numbers, stamps, redaction markers.
    Keeps   : [PAGE_N] markers intact for downstream page tracking.

    Input  : raw text string from extract_text_from_pdf()
    Output : cleaned text string (still contains [PAGE_N] markers)
    """

    # -------------------------------------------------------------------
    # BLOCK 1: Truncate after signature page
    # -------------------------------------------------------------------
    signature_patterns = [
        r'IN WITNESS WHEREOF.*',
        r'SIGNATURE PAGE FOLLOWS.*',
        r'\[Signature Page Follows\].*',
        r'EXECUTED as of the date.*',
    ]
    for pattern in signature_patterns:
        match = re.search(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            raw_text = raw_text[:match.start()]
            break

    # -------------------------------------------------------------------
    # BLOCK 2: Remove redaction markers
    # -------------------------------------------------------------------
    raw_text = re.sub(
        r'CERTAIN INFORMATION.*?REDACTED\.',
        '', raw_text, flags=re.DOTALL | re.IGNORECASE
    )
    raw_text = re.sub(r'\[\*+\]', '', raw_text)
    raw_text = re.sub(r'\[REDACTED\]', '', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\[Redacted\]', '', raw_text)
    raw_text = re.sub(
        r'OR\s+INDICATES\s+THAT\s+INFORMATION\s+HAS\s+BEEN\s+REDACTED\.?',
        '', raw_text, flags=re.IGNORECASE
    )

    # -------------------------------------------------------------------
    # BLOCK 3: Remove document stamps
    # -------------------------------------------------------------------
    raw_text = re.sub(r'\bEXECUTION\s+COPY\b', '', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\bCONFIDENTIAL\b', '', raw_text)
    raw_text = re.sub(r'\bDRAFT\b', '', raw_text)
    raw_text = re.sub(r'\bPROPRIETARY\b', '', raw_text)

    # -------------------------------------------------------------------
    # BLOCK 4: Remove headers and footers
    # -------------------------------------------------------------------
    raw_text = re.sub(r'Source:\s+.*?\d{4}', '', raw_text, flags=re.IGNORECASE)

    # Replace exhibit references with placeholder — do not delete silently
    # "See Exhibit A for full terms" is meaningful context
    raw_text = re.sub(
        r'\bExhibit\s+[\w.-]+\b',
        '[EXHIBIT REFERENCE]',
        raw_text,
        flags=re.IGNORECASE
    )
    raw_text = re.sub(r'\bExh\.\s+[A-Z]-\d+\b', '[EXHIBIT REFERENCE]', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\bEXH\.\s+[A-Z]-\d+\b', '[EXHIBIT REFERENCE]', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'EXHIBIT\s+[A-Z]\s+[A-Z\s&/]+', '', raw_text)

    # -------------------------------------------------------------------
    # BLOCK 5: Line by line cleaning
    # -------------------------------------------------------------------
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Always preserve page markers
        if re.match(r'^\[PAGE_\d+\]$', line):
            cleaned_lines.append(line)
            continue

        # Skip standalone page numbers
        if re.match(r'^-?\s*\d+\s*-?$', line):
            continue
        if re.match(r'^[Pp]age\s+\d+(\s+of\s+\d+)?$', line):
            continue

        # Skip schedule and annex headers
        if re.match(r'^(Schedule|Annex|Appendix)\s+[\w.]+\s*$', line, re.IGNORECASE):
            continue

        # Skip short all-caps lines UNLESS they are known section headers
        # FIX: original code was removing valid section headers like "DEFINITIONS"
        if line.isupper() and len(line.split()) <= 5:
            if line.strip() not in KNOWN_SECTION_KEYWORDS:
                continue

        # Skip lines with only special characters
        if re.match(r'^[\W_]+$', line):
            continue

        # Skip very short lines (but not page markers — already handled above)
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    # -------------------------------------------------------------------
    # BLOCK 6: Final normalization
    # -------------------------------------------------------------------
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)

    return cleaned_text.strip()


# ===========================================================================
# TASK 3 — SECTION DETECTION
# ===========================================================================

def detect_sections(cleaned_text):
    """
    Detects legal section headers in cleaned text.
    Returns list of dicts with section name and character position.
    Page markers are stripped before matching so positions are accurate.

    Input  : cleaned text from clean_text() (may contain [PAGE_N] markers)
    Output : list of dicts {name, start, end}
    """

    # Strip page markers for section detection only
    # We match on clean text but positions map back to original
    text_for_detection = re.sub(r'\[PAGE_\d+\]\n?', '', cleaned_text)

    section_patterns = [
        r'(?<!\w)ARTICLE\s+(?:[IVX]+|\d+)\.?\s*\n?\s*([A-Z][A-Z\s&;,]+)',
        r'(?<!\w)SECTION\s+\d+\.?\s*\n?\s*([A-Z][A-Z\s&;,]+)',
        r'^\d+\.(?:\d+)?\s+([A-Z][A-Z\s&;,]{3,})',
        r'(?<!\w)(WHEREAS|WITNESSETH|NOW[,\s]+THEREFORE)[,:]?',
        r'(?<!\w)(IN\s+WITNESS\s+WHEREOF)',
        r'^(RECITALS|DEFINITIONS|REPRESENTATIONS AND WARRANTIES|'
        r'INDEMNIFICATION|MISCELLANEOUS|GOVERNING LAW|'
        r'CONFIDENTIALITY|TERM AND TERMINATION|'
        r'LIMITATION OF LIABILITY|DISPUTE RESOLUTION)$',
    ]

    sections = []
    for pattern in section_patterns:
        for match in re.finditer(pattern, text_for_detection, flags=re.MULTILINE):
            section_name = match.group(0).strip()
            section_name = re.sub(r'\s+', ' ', section_name)
            sections.append({
                'name' : section_name,
                'start': match.start(),
                'end'  : match.end()
            })

    # Sort by position
    sections = sorted(sections, key=lambda x: x['start'])

    # Remove overlapping duplicates
    unique_sections = []
    last_pos = -1
    for section in sections:
        if section['start'] > last_pos + 10:
            unique_sections.append(section)
            last_pos = section['start']

    return unique_sections


# ===========================================================================
# TASK 4 — SENTENCE SEGMENTATION
# ===========================================================================

def segment_sentences(cleaned_text):
    """
    Splits cleaned legal text into individual sentences using regex.
    Handles common legal abbreviations to avoid false splits.

    Input  : cleaned text from clean_text()
    Output : list of sentence strings
    """
    text = re.sub(r'\[PAGE_\d+\]\n?', '', cleaned_text)

    # Protect common abbreviations by replacing their periods temporarily
    abbrev_pattern = re.compile(
        r'\b(Inc|Corp|Ltd|LLC|LLP|Co|No|Sec|Art|vs|etc|e\.g|i\.e|'
        r'U\.S|U\.K|Fig|Dept|Est|approx|Jan|Feb|Mar|Apr|Jun|Jul|'
        r'Aug|Sep|Oct|Nov|Dec|Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.',
        re.IGNORECASE
    )
    text = abbrev_pattern.sub(lambda m: m.group(0).replace('.', '<!DOT!>'), text)

    # Split on sentence-ending punctuation followed by whitespace + capital letter
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z("])', text)

    sentences = []
    for sent in raw_sentences:
        sent = sent.replace('<!DOT!>', '.').strip()
        if not sent or len(sent) < 10:
            continue
        if re.match(r'^[\d\s\W]+$', sent):
            continue
        sentences.append(sent)

    return sentences


# ===========================================================================
# HELPER — PAGE NUMBER RESOLVER
# ===========================================================================

def build_page_map(cleaned_text):
    """
    Builds a mapping from character offset → page number.
    Uses the [PAGE_N] markers embedded during extraction.

    Input  : cleaned text with [PAGE_N] markers
    Output : sorted list of (char_offset, page_number) tuples
    """
    page_map = []
    for match in re.finditer(r'\[PAGE_(\d+)\]', cleaned_text):
        page_num = int(match.group(1))
        page_map.append((match.start(), page_num))
    return sorted(page_map, key=lambda x: x[0])


def get_page_range(char_start, char_end, page_map):
    """
    Returns the page range (start_page, end_page) for a given
    character range in the cleaned text.
    """
    if not page_map:
        return (1, 1)

    start_page = 1
    end_page   = 1

    for offset, page_num in page_map:
        if offset <= char_start:
            start_page = page_num
        if offset <= char_end:
            end_page = page_num

    return (start_page, end_page)


# ===========================================================================
# TASK 5 — CHUNKING
# ===========================================================================

def chunk_document(cleaned_text, sentences, sections, doc_id,
                   chunk_size=350, overlap=50):
    """
    Splits document into overlapping chunks for model input.
    Each chunk is capped at chunk_size words (≈ 400 BPE tokens).
    Attaches full metadata including page numbers and section heading.
    Prepends the current section header to each chunk text so the model
    has context about what part of the document it is reading.

    Input  : cleaned text, sentences, sections, doc_id
    Output : list of chunk dicts with full metadata
    """

    def count_words(text):
        return len(text.split())

    # -------------------------------------------------------------------
    # Build page map from embedded markers
    # -------------------------------------------------------------------
    page_map = build_page_map(cleaned_text)

    # -------------------------------------------------------------------
    # Build a char-offset → section name lookup
    # -------------------------------------------------------------------
    def get_current_section(char_offset):
        current = "PREAMBLE"
        for sec in sections:
            if sec['start'] <= char_offset:
                current = sec['name']
            else:
                break
        return current

    # -------------------------------------------------------------------
    # Map each sentence to its approximate char offset in cleaned_text
    # We do a sequential search — assumes sentences appear in order
    # -------------------------------------------------------------------
    sentence_offsets = []
    search_start = 0
    clean_no_markers = re.sub(r'\[PAGE_\d+\]\n?', '', cleaned_text)

    for sentence in sentences:
        idx = clean_no_markers.find(sentence, search_start)
        if idx == -1:
            idx = search_start  # fallback
        sentence_offsets.append(idx)
        search_start = idx + len(sentence)

    # -------------------------------------------------------------------
    # Identify sentence indices that fall on section boundaries
    # -------------------------------------------------------------------
    section_starts = set()
    for section in sections:
        for i, offset in enumerate(sentence_offsets):
            if offset >= section['start']:
                section_starts.add(i)
                break

    # -------------------------------------------------------------------
    # Build chunks greedily
    # -------------------------------------------------------------------
    chunks       = []
    chunk_index  = 0
    current_sents        = []
    current_word_count   = 0
    i = 0

    while i < len(sentences):
        sentence      = sentences[i]
        sentence_words = count_words(sentence)

        if current_word_count + sentence_words > chunk_size and current_sents:

            # Look ahead up to 3 sentences for a cleaner section boundary
            broke_at_boundary = False
            for lookahead in range(1, 4):
                if (i + lookahead < len(sentences)
                        and (i + lookahead) in section_starts):
                    for j in range(lookahead):
                        current_sents.append(sentences[i + j])
                        current_word_count += count_words(sentences[i + j])
                    i += lookahead
                    broke_at_boundary = True
                    break

            # Compute character range of this chunk
            first_sent_offset = sentence_offsets[i - len(current_sents)]
            last_sent_offset  = sentence_offsets[i - 1] + len(current_sents[-1])
            start_page, end_page = get_page_range(
                first_sent_offset, last_sent_offset, page_map
            )

            # Determine section for this chunk
            section_name = get_current_section(first_sent_offset)

            # Build chunk text with section header prepended
            chunk_body = ' '.join(current_sents)
            chunk_text = f"{section_name}: {chunk_body}"

            chunks.append({
                'chunk_id'      : f"{doc_id}_chunk_{chunk_index}",
                'doc_id'        : doc_id,
                'chunk_index'   : chunk_index,
                'text'          : chunk_text,        # section header + body
                'body'          : chunk_body,        # body only (for display)
                'section'       : section_name,
                'page_start'    : start_page,
                'page_end'      : end_page,
                'word_count'    : current_word_count,
                'sentence_count': len(current_sents),
                'char_start'    : first_sent_offset,
                'char_end'      : last_sent_offset,
                'bpe_token_ids' : [],                # filled later by tokenizer
            })
            chunk_index += 1

            # Start new chunk with overlap sentences
            overlap_sents      = []
            overlap_word_count = 0
            for sent in reversed(current_sents):
                sent_words = count_words(sent)
                if overlap_word_count + sent_words <= overlap:
                    overlap_sents.insert(0, sent)
                    overlap_word_count += sent_words
                else:
                    break

            current_sents      = overlap_sents
            current_word_count = overlap_word_count

            if not broke_at_boundary:
                current_sents.append(sentence)
                current_word_count += sentence_words
                i += 1

        else:
            current_sents.append(sentence)
            current_word_count += sentence_words
            i += 1

    # Save the final remaining chunk
    if current_sents:
        first_sent_offset = sentence_offsets[
            max(0, len(sentences) - len(current_sents))
        ]
        last_sent_offset  = sentence_offsets[-1] + len(sentences[-1])
        start_page, end_page = get_page_range(
            first_sent_offset, last_sent_offset, page_map
        )
        section_name = get_current_section(first_sent_offset)
        chunk_body   = ' '.join(current_sents)
        chunk_text   = f"{section_name}: {chunk_body}"

        chunks.append({
            'chunk_id'      : f"{doc_id}_chunk_{chunk_index}",
            'doc_id'        : doc_id,
            'chunk_index'   : chunk_index,
            'text'          : chunk_text,
            'body'          : chunk_body,
            'section'       : section_name,
            'page_start'    : start_page,
            'page_end'      : end_page,
            'word_count'    : current_word_count,
            'sentence_count': len(current_sents),
            'char_start'    : first_sent_offset,
            'char_end'      : last_sent_offset,
            'bpe_token_ids' : [],
        })

    return chunks


# ===========================================================================
# TASK 6 — DOCUMENT METADATA EXTRACTION
# ===========================================================================

def detect_document_type(text):
    """
    Classifies the document into one of four legal document types
    using keyword frequency in the first 3000 characters.
    """
    keywords = {
        "contract"    : ["agreement", "parties", "whereas", "consideration",
                         "shall", "obligations", "covenants"],
        "court_order" : ["plaintiff", "defendant", "court", "ordered",
                         "judgment", "hereby", "opinion"],
        "statute"     : ["enacted", "legislature", "section", "subsection",
                         "act", "statute", "regulation"],
        "motion"      : ["moves", "motion", "respectfully", "counsel",
                         "memorandum", "relief", "petitioner"],
    }
    text_lower = text[:3000].lower()
    scores = {
        doc_type: sum(1 for kw in kws if kw in text_lower)
        for doc_type, kws in keywords.items()
    }
    return max(scores, key=scores.get)


def extract_document_metadata(cleaned_text, pdf_path, doc_id):
    """
    Extracts document-level metadata using regex patterns.
    Runs on the first 5000 characters only for speed.

    Input  : cleaned text, pdf_path, doc_id
    Output : metadata dict
    """
    text_for_ner = re.sub(r'\[PAGE_\d+\]\n?', '', cleaned_text)
    preview      = text_for_ner[:5000]

    # Extract organisation / party names — look for capitalised name + entity suffix
    org_pattern = re.compile(
        r'\b([A-Z][A-Za-z0-9&,.\'\- ]{2,50}?'
        r'(?:Inc\.?|Corp\.?|Ltd\.?|LLC|LLP|L\.L\.C\.?|L\.P\.?|Company|Corporation|'
        r'Limited|Partners|Group|Holdings|Trust|Authority|Association))\b'
    )
    parties = list(dict.fromkeys(m.group(1).strip() for m in org_pattern.finditer(preview)))[:10]

    # Extract dates
    date_pattern = re.compile(
        r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|'
        r'July|August|September|October|November|December)\s+\d{4}|'
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4}|'
        r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        re.IGNORECASE
    )
    dates = list(dict.fromkeys(m.group(0) for m in date_pattern.finditer(preview)))[:5]

    # Governing law / jurisdiction
    gov_law_match = re.search(
        r'governed by.*?law of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        text_for_ner, re.IGNORECASE
    )
    jurisdiction = gov_law_match.group(1) if gov_law_match else "Unknown"

    doc_type = detect_document_type(text_for_ner)

    return {
        "doc_id"      : doc_id,
        "filename"    : os.path.basename(pdf_path),
        "doc_type"    : doc_type,
        "parties"     : parties,
        "dates"       : dates,
        "jurisdiction": jurisdiction,
        "total_chars" : len(text_for_ner),
    }


# ===========================================================================
# TASK 7 — TF-IDF INDEX CONSTRUCTION
# ===========================================================================

def build_tfidf_index(chunks):
    """
    Builds a TF-IDF index over all chunks for QA retrieval.
    Uses unigrams and bigrams.
    Does NOT remove stopwords — "shall not" ≠ "not" in legal text.

    Input  : list of chunk dicts from chunk_document()
    Output : dict with vectorizer, matrix, chunk_ids
    """
    texts = [chunk['text'] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        max_features = 10000,
        ngram_range  = (1, 2),
        stop_words   = None,        # critical — preserve legal stopwords
        min_df       = 1,
        sublinear_tf = True,        # log normalization on TF
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return {
        "vectorizer": vectorizer,
        "matrix"    : tfidf_matrix,
        "chunk_ids" : [chunk['chunk_id'] for chunk in chunks],
    }


# ===========================================================================
# TASK 8 — SAVE TO DOCUMENT STORE
# ===========================================================================

def save_document_store(doc_id, metadata, chunks, tfidf_index,
                        store_root="document_store",
                        sentences=None, sections=None, cleaned_text=None):
    """
    Saves all processed data for one document to disk.

    Structure:
      document_store/
        {doc_id}/
          metadata.json
          chunks.json
          tfidf_index.pkl
          sentences.json      (optional — for TextRank)
          sections.json       (optional — for section-aware TextRank)
          cleaned_text.txt    (optional — for section-aware TextRank)
    """
    doc_dir = os.path.join(store_root, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    # Save metadata
    with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save chunks (exclude bpe_token_ids from JSON — fill later)
    chunks_serializable = [
        {k: v for k, v in chunk.items() if k != 'bpe_token_ids'}
        for chunk in chunks
    ]
    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_serializable, f, indent=2, ensure_ascii=False)

    # Save TF-IDF index as pickle
    with open(os.path.join(doc_dir, "tfidf_index.pkl"), "wb") as f:
        pickle.dump(tfidf_index, f)

    if sentences is not None:
        with open(os.path.join(doc_dir, "sentences.json"), "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False)

    if sections is not None:
        with open(os.path.join(doc_dir, "sections.json"), "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False)

    if cleaned_text is not None:
        with open(os.path.join(doc_dir, "cleaned_text.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned_text)

    print(f"  Saved to: {doc_dir}/")


# ===========================================================================
# QUERY EXPANSION — used by QA retrieval at inference time
# ===========================================================================

def expand_query(question):
    """
    Expands a plain English question with legal synonyms.
    Bridges the vocabulary gap between user questions and legal document text.

    Input  : question string
    Output : expanded query string (original + synonyms)
    """
    question_lower = question.lower()
    expansions     = [question]

    for plain_word, legal_terms in LEGAL_SYNONYMS.items():
        if plain_word in question_lower:
            expansions.extend(legal_terms)

    return ' '.join(expansions)


# ===========================================================================
# RETRIEVAL — TF-IDF cosine similarity search
# ===========================================================================

def retrieve_chunks(question, tfidf_index, chunks, k=5):
    """
    Retrieves the top-k most relevant chunks for a given question.
    Applies section-weighted boost when the chunk's section heading
    matches keywords in the question.

    Input  : question, tfidf_index dict, chunks list, k
    Output : list of top-k chunk dicts sorted by relevance score
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Expand query with legal synonyms
    expanded_query = expand_query(question)

    # Vectorize the query
    query_vec = tfidf_index["vectorizer"].transform([expanded_query])

    # Compute cosine similarities
    similarities = cosine_similarity(query_vec, tfidf_index["matrix"]).flatten()

    # Apply section-weighted boost
    question_lower = question.lower()
    for i, chunk in enumerate(chunks):
        section_lower = chunk.get('section', '').lower()
        # Boost if any word from section heading appears in question
        section_words = set(section_lower.split())
        question_words = set(question_lower.split())
        if section_words & question_words:
            similarities[i] *= 1.5

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_k_indices:
        chunk = chunks[idx].copy()
        chunk['retrieval_score'] = float(similarities[idx])
        results.append(chunk)

    return results


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================

def generate_doc_id(pdf_path):
    """Generates a stable unique ID from the PDF file path."""
    return hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:12]


def preprocess_document(pdf_path, store_root="document_store"):
    """
    Master pipeline: runs all 8 tasks in sequence.

    Input  : path to PDF file
    Output : (cleaned_text, sections, sentences, chunks, metadata, tfidf_index)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(pdf_path)}")
    print(f"{'='*60}")

    doc_id = generate_doc_id(pdf_path)
    print(f"Document ID: {doc_id}")

    # Task 1 — Extract
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"\nTask 1 done — Extracted {len(raw_text):,} characters")

    # Task 2 — Clean
    cleaned_text = clean_text(raw_text)
    removed = len(raw_text) - len(cleaned_text)
    print(f"Task 2 done — Cleaned to {len(cleaned_text):,} characters "
          f"({removed:,} removed)")

    # Task 3 — Detect sections
    sections = detect_sections(cleaned_text)
    print(f"Task 3 done — Detected {len(sections)} sections")

    # Task 4 — Segment sentences
    sentences = segment_sentences(cleaned_text)
    print(f"Task 4 done — Segmented into {len(sentences)} sentences")

    # Task 5 — Chunk document
    chunks = chunk_document(cleaned_text, sentences, sections, doc_id)
    print(f"Task 5 done — Created {len(chunks)} chunks "
          f"(avg {sum(c['word_count'] for c in chunks)//max(len(chunks),1)} words/chunk)")

    # Task 6 — Extract document metadata
    metadata = extract_document_metadata(cleaned_text, pdf_path, doc_id)
    print(f"Task 6 done — Type: {metadata['doc_type']} | "
          f"Parties: {len(metadata['parties'])} | "
          f"Jurisdiction: {metadata['jurisdiction']}")

    # Task 7 — Build TF-IDF index
    tfidf_index = build_tfidf_index(chunks)
    print(f"Task 7 done — TF-IDF index built "
          f"({tfidf_index['matrix'].shape[0]} chunks × "
          f"{tfidf_index['matrix'].shape[1]} features)")

    # Task 8 — Save to document store
    save_document_store(doc_id, metadata, chunks, tfidf_index, store_root,
                        sentences=sentences, sections=sections, cleaned_text=cleaned_text)
    print(f"Task 8 done — Saved to document store")

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    print(f"{'='*60}\n")

    return cleaned_text, sections, sentences, chunks, metadata, tfidf_index


# ===========================================================================
# LOAD FROM DOCUMENT STORE (for inference time)
# ===========================================================================

def load_document_store(doc_id, store_root="document_store"):
    """
    Loads a previously processed document from disk.
    Use this at inference time instead of reprocessing.

    Input  : doc_id string
    Output : (metadata, chunks, tfidf_index, sentences, sections, cleaned_text)
             sentences, sections, cleaned_text are None if not cached.
    """
    doc_dir = os.path.join(store_root, doc_id)

    with open(os.path.join(doc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(os.path.join(doc_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(os.path.join(doc_dir, "tfidf_index.pkl"), "rb") as f:
        tfidf_index = pickle.load(f)

    sentences_path = os.path.join(doc_dir, "sentences.json")
    sentences = None
    if os.path.exists(sentences_path):
        with open(sentences_path, "r", encoding="utf-8") as f:
            sentences = json.load(f)

    sections_path = os.path.join(doc_dir, "sections.json")
    sections = None
    if os.path.exists(sections_path):
        with open(sections_path, "r", encoding="utf-8") as f:
            sections = json.load(f)

    cleaned_text_path = os.path.join(doc_dir, "cleaned_text.txt")
    cleaned_text = None
    if os.path.exists(cleaned_text_path):
        with open(cleaned_text_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()

    return metadata, chunks, tfidf_index, sentences, sections, cleaned_text


# ===========================================================================
# TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.pdf"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        print("Usage: python preprocess.py path/to/document.pdf")
        sys.exit(1)

    cleaned_text, sections, sentences, chunks, metadata, tfidf_index = \
        preprocess_document(pdf_path)

    # -------------------------------------------------------------------
    print("\n--- DOCUMENT METADATA ---")
    print(f"  Type        : {metadata['doc_type']}")
    print(f"  Parties     : {', '.join(metadata['parties'][:5])}")
    print(f"  Dates       : {', '.join(metadata['dates'][:3])}")
    print(f"  Jurisdiction: {metadata['jurisdiction']}")

    # -------------------------------------------------------------------
    print("\n--- DETECTED SECTIONS ---")
    for i, section in enumerate(sections[:10]):
        print(f"  {i+1:2d}. Pos {section['start']:6d} | {section['name'][:70]}")

    # -------------------------------------------------------------------
    print("\n--- CHUNK SUMMARY ---")
    for chunk in chunks:
        print(f"  Chunk {chunk['chunk_index']+1:2d} | "
              f"Words: {chunk['word_count']:4d} | "
              f"Pages: {chunk['page_start']}-{chunk['page_end']} | "
              f"Section: {chunk['section'][:40]}")

    # -------------------------------------------------------------------
    print("\n--- FIRST CHUNK PREVIEW ---")
    print(chunks[0]['text'][:500])

    # -------------------------------------------------------------------
    print("\n--- RETRIEVAL TEST ---")
    test_question = "What are the termination conditions?"
    print(f"  Question: {test_question}")
    print(f"  Expanded: {expand_query(test_question)}")

    _, loaded_chunks, loaded_index = load_document_store(
        metadata['doc_id']
    )
    results = retrieve_chunks(test_question, loaded_index, loaded_chunks, k=3)
    for i, result in enumerate(results):
        print(f"\n  Result {i+1} | Score: {result['retrieval_score']:.4f} | "
              f"Section: {result['section'][:40]} | "
              f"Pages: {result['page_start']}-{result['page_end']}")
        print(f"  Preview: {result['body'][:200]}...")