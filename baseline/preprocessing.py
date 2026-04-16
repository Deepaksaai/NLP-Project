import pdfplumber
import re
import spacy

# Load spaCy model once at module level
# This avoids reloading it every time the function is called
nlp = spacy.load('en_core_web_sm')


def extract_text_from_pdf(pdf_path):
    """
    Task 1: Extracts raw text from a digital PDF file.
    Input: path to PDF file
    Output: raw text string
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text is None or text.strip() == "":
                continue

            full_text.append(text)

    raw_text = "\n".join(full_text)

    # Fix hyphenated line breaks
    raw_text = re.sub(r'-\n', '', raw_text)

    # Fix irregular line breaks within sentences
    raw_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text)

    # Fix multiple spaces
    raw_text = re.sub(r' +', ' ', raw_text)

    # Fix weird unicode characters
    raw_text = raw_text.replace('\xa0', ' ')
    raw_text = raw_text.replace('\x0c', '\n')

    return raw_text.strip()


def clean_text(raw_text):
    """
    Task 2: Cleans raw extracted text from legal PDF.
    General purpose - works for any legal document.
    Input: raw text string from extract_text_from_pdf()
    Output: cleaned text string
    """

    # -------------------------------------------------------
    # BLOCK 1: Truncate after signature page
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # BLOCK 2: Remove legal document stamps and notices
    # -------------------------------------------------------

    # Remove redaction notices
    raw_text = re.sub(
        r'CERTAIN INFORMATION.*?REDACTED\.',
        '',
        raw_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove all forms of redaction markers
    raw_text = re.sub(r'\[\*+\]', '', raw_text)
    raw_text = re.sub(r'\[REDACTED\]', '', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\[Redacted\]', '', raw_text)

    # Remove leftover redaction fragment
    raw_text = re.sub(
        r'OR\s+INDICATES\s+THAT\s+INFORMATION\s+HAS\s+BEEN\s+REDACTED\.?',
        '',
        raw_text,
        flags=re.IGNORECASE
    )

    # Remove common document stamps
    raw_text = re.sub(r'\bEXECUTION\s+COPY\b', '', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\bCONFIDENTIAL\b', '', raw_text)
    raw_text = re.sub(r'\bDRAFT\b', '', raw_text)
    raw_text = re.sub(r'\bPROPRIETARY\b', '', raw_text)

    # -------------------------------------------------------
    # BLOCK 3: Remove headers and footers
    # -------------------------------------------------------

    # Remove source citation footers
    raw_text = re.sub(
        r'Source:\s+.*?\d{4}',
        '',
        raw_text,
        flags=re.IGNORECASE
    )

    # Remove exhibit references inline and standalone
    raw_text = re.sub(
        r'\bExhibit\s+[\w.-]+\b',
        '',
        raw_text,
        flags=re.IGNORECASE
    )

    # Remove exhibit page markers
    raw_text = re.sub(r'\bExh\.\s+[A-Z]-\d+\b', '', raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\bEXH\.\s+[A-Z]-\d+\b', '', raw_text, flags=re.IGNORECASE)

    # Remove exhibit section headers
    raw_text = re.sub(
        r'EXHIBIT\s+[A-Z]\s+[A-Z\s&/]+',
        '',
        raw_text
    )

    # -------------------------------------------------------
    # BLOCK 4: Line by line cleaning
    # -------------------------------------------------------

    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip standalone page numbers
        if re.match(r'^-?\s*\d+\s*-?$', line):
            continue
        if re.match(r'^[Pp]age\s+\d+(\s+of\s+\d+)?$', line):
            continue

        # Skip schedule and annex headers
        if re.match(r'^(Schedule|Annex|Appendix)\s+[\w.]+\s*$',
                    line, re.IGNORECASE):
            continue

        # Skip short all-caps lines
        if line.isupper() and len(line.split()) <= 5:
            continue

        # Skip lines with only special characters
        if re.match(r'^[\W_]+$', line):
            continue

        # Skip very short lines
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    # -------------------------------------------------------
    # BLOCK 5: Final cleanup
    # -------------------------------------------------------

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)

    return cleaned_text.strip()


def detect_sections(cleaned_text):
    """
    Task 3: Detects legal section headers in cleaned text.
    General purpose - works for any legal document.
    Input: cleaned text string from clean_text()
    Output: list of dicts with section name and character position
    """

    sections = []

    section_patterns = [
        # ARTICLE 1, ARTICLE I, ARTICLE 1.
        r'(?<!\w)ARTICLE\s+(?:[IVX]+|\d+)\.?\s*\n?\s*([A-Z][A-Z\s&;,]+)',

        # SECTION 1, SECTION 1., Section 1.
        r'(?<!\w)SECTION\s+\d+\.?\s*\n?\s*([A-Z][A-Z\s&;,]+)',

        # 1. DEFINITIONS or 1.1 Definitions (numbered sections)
        r'^\d+\.(?:\d+)?\s+([A-Z][A-Z\s&;,]{3,})',

        # Preamble markers
        r'(?<!\w)(WHEREAS|WITNESSETH|NOW[,\s]+THEREFORE)[,:]?',

        # Signature block marker
        r'(?<!\w)(IN\s+WITNESS\s+WHEREOF)',

        # Common legal section keywords standing alone
        r'^(RECITALS|DEFINITIONS|REPRESENTATIONS AND WARRANTIES|'
        r'INDEMNIFICATION|MISCELLANEOUS|GOVERNING LAW|'
        r'CONFIDENTIALITY|TERM AND TERMINATION|'
        r'LIMITATION OF LIABILITY|DISPUTE RESOLUTION)$',
    ]

    for pattern in section_patterns:
        for match in re.finditer(pattern, cleaned_text, flags=re.MULTILINE):
            section_name = match.group(0).strip()
            section_name = re.sub(r'\s+', ' ', section_name)
            sections.append({
                'name': section_name,
                'start': match.start(),
                'end': match.end()
            })

    # Sort sections by position in document
    sections = sorted(sections, key=lambda x: x['start'])

    # Remove duplicates where patterns overlap
    unique_sections = []
    last_pos = -1
    for section in sections:
        if section['start'] > last_pos + 10:
            unique_sections.append(section)
            last_pos = section['start']

    return unique_sections


def segment_sentences(cleaned_text):
    """
    Task 4: Splits cleaned legal text into individual sentences.
    Uses spaCy with custom rules for legal text edge cases.
    Input: cleaned text string from clean_text()
    Output: list of sentence strings
    """

    # -------------------------------------------------------
    # BLOCK 1: Add custom rules for legal abbreviations
    # -------------------------------------------------------

    legal_abbreviations = [
        'Inc', 'Corp', 'Ltd', 'LLC', 'LLP', 'Co',
        'No', 'Sec', 'Art', 'vs', 'etc', 'e.g', 'i.e',
        'U.S', 'U.K', 'Fig', 'Dept', 'Est', 'approx',
        'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul',
        'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr'
    ]

    for abbr in legal_abbreviations:
        nlp.tokenizer.add_special_case(
            f'{abbr}.',
            [{'ORTH': f'{abbr}.'}]
        )

    # -------------------------------------------------------
    # BLOCK 2: Set max length and process text
    # -------------------------------------------------------

    nlp.max_length = 2000000

    doc = nlp(cleaned_text)

    # -------------------------------------------------------
    # BLOCK 3: Extract sentences with quality filters
    # -------------------------------------------------------

    sentences = []

    for sent in doc.sents:
        sentence = sent.text.strip()

        # Skip empty sentences
        if not sentence:
            continue

        # Skip very short sentences
        if len(sentence) < 10:
            continue

        # Skip sentences that are just numbers or symbols
        if re.match(r'^[\d\s\W]+$', sentence):
            continue

        sentences.append(sentence)

    return sentences


def chunk_document(cleaned_text, sentences, sections, chunk_size=800, overlap=50):
    """
    Task 5: Splits document into overlapping chunks for model input.
    Uses section boundaries to guide splitting where possible.
    Input: cleaned text, sentences list, sections list
    Output: list of dicts with chunk text and metadata
    """

    # -------------------------------------------------------
    # BLOCK 1: Simple whitespace token counter
    # Used for size estimation not actual model tokenization
    # -------------------------------------------------------

    def count_tokens(text):
        return len(text.split())

    # -------------------------------------------------------
    # BLOCK 2: Get section boundary sentence indices
    # -------------------------------------------------------

    section_starts = set()
    for section in sections:
        for i, sentence in enumerate(sentences):
            if sentence.strip() in cleaned_text[section['start']:section['start']+200]:
                section_starts.add(i)
                break

    # -------------------------------------------------------
    # BLOCK 3: Build chunks greedily with section awareness
    # -------------------------------------------------------

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    chunk_index = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)

        # Check if adding this sentence exceeds chunk size
        if current_token_count + sentence_tokens > chunk_size and current_chunk_sentences:

            # Look ahead up to 3 sentences for a section boundary
            # Prefer to split at natural section boundaries
            for lookahead in range(1, 4):
                if i + lookahead < len(sentences) and (i + lookahead) in section_starts:
                    for j in range(lookahead):
                        current_chunk_sentences.append(sentences[i + j])
                        current_token_count += count_tokens(sentences[i + j])
                    i += lookahead
                    break

            # Save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                'chunk_index': chunk_index,
                'text': chunk_text,
                'token_count': current_token_count,
                'sentence_count': len(current_chunk_sentences)
            })
            chunk_index += 1

            # Start new chunk with overlap
            # Take last sentences worth up to overlap token limit
            overlap_sentences = []
            overlap_token_count = 0

            for sent in reversed(current_chunk_sentences):
                sent_tokens = count_tokens(sent)
                if overlap_token_count + sent_tokens <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_token_count += sent_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_token_count

        else:
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            i += 1

    # Save the last remaining chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append({
            'chunk_index': chunk_index,
            'text': chunk_text,
            'token_count': current_token_count,
            'sentence_count': len(current_chunk_sentences)
        })

    return chunks


def preprocess_document(pdf_path):
    """
    Master function: runs all preprocessing tasks in sequence.
    Input: path to PDF file
    Output: cleaned text, detected sections, segmented sentences, chunks
    """
    print(f"\nProcessing: {pdf_path}")

    # Task 1 - Extract
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Task 1 done — Extracted {len(raw_text)} characters")

    # Task 2 - Clean
    cleaned_text = clean_text(raw_text)
    print(f"Task 2 done — Cleaned to {len(cleaned_text)} characters "
          f"({len(raw_text) - len(cleaned_text)} removed)")

    # Task 3 - Detect sections
    sections = detect_sections(cleaned_text)
    print(f"Task 3 done — Detected {len(sections)} sections")

    # Task 4 - Segment sentences
    sentences = segment_sentences(cleaned_text)
    print(f"Task 4 done — Segmented into {len(sentences)} sentences")

    # Task 5 - Chunk document
    chunks = chunk_document(cleaned_text, sentences, sections)
    print(f"Task 5 done — Created {len(chunks)} chunks")

    return cleaned_text, sections, sentences, chunks


# Single combined test block
if __name__ == "__main__":
    pdf_path = "sample_contract.pdf"

    cleaned_text, sections, sentences, chunks = preprocess_document(pdf_path)

    print("\n--- DETECTED SECTIONS ---")
    for i, section in enumerate(sections):
        print(f"{i+1}. Position {section['start']:5d} | {section['name'][:80]}")

    print("\n--- CHUNK SUMMARY ---")
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']+1:2d} | "
              f"Tokens: {chunk['token_count']:4d} | "
              f"Sentences: {chunk['sentence_count']:3d}")

    print("\n--- FIRST CHUNK PREVIEW ---")
    print(chunks[0]['text'][:500])

    print("\n--- LAST CHUNK PREVIEW ---")
    print(chunks[-1]['text'][:500])