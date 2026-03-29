# chunker.py
import spacy
import re

# Load the English NLP model (downloaded in Step 1)
nlp = spacy.load("en_core_web_sm")

# ── tuneable parameters ──────────────────────────────────────────────
MIN_CHUNK_WORDS  = 60    # a chunk must have at least this many words
MAX_CHUNK_WORDS  = 200   # a chunk must not exceed this many words
OVERLAP_SENTENCES = 2    # sentences shared between consecutive chunks
                         # (preserves cross-boundary context)
# ─────────────────────────────────────────────────────────────────────


def _split_into_sentences(text: str) -> list[str]:
    """Use spaCy to split a block of text into individual sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _count_words(text: str) -> int:
    return len(text.split())


def _is_section_header(sentence: str) -> bool:
    """
    Detect likely section headers — short lines that are
    ALL CAPS, Title Case, or end with a colon.
    These are strong signals of a topic boundary.
    """
    s = sentence.strip()
    if len(s.split()) > 10:          # headers are short
        return False
    if s.endswith(":"):              # e.g. "Introduction:"
        return True
    if s.isupper():                  # e.g. "METHODOLOGY"
        return True
    if re.match(r'^(\d+\.)+\s+[A-Z]', s):  # e.g. "2.1 Related Work"
        return True
    return False


def _build_chunks_from_sentences(
    sentences : list[str],
    page_num  : int
) -> list[dict]:
    """
    Core adaptive logic:
    Accumulate sentences into a chunk. Flush (save) the chunk when:
      • adding the next sentence would exceed MAX_CHUNK_WORDS, OR
      • the next sentence looks like a section header (topic shift).
    Then start the new chunk with OVERLAP_SENTENCES from the end of
    the previous chunk so context is never lost at boundaries.
    """
    chunks   = []
    buffer   = []   # sentences being accumulated into current chunk
    chunk_id = 0

    for i, sentence in enumerate(sentences):
        word_count = _count_words(" ".join(buffer + [sentence]))
        at_boundary = _is_section_header(sentence) and len(buffer) > 0

        should_flush = (word_count > MAX_CHUNK_WORDS) or at_boundary

        if should_flush and _count_words(" ".join(buffer)) >= MIN_CHUNK_WORDS:
            # ── save the current chunk ──
            chunk_text = " ".join(buffer).strip()
            chunks.append({
                "chunk_id" : chunk_id,
                "page"     : page_num,
                "text"     : chunk_text,
                "word_count": _count_words(chunk_text)
            })
            chunk_id += 1

            # ── start new buffer with overlap ──
            buffer = buffer[-OVERLAP_SENTENCES:] + [sentence]
        else:
            buffer.append(sentence)

    # ── flush whatever is left in the buffer ──
    if buffer and _count_words(" ".join(buffer)) >= MIN_CHUNK_WORDS // 2:
        chunk_text = " ".join(buffer).strip()
        chunks.append({
            "chunk_id" : chunk_id,
            "page"     : page_num,
            "text"     : chunk_text,
            "word_count": _count_words(chunk_text)
        })

    return chunks


def adaptive_chunk(pages: list[dict]) -> list[dict]:
    """
    Main entry point.
    Takes the list of pages from extractor.py and returns
    a flat list of all chunks across the entire document.
    Each chunk carries its source page number for citations later.
    """
    all_chunks = []

    for page_data in pages:
        page_num  = page_data["page"]
        page_text = page_data["text"]

        sentences = _split_into_sentences(page_text)
        page_chunks = _build_chunks_from_sentences(sentences, page_num)
        all_chunks.extend(page_chunks)

    # Re-assign global chunk IDs across the whole document
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    return all_chunks