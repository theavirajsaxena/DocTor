# chunker.py
import spacy
import re

nlp = spacy.load("en_core_web_sm")

MIN_CHUNK_WORDS  = 60
MAX_CHUNK_WORDS  = 200
OVERLAP_SENTENCES = 2


def _split_into_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _count_words(text: str) -> int:
    return len(text.split())


def _is_section_header(sentence: str) -> bool:
    s = sentence.strip()
    if len(s.split()) > 10:
        return False
    if s.endswith(":"):
        return True
    if s.isupper():
        return True
    if re.match(r'^(\d+\.)+\s+[A-Z]', s):
        return True
    return False


def _build_chunks_from_sentences(
    sentences : list[str],
    page_num  : int
) -> list[dict]:
    chunks   = []
    buffer   = []
    chunk_id = 0

    for i, sentence in enumerate(sentences):
        word_count  = _count_words(" ".join(buffer + [sentence]))
        at_boundary = _is_section_header(sentence) and len(buffer) > 0
        should_flush = (word_count > MAX_CHUNK_WORDS) or at_boundary

        if should_flush and _count_words(" ".join(buffer)) >= MIN_CHUNK_WORDS:
            chunk_text = " ".join(buffer).strip()
            chunks.append({
                "chunk_id"  : chunk_id,
                "page"      : page_num,
                "text"      : chunk_text,
                "word_count": _count_words(chunk_text)
            })
            chunk_id += 1
            buffer = buffer[-OVERLAP_SENTENCES:] + [sentence]
        else:
            buffer.append(sentence)

    if buffer and _count_words(" ".join(buffer)) >= MIN_CHUNK_WORDS // 2:
        chunk_text = " ".join(buffer).strip()
        chunks.append({
            "chunk_id"  : chunk_id,
            "page"      : page_num,
            "text"      : chunk_text,
            "word_count": _count_words(chunk_text)
        })

    return chunks


def adaptive_chunk(pages: list[dict]) -> list[dict]:
    all_chunks = []

    for page_data in pages:
        page_num    = page_data["page"]
        page_text   = page_data["text"]
        sentences   = _split_into_sentences(page_text)
        page_chunks = _build_chunks_from_sentences(sentences, page_num)
        all_chunks.extend(page_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    return all_chunks