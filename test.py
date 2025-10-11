# ====== COMP479 Project 1 Driver ======
"""Utilities to build Reuters-21578 indexes and run the four subprojects.

The script is organised to mirror the assignment description:

Subproject I   – Naive indexer
Subproject II  – Query processing
Subproject III – Lossy dictionary compression experiments
Subproject IV  – SPIMI indexer and timing comparison
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure the NLTK resources required by the assignment are available.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

REUTERS_DIR = "reuters21578"
DocID = int
PostingList = List[DocID]
InvertedIndex = Dict[str, PostingList]
TokenTransform = Callable[[Sequence[str]], Sequence[str]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_reuters_docs(path: str) -> List[Tuple[DocID, str]]:
    """Parse the SGML Reuters dump and keep ``(doc_id, raw_text)`` tuples."""

    docs: List[Tuple[DocID, str]] = []
    for filename in os.listdir(path):
        if not filename.endswith(".sgm"):
            continue
        with open(os.path.join(path, filename), "r", encoding="latin-1") as handle:
            soup = BeautifulSoup(handle.read(), "html.parser")
        for item in soup.find_all("reuters"):
            doc_id = int(item["newid"])
            text = item.find("text")
            if text:
                docs.append((doc_id, text.get_text(" ", strip=True)))
    return docs


def tokenize_documents(docs: Iterable[Tuple[DocID, str]]) -> List[Tuple[DocID, List[str]]]:
    """Tokenise every Reuters document exactly once for reuse across variants."""

    return [(doc_id, word_tokenize(text)) for doc_id, text in docs]


# ---------------------------------------------------------------------------
# Subproject I – Naive indexer
# ---------------------------------------------------------------------------
def naive_pairs(tokenised_docs: Sequence[Tuple[DocID, Sequence[str]]], transform: TokenTransform) -> List[Tuple[str, DocID]]:
    """Collect term–docID pairs (list ``F``) using the naive algorithm."""

    pairs: List[Tuple[str, DocID]] = []
    for doc_id, tokens in tokenised_docs:
        for term in transform(tokens):
            pairs.append((term, doc_id))
    return pairs


def dedupe_sorted_pairs(pairs: Sequence[Tuple[str, DocID]]) -> List[Tuple[str, DocID]]:
    """Remove duplicates from a sorted list of term–docID pairs."""

    deduped: List[Tuple[str, DocID]] = []
    prev: Tuple[str, DocID] | None = None
    for pair in pairs:
        if pair != prev:
            deduped.append(pair)
            prev = pair
    return deduped


def build_index_from_pairs(pairs: Iterable[Tuple[str, DocID]]) -> InvertedIndex:
    index: InvertedIndex = defaultdict(list)
    for term, doc_id in pairs:
        postings = index[term]
        if not postings or postings[-1] != doc_id:
            postings.append(doc_id)
    return index


def naive_build_index(tokenised_docs: Sequence[Tuple[DocID, Sequence[str]]], transform: TokenTransform) -> Tuple[InvertedIndex, int, int]:
    """Apply the three-step naive indexing pipeline and return stats."""

    pairs = naive_pairs(tokenised_docs, transform)
    pairs.sort()
    unique_pairs = dedupe_sorted_pairs(pairs)
    index = build_index_from_pairs(unique_pairs)
    return index, len(pairs), len(unique_pairs)


# ---------------------------------------------------------------------------
# Subproject II – Single-term and AND query processing
# ---------------------------------------------------------------------------
def search_single(term: str, index: InvertedIndex) -> PostingList:
    return index.get(term, [])


def intersect(postings_a: Sequence[DocID], postings_b: Sequence[DocID]) -> PostingList:
    ia = ib = 0
    out: PostingList = []
    while ia < len(postings_a) and ib < len(postings_b):
        a, b = postings_a[ia], postings_b[ib]
        if a == b:
            out.append(a)
            ia += 1
            ib += 1
        elif a < b:
            ia += 1
        else:
            ib += 1
    return out


def search_and(terms: Sequence[str], index: InvertedIndex) -> PostingList:
    if not terms:
        return []
    result = list(search_single(terms[0], index))
    for term in terms[1:]:
        result = intersect(result, search_single(term, index))
        if not result:
            break
    return result


# ---------------------------------------------------------------------------
# Subproject III – Lossy dictionary compression experiments
# ---------------------------------------------------------------------------
EN_STOP = set(stopwords.words("english"))
TOP30 = {"the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not", "are", "but", "from", "or", "have", "an", "they"}
STOP150 = set(sorted(EN_STOP)[:150])
STEMMER = PorterStemmer()


def keep_all(tokens: Sequence[str]) -> Sequence[str]:
    return tokens


def drop_numbers(tokens: Sequence[str]) -> Sequence[str]:
    return [t for t in tokens if not any(ch.isdigit() for ch in t)]


def case_fold(tokens: Sequence[str]) -> Sequence[str]:
    return [t.lower() for t in tokens]


def stop_30(tokens: Sequence[str]) -> Sequence[str]:
    lower = case_fold(tokens)
    return [t for t in lower if t not in TOP30]


def stop_150(tokens: Sequence[str]) -> Sequence[str]:
    lower = case_fold(tokens)
    return [t for t in lower if t not in STOP150]


def stemmed(tokens: Sequence[str]) -> Sequence[str]:
    lower = [t for t in case_fold(tokens) if t not in EN_STOP]
    return [STEMMER.stem(t) if any(ch.isalpha() for ch in t) else t for t in lower]


VARIANTS: List[Tuple[str, TokenTransform]] = [
    ("UNFILTERED", keep_all),
    ("CASE FOLD", case_fold),  # column 1 → column 2 from Table 5.1 (lossy)
    ("NO NUMBERS", drop_numbers),
    ("STOP 30", stop_30),
    ("STOP 150", stop_150),
    ("STEMMED", stemmed),
]


def describe_index(index: InvertedIndex) -> Tuple[int, int]:
    return len(index), sum(len(pl) for pl in index.values())


def format_int(number: int) -> str:
    return f"{number:,}"


def run_compression_table(index_builds: List[Tuple[str, InvertedIndex]]) -> None:
    print("\nDictionary compression impact (relative deltas)")
    print("-" * 100)
    print(f"{'Variant':<16} | {'Dict':>10} | {'ΔPrev%':>10} | {'ΔBase%':>10} | "
          f"{'Postings':>10} | {'ΔPrev%':>10} | {'ΔBase%':>10}")
    print("-" * 100)

    base_dict, base_post = describe_index(index_builds[0][1])
    prev_dict, prev_post = base_dict, base_post
    for name, index in index_builds:
        cur_dict, cur_post = describe_index(index)
        prev_dict_delta = round(100.0 * (cur_dict - prev_dict) / prev_dict, 1)
        base_dict_delta = round(100.0 * (cur_dict - base_dict) / base_dict, 1)
        prev_post_delta = round(100.0 * (cur_post - prev_post) / prev_post, 1)
        base_post_delta = round(100.0 * (cur_post - base_post) / base_post, 1)

        print(f"{name:<16} | {format_int(cur_dict):>10} | {prev_dict_delta:+9.1f}% | {base_dict_delta:+9.1f}% | "
              f"{format_int(cur_post):>10} | {prev_post_delta:+9.1f}% | {base_post_delta:+9.1f}%")

        prev_dict, prev_post = cur_dict, cur_post
    print("-" * 100)


def compare_query_results(name: str, transform: TokenTransform, index: InvertedIndex, samples: List[List[str]]) -> None:
    print(f"\n[{name}] query results")
    for query in samples:
        transformed = list(transform(query))
        if not transformed:
            print(f" {query} -> all terms removed by preprocessing")
            continue
        if len(transformed) == 1:
            hits = search_single(transformed[0], index)
            label = "single"
        else:
            hits = search_and(transformed, index)
            label = "AND"
        print(f" {label:<6} {query} -> {len(hits)} docs (sample {hits[:10]})")


# ---------------------------------------------------------------------------
# Subproject IV – SPIMI indexer
# ---------------------------------------------------------------------------
def spimi_build_index(tokenised_docs: Sequence[Tuple[DocID, Sequence[str]]], transform: TokenTransform) -> InvertedIndex:
    index: InvertedIndex = defaultdict(list)
    for doc_id, tokens in tokenised_docs:
        for term in transform(tokens):
            postings = index[term]
            if postings and postings[-1] == doc_id:
                continue
            postings.append(doc_id)
    for postings in index.values():
        postings.sort()
    return index


def time_builder(builder: Callable[[Sequence[Tuple[DocID, Sequence[str]]], TokenTransform], InvertedIndex],
                 tokenised_docs: Sequence[Tuple[DocID, Sequence[str]]], transform: TokenTransform,
                 label: str) -> Tuple[InvertedIndex, float]:
    start = time.perf_counter()
    index = builder(tokenised_docs, transform)
    elapsed = time.perf_counter() - start
    print(f" {label:<8} -> {elapsed:.3f}s")
    return index, elapsed


def limit_pairs(tokenised_docs: Sequence[Tuple[DocID, Sequence[str]]], transform: TokenTransform, limit: int) -> List[Tuple[DocID, Sequence[str]]]:
    """Trim documents so that the naive list ``F`` is roughly ``limit`` pairs long."""

    total = 0
    trimmed: List[Tuple[DocID, Sequence[str]]] = []
    for doc in tokenised_docs:
        trimmed.append(doc)
        total += len(transform(doc[1]))
        if total >= limit:
            break
    return trimmed


# ---------------------------------------------------------------------------
# Assignment runner
# ---------------------------------------------------------------------------
SAMPLE_SINGLE_QUERIES = [["oil"], ["market"], ["british"]]
SAMPLE_AND_QUERIES = [["oil", "market"], ["gold", "prices"], ["u.s.", "trade"]]


def main() -> None:
    docs = tokenize_documents(load_reuters_docs(REUTERS_DIR))

    # --- Subproject I: naive indexer -------------------------------------
    print("Subproject I – naive indexer")
    naive_indexes: List[Tuple[str, TokenTransform, InvertedIndex]] = []
    for name, transform in VARIANTS:
        index, pair_count, unique_pairs = naive_build_index(docs, transform)
        dict_size, postings_size = describe_index(index)
        print(f" {name:<16} | F={format_int(pair_count):>10} | unique={format_int(unique_pairs):>10} | "
              f"dict={format_int(dict_size):>10} | postings={format_int(postings_size):>10}")
        naive_indexes.append((name, transform, index))

    # --- Subproject II: query processor validation -----------------------
    print("\nSubproject II – query processing samples")
    samples = SAMPLE_SINGLE_QUERIES + SAMPLE_AND_QUERIES
    for name, transform, index in naive_indexes:
        compare_query_results(name, transform, index, samples)

    # --- Subproject III: lossy dictionary compression --------------------
    print("\nSubproject III – lossy dictionary compression table")
    run_compression_table([(name, index) for name, _, index in naive_indexes])

    # Compare sample query behaviour on the compressed (case-folded) index
    case_fold_index = next(idx for name, _, idx in naive_indexes if name == "CASE FOLD")
    compare_query_results("CASE FOLD (compressed)", case_fold, case_fold_index, samples)

    # --- Subproject IV: SPIMI indexer ------------------------------------
    print("\nSubproject IV – SPIMI timing comparison")
    subset = limit_pairs(docs, case_fold, 10_000)
    _, naive_time = time_builder(lambda d, t: naive_build_index(d, t)[0], subset, case_fold, "Naive")
    _, spimi_time = time_builder(spimi_build_index, subset, case_fold, "SPIMI")
    print(f" Speedup (Naive/SPIMI): {naive_time / spimi_time if spimi_time else float('inf'):.2f}x")


if __name__ == "__main__":
    main()

