# ====== Challenge Queries Runner ======
import os
import time
from collections import defaultdict

import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize

nltk.download("punkt", quiet=True)

REUTERS_DIR = "reuters21578"


# ===== Load and tokenize =====
def load_reuters_docs(path):
    docs = []
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


def tokenize_documents(docs):
    tokenised = []
    for doc_id, text in docs:
        tokenised.append((doc_id, word_tokenize(text)))
    return tokenised


def keep_all(tokens):
    return tokens


# ===== Naive indexer =====
def naive_pairs(tokenised_docs, transform):
    F = []
    for doc_id, tokens in tokenised_docs:
        for term in transform(tokens):
            F.append((term, doc_id))
    return F


def dedupe_sorted_pairs(pairs):
    deduped = []
    prev = None
    for pair in pairs:
        if pair != prev:
            deduped.append(pair)
            prev = pair
    return deduped


def build_index_from_pairs(pairs):
    index = defaultdict(list)
    for term, doc_id in pairs:
        postings = index[term]
        if not postings or postings[-1] != doc_id:
            postings.append(doc_id)
    return index


def naive_build_index(tokenised_docs, transform):
    F = naive_pairs(tokenised_docs, transform)
    F.sort()
    unique_pairs = dedupe_sorted_pairs(F)
    index = build_index_from_pairs(unique_pairs)
    return index, len(F), len(unique_pairs)


# ===== SPIMI indexer =====
def spimi_build_index(tokenised_docs, transform):
    index = defaultdict(list)
    for doc_id, tokens in tokenised_docs:
        for term in transform(tokens):
            postings = index[term]
            if postings and postings[-1] == doc_id:
                continue
            postings.append(doc_id)
    for postings in index.values():
        postings.sort()
    return index


# ===== Query processing =====
def search_single(term, index):
    return index.get(term, [])


def intersect(postings_a, postings_b):
    ia = 0
    ib = 0
    out = []
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


def search_and(terms, index):
    if not terms:
        return []
    result = list(search_single(terms[0], index))
    for term in terms[1:]:
        result = intersect(result, search_single(term, index))
        if not result:
            break
    return result


# ===== Main =====
def main():
    print("Loading and tokenizing documents...")
    docs = tokenize_documents(load_reuters_docs(REUTERS_DIR))
    
    print("Building indexes (uncompressed, full corpus)...")
    naive_index_full, _, _ = naive_build_index(docs, keep_all)
    spimi_index_full = spimi_build_index(docs, keep_all)

    print(f"Naive index size: {len(naive_index_full)} terms")
    print(f"SPIMI index size: {len(spimi_index_full)} terms")

    # Challenge queries (raw terms, no transformation)
    CHALLENGE_QUERIES = [["Chrysler", "Bundesbank"], ["pineapple"]]
    
    def run_queries(index, label):
        print(f"\n=== Challenge Queries on {label} Index ===")
        for q in CHALLENGE_QUERIES:
            start = time.perf_counter()
            if len(q) == 1:
                results = search_single(q[0], index)
            else:
                results = search_and(q, index)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            print(f"Query {q} -> {len(results)} docs (sample {results[:10]}) | {elapsed:.3f} ms")
    
    run_queries(naive_index_full, "Naive")
    run_queries(spimi_index_full, "SPIMI")


if __name__ == "__main__":
    main()
