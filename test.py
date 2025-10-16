# ====== COMP479 Project 1 Driver ======
"""Build Reuters-21578 indexes and run the assignment subprojects."""

import os
import time
from collections import defaultdict

import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure the NLTK resources required by the assignment are available.
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

REUTERS_DIR = "reuters21578"


# ===== Shared helpers =====
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


# ===== Subproject I – Naive indexer =====
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


# ===== Subproject II – Single-term and AND query processing =====
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


# ===== Subproject III – Lossy dictionary compression experiments =====
EN_STOP = set(stopwords.words("english"))
TOP30 = {"the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not", "are", "but", "from", "or", "have", "an", "they"}
STOP150 = set(sorted(EN_STOP)[:150])
STEMMER = PorterStemmer()


def keep_all(tokens):
    return tokens


def drop_numbers(tokens):
    return [t for t in tokens if not any(ch.isdigit() for ch in t)]


def case_fold(tokens):
    return [t.lower() for t in tokens]


def drop_numbers_then_case(tokens):
    return case_fold(drop_numbers(tokens))


def stop_30(tokens):
    folded = drop_numbers_then_case(tokens)
    return [t for t in folded if t not in TOP30]


def stop_150(tokens):
    folded = drop_numbers_then_case(tokens)
    return [t for t in folded if t not in STOP150]


def stemmed(tokens):
    filtered = [t for t in drop_numbers_then_case(tokens) if t not in EN_STOP]
    return [STEMMER.stem(t) if any(ch.isalpha() for ch in t) else t for t in filtered]


BASE_VARIANT = ("UNFILTERED", keep_all)
COMPRESSION_VARIANTS = [
    ("NO NUMBERS", drop_numbers),
    ("CASE FOLD", drop_numbers_then_case),
    ("STOP 30", stop_30),
    ("STOP 150", stop_150),
    ("STEMMING", stemmed),
]


def describe_index(index):
    return len(index), sum(len(pl) for pl in index.values())


def format_int(number):
    return f"{number:,}"


def run_compression_table(index_builds):
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


def compare_query_results(name, transform, index, samples):
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


# ===== Subproject IV – SPIMI indexer =====
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


def time_builder(builder, tokenised_docs, transform, label):
    start = time.perf_counter()
    index = builder(tokenised_docs, transform)
    elapsed = time.perf_counter() - start
    print(f" {label:<8} -> {elapsed:.3f}s")
    return index, elapsed


def limit_pairs(tokenised_docs, transform, limit):
    total = 0
    trimmed = []
    for doc in tokenised_docs:
        trimmed.append(doc)
        total += len(transform(doc[1]))
        if total >= limit:
            break
    return trimmed


# ===== Assignment runner =====
SAMPLE_SINGLE_QUERIES = [["oil"], ["market"], ["british"]]
SAMPLE_AND_QUERIES = [["oil", "market"], ["gold", "prices"], ["u.s.", "trade"]]


def main():
    docs = tokenize_documents(load_reuters_docs(REUTERS_DIR))

    # --- Subproject I: naive indexer -------------------------------------
    print("Subproject I – naive indexer")
    base_name, base_transform = BASE_VARIANT
    base_index, pair_count, unique_pairs = naive_build_index(docs, base_transform)
    dict_size, postings_size = describe_index(base_index)
    print(f" {base_name:<16} | F={format_int(pair_count):>10} | unique={format_int(unique_pairs):>10} | "
          f"dict={format_int(dict_size):>10} | postings={format_int(postings_size):>10}")
    base_entry = (base_name, base_transform, base_index)

    # --- Subproject II: query processor validation -----------------------
    print("\nSubproject II – query processing samples")
    samples = SAMPLE_SINGLE_QUERIES + SAMPLE_AND_QUERIES
    compare_query_results(base_name, base_transform, base_index, samples)

    # --- Subproject III: lossy dictionary compression --------------------
    print("\nSubproject III – lossy dictionary compression table")
    compression_indexes = [base_entry]
    for name, transform in COMPRESSION_VARIANTS:
        index, _, _ = naive_build_index(docs, transform)
        compression_indexes.append((name, transform, index))
    run_compression_table([(name, index) for name, _, index in compression_indexes])

    # Compare sample query behaviour on the case folded index
    case_fold_index = next(idx for name, _, idx in compression_indexes if name == "CASE FOLD")
    compare_query_results("CASE FOLD (compressed)", drop_numbers_then_case, case_fold_index, samples)

    # --- Subproject IV: SPIMI indexer ------------------------------------
    print("\nSubproject IV – SPIMI timing comparison")
    subset = limit_pairs(docs, case_fold, 10_000)
    _, naive_time = time_builder(lambda d, t: naive_build_index(d, t)[0], subset, case_fold, "Naive")
    _, spimi_time = time_builder(spimi_build_index, subset, case_fold, "SPIMI")
    print(f" Speedup (Naive/SPIMI): {naive_time / spimi_time if spimi_time else float('inf'):.2f}x")


if __name__ == "__main__":
    main()

