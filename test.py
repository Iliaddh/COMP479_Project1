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
# Load and tokenize the Reuters-21578 documents.
# keep_all: No transformation, keep all tokens.
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

# Tokenize a list of (doc_id, text) tuples into (doc_id, [tokens]) tuples.
def tokenize_documents(docs):
    tokenised = []
    for doc_id, text in docs:
        tokenised.append((doc_id, word_tokenize(text)))
    return tokenised

def keep_all(tokens):
    return tokens
BASE_VARIANT = ("UNFILTERED", keep_all)

# Describe an index by returning its dictionary size and total postings size.
def describe_index(index):
    return len(index), sum(len(pl) for pl in index.values())

# Format integers with commas for readability.
def format_int(number):
    return f"{number:,}"

# ===== Subproject I – Naive indexer =================================
# Build an inverted index using the naive algorithm provided.
def naive_pairs(tokenised_docs, transform):
    F = []
    for doc_id, tokens in tokenised_docs:
        for term in transform(tokens):
            F.append((term, doc_id))
    return F

# Remove duplicates from a sorted list of (term, doc_id) pairs.
def dedupe_sorted_pairs(pairs):
    deduped = []
    prev = None
    for pair in pairs:
        if pair != prev:
            deduped.append(pair)
            prev = pair
    return deduped

# Turn a list of (term, doc_id) pairs into an inverted index.
def build_index_from_pairs(pairs):
    index = defaultdict(list)
    # Loops through each term, doc_id pair and adds the doc_id to the postings list for the term
    for term, doc_id in pairs:
        postings = index[term]
        # Avoiding duplicates in postings list
        if not postings or postings[-1] != doc_id:
            postings.append(doc_id)
    return index

# Full naive index building process: generate pairs, sort, dedupe, and build index.
def naive_build_index(tokenised_docs, transform):
    F = naive_pairs(tokenised_docs, transform)
    F.sort()
    unique_pairs = dedupe_sorted_pairs(F)
    index = build_index_from_pairs(unique_pairs)
    return index, len(F), len(unique_pairs)


# ===== Subproject II – Single-term and AND query processing ====================================
# Search for a single term in the index, returning its postings list or an empty list.
def search_single(term, index):
    return index.get(term, [])

# Intersect two postings lists using the intersection algorithm.
def intersect(postings_a, postings_b):
    ia = 0
    ib = 0
    out = []
    # Intersection algorithm
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

#  Search for multiple terms using AND semantics, returning the intersection of their postings lists.
def search_and(terms, index):
    if not terms:
        return []
    result = list(search_single(terms[0], index))
    for term in terms[1:]:
        result = intersect(result, search_single(term, index))
        if not result:
            break
    return result


# 
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


# ===== Subproject IV – SPIMI indexer ==============================================
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

# Given a builder function, time how long it takes to build an index and print the result.
def time_builder(builder, tokenised_docs, transform, label):
    start = time.perf_counter()
    index = builder(tokenised_docs, transform)
    elapsed = time.perf_counter() - start
    print(f" {label:<8} -> {elapsed:.3f}s")
    return index, elapsed

# Limit the number of (doc_id, tokens) pairs processed to approximately 'limit' terms after transformation.
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
    # Build the base unfiltered index
    base_name, base_transform = BASE_VARIANT
    base_index, pair_count, unique_pairs = naive_build_index(docs, base_transform)
    dict_size, postings_size = describe_index(base_index)
    print(f" {base_name:<16} | F={format_int(pair_count):>10} | unique={format_int(unique_pairs):>10} | "
          f"dict={format_int(dict_size):>10} | postings={format_int(postings_size):>10}")
    

    # --- Subproject II: query processor validation -----------------------
    print("\nSubproject II – query processing samples")
    samples = SAMPLE_SINGLE_QUERIES + SAMPLE_AND_QUERIES
    compare_query_results(base_name, base_transform, base_index, samples)

    

    # --- Subproject IV: SPIMI indexer ------------------------------------
    print("\nSubproject IV – SPIMI timing comparison")
    # Use keep_all (no compression techniques) and limit to 10,000 term-docID pairings
    subset = limit_pairs(docs, keep_all, 10_000)
    _, naive_time = time_builder(lambda d, t: naive_build_index(d, t)[0], subset, keep_all, "Naive")
    _, spimi_time = time_builder(spimi_build_index, subset, keep_all, "SPIMI")
    print(f" Speedup (Naive/SPIMI): {naive_time / spimi_time if spimi_time else float('inf'):.2f}x")




# -------------------------------------------------------------------------------------------------------------------
# -------------------Sub project III and Challenge query codes removed for reference-----------------
# ------------------------------------------------------------------------------



# ===== Subproject III – Lossy dictionary compression experiments =====

 
# EN_STOP = set(stopwords.words("english"))
# TOP30 = {"the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not", "are", "but", "from", "or", "have", "an", "they"}
# STOP150 = set(sorted(EN_STOP)[:150])
# STEMMER = PorterStemmer()

# # keep_all: No transformation, keep all tokens.
# def keep_all(tokens):
#     return tokens

# # drop_numbers: Remove tokens containing any digits.
# def drop_numbers(tokens):
#     return [t for t in tokens if not any(ch.isdigit() for ch in t)]

# # case_fold: Convert all tokens to lowercase.
# def case_fold(tokens):
#     return [t.lower() for t in tokens]

# # case_fold after drop_numbers: Remove numbers, then case fold.
# def drop_numbers_then_case(tokens):
#     return case_fold(drop_numbers(tokens))

# # stop_30: Remove top 30 English stopwords after dropping numbers and case folding.
# def stop_30(tokens):
#     folded = drop_numbers_then_case(tokens)
#     return [t for t in folded if t not in TOP30]

# # stop_150: Remove top 150 English stopwords after dropping numbers and case folding.
# def stop_150(tokens):
#     folded = drop_numbers_then_case(tokens)
#     return [t for t in folded if t not in STOP150]

# # stemming: Apply Porter stemming after dropping numbers, case folding, and removing top 150 stopwords.
# def stemmed(tokens):
#     filtered = [t for t in drop_numbers_then_case(tokens) if t not in EN_STOP]
#     return [STEMMER.stem(t) if any(ch.isalpha() for ch in t) else t for t in filtered]

# #  Define the compression variants to test.
# BASE_VARIANT = ("UNFILTERED", keep_all)
# COMPRESSION_VARIANTS = [
#     ("NO NUMBERS", drop_numbers),
#     ("CASE FOLD", drop_numbers_then_case),
#     ("STOP 30", stop_30),
#     ("STOP 150", stop_150),
#     ("STEMMING", stemmed),
# ]

# # Describe an index by returning its dictionary size and total postings size.
# def describe_index(index):
#     return len(index), sum(len(pl) for pl in index.values())

# # Format integers with commas for readability.
# def format_int(number):
#     return f"{number:,}"

# #  Generate and print a table comparing dictionary and postings sizes across index variants.
# # Used GPT to help generate this table formatting code.
# def run_compression_table(index_builds):
#     print("\nDictionary compression impact (relative deltas)")
#     print("-" * 100)
#     print(f"{'Variant':<16} | {'Dict':>10} | {'ΔPrev%':>10} | {'ΔBase%':>10} | "
#           f"{'Postings':>10} | {'ΔPrev%':>10} | {'ΔBase%':>10}")
#     print("-" * 100)

#     # Note: Used GPT to generate this table formatting code
#     base_dict, base_post = describe_index(index_builds[0][1])
#     prev_dict, prev_post = base_dict, base_post
#     for name, index in index_builds:
#         cur_dict, cur_post = describe_index(index)
#         prev_dict_delta = round(100.0 * (cur_dict - prev_dict) / prev_dict, 1) 
#         base_dict_delta = round(100.0 * (cur_dict - base_dict) / base_dict, 1)
#         prev_post_delta = round(100.0 * (cur_post - prev_post) / prev_post, 1)
#         base_post_delta = round(100.0 * (cur_post - base_post) / base_post, 1)

#         print(f"{name:<16} | {format_int(cur_dict):>10} | {prev_dict_delta:+9.1f}% | {base_dict_delta:+9.1f}% | "
#               f"{format_int(cur_post):>10} | {prev_post_delta:+9.1f}% | {base_post_delta:+9.1f}%")

#         prev_dict, prev_post = cur_dict, cur_post
#     print("-" * 100)


#     print("\nSubproject III – lossy dictionary compression table")
#     docs = tokenize_documents(load_reuters_docs(REUTERS_DIR))
#     base_name, base_transform = BASE_VARIANT
#     base_index, pair_count, unique_pairs = naive_build_index(docs, base_transform)
#     base_entry = (base_name, base_transform, base_index)
#     compression_indexes = [base_entry]

#     for name, transform in COMPRESSION_VARIANTS:
#         index, _, _ = naive_build_index(docs, transform)
#         compression_indexes.append((name, transform, index))
#     run_compression_table([(name, index) for name, _, index in compression_indexes])


#     # --- Subproject III (Part 2): query results on compressed indexes -----
#     samples = SAMPLE_SINGLE_QUERIES + SAMPLE_AND_QUERIES
#     print("\nSubproject III (Part 2) – Query results on compressed indexes")
#     for name, transform, index in compression_indexes:
        
#         compare_query_results(name, transform, index, samples)



# Challenge queries
#  -------------------------------------------------------------------------------------
    
    # print("\n=== Challenge Queries (Full Index) ===")
    # naive_index_full, _, _ = naive_build_index(docs, keep_all)
    # spimi_index_full = spimi_build_index(docs, keep_all)

    # CHALLENGE_QUERIES = [["copper"], ["Chrysler"], ["Bundesbank"]]
    # def run_queries(index, label):
    #     print(f"\n=== Challenge Queries on {label} Index ===")
    #     for q in CHALLENGE_QUERIES:
    #         start = time.perf_counter()
    #         if len(q) == 1:
    #             results = search_single(q[0], index)
    #         else:
    #             results = search_and(q, index)
    #         elapsed = (time.perf_counter() - start) * 1000  # ms
    #         print(f"Query {q} -> {len(results)} docs (sample {results[:10]}) | {elapsed:.3f} ms")
    # run_queries(naive_index_full, "Naive")
    # run_queries(spimi_index_full, "SPIMI")

if __name__ == "__main__":
    main()

