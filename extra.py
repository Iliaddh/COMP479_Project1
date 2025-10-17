
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

# # drop_numbers: Remove tokens containing on
def drop_numbers(tokens):
    return [t for t in tokens if not t.isdigit()]

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
