# ====== Subproject 3 — Simple refactor: one transform per variant, reused for queries ======
import os, re, nltk
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

REUTERS_DIR = "reuters21578"   # <-- set your folder

# ------------------------------
# Load Reuters docs (SGML -> (doc_id, text))
# ------------------------------
def load_reuters_docs(reuters_dir):
    docs = []
    for file in os.listdir(reuters_dir):
        if not file.endswith(".sgm"):
            continue
        with open(os.path.join(reuters_dir, file), "r", encoding="latin-1") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for r in soup.find_all("reuters"):
            doc_id = int(r["newid"])
            t = r.find("text")
            if t:
                docs.append((doc_id, t.get_text(" ", strip=True)))
    return docs

# ------------------------------
# Project-2 query processor (index-agnostic)
# ------------------------------
def search_single(term, index):
    return index.get(term, [])

def intersect(p1, p2):
    i=j=0; out=[]
    while i<len(p1) and j<len(p2):
        if p1[i]==p2[j]:
            out.append(p1[i]); i+=1; j+=1
        elif p1[i]<p2[j]: i+=1
        else: j+=1
    return out

def search_and_terms(terms, index):
    if not terms: return []
    res = search_single(terms[0], index)
    for t in terms[1:]:
        res = intersect(res, search_single(t, index))
        if not res: break
    return res

# ------------------------------
# Variant token transforms (reused for indexing AND queries)
# ------------------------------
EN_STOP  = set(stopwords.words("english"))
TOP30    = {"the","of","and","to","a","in","that","is","was","he","for","it","with","as","his","on","be","at","by","i","this","had","not","are","but","from","or","have","an","they"}
STOP150  = set(sorted(EN_STOP)[:150])
PS       = PorterStemmer()

def T_unfiltered(toks):
    # Keep everything from word_tokenize (case, numbers, punctuation)
    return toks

def T_no_numbers(toks):
    # Drop tokens containing any digit (e.g., '1987', '3.5', 'A320', '1987-05-13')
    return [t for t in toks if not any(ch.isdigit() for ch in t)]

def T_casefold(toks):
    return [t.lower() for t in toks]

def T_stop30(toks):
    toks = [t.lower() for t in toks]
    return [t for t in toks if t not in TOP30]

def T_stop150(toks):
    toks = [t.lower() for t in toks]
    return [t for t in toks if t not in STOP150]

def T_stemmed(toks):
    toks = [t.lower() for t in toks]
    toks = [t for t in toks if t not in EN_STOP]
    # Stem only tokens that contain letters; keep numbers/punct unchanged
    return [PS.stem(t) if any(ch.isalpha() for ch in t) else t for t in toks]

# ------------------------------
# Build index given a transform
# ------------------------------
def build_index(docs, transform):
    index = defaultdict(list)
    for doc_id, text in docs:
        toks = word_tokenize(text)   # NLTK only
        toks = transform(toks)       # apply variant rules
        for term in set(toks):       # binary postings
            index[term].append(doc_id)
    for term in index:
        index[term].sort()
    return index

def measure(index):
    return len(index), sum(len(pl) for pl in index.values())

def format_int(n): return f"{n:,}"

# ------------------------------
# Run all variants, print rounded table, then log query results
# ------------------------------
def run_all():
    docs = load_reuters_docs(REUTERS_DIR)

    # (name, transform)
    variants = [
        ("UNFILTERED",     T_unfiltered),
        ("NO NUMBERS",     T_no_numbers),
        ("CASE FOLDING",   T_casefold),
        ("30 STOP WORDS",  T_stop30),
        ("150 STOP WORDS", T_stop150),
        ("STEMMING",       T_stemmed),
    ]

    built = []
    for name, T in variants:
        print(f"Building {name} ...")
        idx = build_index(docs, T)
        D, P = measure(idx)
        built.append((name, T, idx, D, P))

    # Table (rounded deltas to 1 decimal)
    base_D, base_P = built[0][3], built[0][4]
    prev_D, prev_P = None, None

    print("\n" + "-"*100)
    print(f"{'Variant':<16} | {'Dict':>10} | {'ΔDict Prev':>10} | {'ΔDict Base':>10} | "
          f"{'Postings':>10} | {'ΔPost Prev':>10} | {'ΔPost Base':>10}")
    print("-"*100)

    for i, (name, T, idx, D, P) in enumerate(built):
        if i == 0:
            d_prev = d_base = p_prev = p_base = 0.0
        else:
            d_prev = round(100.0 * (D - prev_D) / prev_D, 1)
            p_prev = round(100.0 * (P - prev_P) / prev_P, 1)
            d_base = round(100.0 * (D - base_D) / base_D, 1)
            p_base = round(100.0 * (P - base_P) / base_P, 1)

        print(f"{name:<16} | {format_int(D):>10} | {d_prev:+9.1f}% | {d_base:+9.1f}% | "
              f"{format_int(P):>10} | {p_prev:+9.1f}% | {p_base:+9.1f}%")

        prev_D, prev_P = D, P

    print("-"*100)

    # Log query results (reuse SAME transform for queries)
    single_queries = [["oil"], ["market"], ["british"]]
    and_queries    = [["oil","market"], ["gold","prices"], ["u.s.","trade"]]

    for name, T, idx, D, P in built:
        print(f"\n[{name}]")
        # Singles
        for q in single_queries:
            # apply same transform to query terms
            q_transformed = T(q)
            if not q_transformed:
                print(f" single {q} -> all terms filtered by preprocessing")
                continue
            hits = search_single(q_transformed[0], idx)
            print(f" single {q} -> {len(hits)} docs  sample: {hits[:10]}")
        # ANDs
        for q in and_queries:
            q_transformed = T(q)
            if not q_transformed:
                print(f" AND  {q} -> all terms filtered by preprocessing")
                continue
            hits = search_and_terms(q_transformed, idx)
            print(f" AND  {q} -> {len(hits)} docs  sample: {hits[:10]}")

# ---- go! ----
run_all()
# ====== End ======
