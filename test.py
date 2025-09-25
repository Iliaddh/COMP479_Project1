from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import defaultdict
import os
# Download resources (only once needed)
nltk.download('stopwords')
nltk.download('punkt')


# Subproject 1: Inverted Index Construction
stop_words = set(stopwords.words('english'))

pairs = []


for file in os.listdir("reuters21578"):
    if file.endswith(".sgm"):
        with open(os.path.join("reuters21578", file), "r", encoding="latin-1") as f:
            data = f.read()
            soup = BeautifulSoup(data, "html.parser")

            for doc in soup.find_all("reuters"):
                doc_id = int(doc['newid'])
                text_tag = doc.find("text")
                if text_tag:
                    text = text_tag.get_text(" ", strip=True)
                    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha() and t not in stop_words]

                    for token in set(tokens):
                        pairs.append((token, doc_id))

# Sort & deduplicate
pairs = sorted(set(pairs))
# print("Total unique (term, docID) pairs:", len(pairs))
# print("First 20 pairs:", pairs[:20])

# Build index
index = defaultdict(list)
for term, doc_id in pairs:
    index[term].append(doc_id)

# print("Docs containing 'oil':", index.get("british", []))
# print("Docs containing 'market':", index.get("market", []))

 



# Subproject 2: Query Processing


def search_single(term):
    return index.get(term.lower(), [])


def intersect(p1, p2):
    answer = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return answer

def search_and_terms(terms, index):
    if not terms:
        return []
    
    # Start with postings list of the first term
    result = search_single(terms[0])
    
    # Iteratively intersect with the rest
    for term in terms[1:]:
        result = intersect(result, search_single(term))
    
    return result

print(search_and_terms(["oil", "market", "british"], index))


# Subproject 3:  Dictionary compression table