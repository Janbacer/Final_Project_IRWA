"""
Inverted index and TF-IDF ranking, adapted from Practice Session 1
(01_Model_Tf_IDF_Index): create_index, search, create_index_tfidf,
rank_documents and search_tf_idf.

Differences with the lab version:
* Documents are identified by their product ``pid`` (a string) instead of a
  numeric page id.
* The functions receive documents that are *already pre-processed*
  (``{pid: [terms]}``, e.g. the ``all_terms`` column of
  ``data/processed_corpus.pkl`` built in Part 1), so the lab's ``build_terms``
  is replaced by our Part 1 pipeline (``irwa_preprocessing.preprocess_text``).
  Queries must be cleaned with that same function before searching.
"""
import collections
import math
from array import array
from collections import defaultdict

import numpy as np
from numpy import linalg as la


def create_index(docs_terms):
    """
    Build an inverted index with term positions (lab Figure 2).

    Argument:
    docs_terms -- dict {pid: list of pre-processed terms}

    Returns:
    index -- dict {term: [[pid, array of positions], ...]}
    """
    index = defaultdict(list)
    for pid, terms in docs_terms.items():
        current_page_index = {}
        for position, term in enumerate(terms):
            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [pid, array('I', [position])]
        for term, posting in current_page_index.items():
            index[term].append(posting)
    return index


def search(query_terms, index):
    """
    Return the pids of the documents that contain ANY of the query terms (no ranking).

    Argument:
    query_terms -- list of pre-processed query terms
    index -- inverted index from create_index / create_index_tfidf
    """
    docs = set()
    for term in query_terms:
        docs.update(posting[0] for posting in index.get(term, []))
    return list(docs)


def create_index_tfidf(docs_terms):
    """
    Build the inverted index and compute normalised TF, DF and IDF (lab formulas):
    tf = count / ||D|| (Euclidean norm of the document's term counts), idf = log(N / df).

    Argument:
    docs_terms -- dict {pid: list of pre-processed terms}

    Returns:
    index -- dict {term: [[pid, array of positions], ...]}
    tf -- dict {term: [tf of the term in each posting, same order as index[term]]}
    df -- dict {term: number of documents containing the term}
    idf -- dict {term: inverse document frequency}
    """
    num_documents = len(docs_terms)
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)

    for pid, terms in docs_terms.items():
        current_page_index = {}
        for position, term in enumerate(terms):
            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [pid, array('I', [position])]

        norm = math.sqrt(sum(len(posting[1]) ** 2 for posting in current_page_index.values()))
        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4) if norm else 0.0)
            df[term] += 1
            index[term].append(posting)

    for term, document_frequency in df.items():
        idf[term] = np.round(np.log(float(num_documents / document_frequency)), 4)

    return index, tf, df, idf


def rank_documents(terms, docs, index, idf, tf):
    """
    Rank the matching documents by cosine similarity between TF-IDF vectors.

    Argument:
    terms -- list of pre-processed query terms
    docs -- pids of the documents to rank
    index, idf, tf -- output of create_index_tfidf

    Returns:
    list of pids sorted by decreasing score
    """
    docs = set(docs)
    doc_vectors = defaultdict(lambda: [0.0] * len(terms))
    query_vector = [0.0] * len(terms)
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))
    if query_norm == 0:
        return []

    for term_index, term in enumerate(terms):
        if term not in index:
            continue
        query_vector[term_index] = (query_terms_count[term] / query_norm) * idf[term]
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][term_index] = tf[term][doc_index] * idf[term]

    doc_scores = [[np.dot(vector, query_vector), doc] for doc, vector in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    return [doc for _, doc in doc_scores]


def search_tf_idf(query_terms, index, tf, idf):
    """
    Return the documents containing any query term, ordered by TF-IDF relevance.

    Argument:
    query_terms -- list of pre-processed query terms (use preprocess_text on the raw query)
    index, tf, idf -- output of create_index_tfidf
    """
    docs = search(query_terms, index)
    return rank_documents(query_terms, docs, index, idf, tf)


def search_in_corpus(query):
    """Template placeholder for the search engine (to be implemented in Parts 2-4)."""
    # Template placeholder: the search engine will call search_tf_idf here (Parts 2-4).
    # 1. create create_tfidf_index

    # 2. apply ranking
    return ""
