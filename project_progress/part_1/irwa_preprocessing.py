"""
IRWA Final Project - Part 1: text pre-processing.

The same functions are used for the products and, in Part 2, for the queries:
a query and a document must be cleaned the same way so that their terms match
in the inverted index.

Main functions
--------------
- ``preprocess_text(text)``      -> list of stemmed terms (titles, descriptions, queries)
- ``preprocess_document(doc)``   -> dict with every processed field of one product
- ``preprocess_corpus(docs)``    -> pandas DataFrame with one row per product
"""

import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()  # same stemmer as in the lab sessions
STOP_WORDS = set(stopwords.words("english"))

# Fields that must be returned for every retrieved document (statement, point 2).
OUTPUT_FIELDS = [
    "pid", "title", "description", "brand", "category", "sub_category",
    "product_details", "seller", "out_of_stock", "selling_price", "discount",
    "actual_price", "average_rating", "url",
]

# ---------------------------------------------------------------------------
# 1. Text pipeline
# ---------------------------------------------------------------------------
# "T-shirt" is written in 8 different ways in the data (t-shirt, t shirt,
# tshirt, tee...). Removing the hyphen of "t-shirt" would give "t" + "shirt",
# and every T-shirt would match the query "shirt". So a few fashion words are
# written in a single way before removing punctuation.
FASHION_WORDS = [
    (r"\bt[\s\-]?shirts?\b|\btees?\b", " tshirt "),
    (r"\bsweat[\s\-]?shirts?\b", " sweatshirt "),
    (r"\bpoly[\s\-]?cotton\b", " poly cotton "),  # so that "cotton" matches polycotton
    (r"\bslim[\s\-]?fit\b", " slim fit "),        # "Slimfit" in product_details
]


def normalize_fashion_words(text):
    """Write t-shirt, sweatshirt, polycotton and slim fit in a single way (expects lowercase text)."""
    for pattern, replacement in FASHION_WORDS:
        text = re.sub(pattern, replacement, text)
    return text


def remove_punctuation(text):
    """Replace every punctuation mark by a space ('look.Designed' -> 'look Designed')."""
    return re.sub(r"[^\w\s]|_", " ", text)


def tokenize(text):
    """Split the text into tokens on white space, as in the lab sessions."""
    return text.split()


def remove_stopwords(tokens):
    """Remove NLTK English stop words ('the', 'for', 'with'...). 'men'/'women' are not stop words."""
    return [t for t in tokens if t not in STOP_WORDS]


def remove_numbers(tokens):
    """Remove pure numbers and one-letter tokens ('100', '2', 's'); keep '3xl', '4th'."""
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


def stem(tokens):
    """Reduce each token to its stem with the Porter stemmer ('jeans' -> 'jean')."""
    return [STEMMER.stem(t) for t in tokens]


def preprocess_text(text, do_stem=True):
    """
    Full pipeline for a title, a description or a query:
    lowercase -> fashion words -> remove punctuation -> tokenize (split)
    -> remove stop words -> remove numbers / 1-letter tokens -> stem.
    """
    if not text:
        return []
    text = text.lower()
    text = normalize_fashion_words(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_numbers(tokens)
    return stem(tokens) if do_stem else tokens


# ---------------------------------------------------------------------------
# 2. Categorical fields
# ---------------------------------------------------------------------------
# product_details keys whose values are codes or boilerplate, not searchable text.
DETAIL_KEYS_EXCLUDED = {"Style Code", "Fabric Care", "Sales Package", "Country of Origin"}


def flatten_details(details):
    """[{'Fabric': 'Cotton'}, {'Color': 'Blue'}] -> {'Fabric': 'Cotton', 'Color': 'Blue'}."""
    flat = {}
    for entry in details or []:
        for key, value in entry.items():
            flat[key.strip()] = str(value).strip()
    return flat


def details_to_text(flat):
    """Text of product_details: the values only (keys like 'Fabric' are in every product), without Yes/No values."""
    return " ".join(value for key, value in flat.items()
                    if key not in DETAIL_KEYS_EXCLUDED and value.lower() not in ("yes", "no"))


def recover_brand(brand, url):
    """
    Recover the full brand name from the product URL.

    The brand field is cut off ("Pu" = Puma, "REEB" = Reebok), but the URL starts
    with the full name: flipkart.com/free-authority-full-sleeve-.../p/...
    We take the shortest group of URL words that starts with the cut brand.
    If nothing matches, the (lower-cased) original brand is returned.
    """
    brand = (brand or "").strip().lower()
    target = re.sub(r"[^a-z0-9]", "", brand)
    match = re.search(r"flipkart\.com/([^/]+)/p/", url or "")
    if not target or not match:
        return brand
    words = match.group(1).split("-")
    joined = ""
    for i, word in enumerate(words):
        joined += word
        if len(joined) >= len(target):
            return " ".join(words[: i + 1]) if joined.startswith(target) else brand
    return brand


# ---------------------------------------------------------------------------
# 3. Numeric fields (NOT indexed as text)
# ---------------------------------------------------------------------------
def parse_price(value):
    """'2,999' -> 2999.0 ; '' -> None."""
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_discount(value):
    """'69% off' -> 69.0 ; '' -> None."""
    match = re.search(r"\d+", str(value or ""))
    return float(match.group()) if match else None


def parse_rating(value):
    """'3.9' -> 3.9 ; '' -> None (a missing rating is NOT a rating of 0)."""
    try:
        return float(str(value).strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 4. Whole document / corpus
# ---------------------------------------------------------------------------
def preprocess_document(doc):
    """Return the original output fields of one product plus its processed fields."""
    flat = flatten_details(doc.get("product_details"))
    brand_full = recover_brand(doc.get("brand"), doc.get("url"))

    title_terms = preprocess_text(doc.get("title"))
    desc_terms = preprocess_text(doc.get("description"))
    details_terms = preprocess_text(details_to_text(flat))
    brand_terms = preprocess_text(brand_full)
    subcat_terms = preprocess_text(doc.get("sub_category"))

    processed = {field: doc.get(field) for field in OUTPUT_FIELDS}  # unchanged, for the output
    processed.update({
        # text fields (inverted index)
        "title_terms": title_terms,
        "description_terms": desc_terms,
        "details_terms": details_terms,
        "brand_terms": brand_terms,
        "sub_category_terms": subcat_terms,
        "all_terms": title_terms + details_terms + subcat_terms + brand_terms + desc_terms,
        # extra structured information
        "brand_full": brand_full,
        "details": flat,
        # numeric fields (filters and ranking signals)
        "selling_price_num": parse_price(doc.get("selling_price")),
        "actual_price_num": parse_price(doc.get("actual_price")),
        "discount_pct": parse_discount(doc.get("discount")),
        "average_rating_num": parse_rating(doc.get("average_rating")),
        "in_stock": not doc.get("out_of_stock"),
    })
    return processed


def preprocess_corpus(docs):
    """Process every product and return a DataFrame indexed by pid (one row per product)."""
    return pd.DataFrame([preprocess_document(d) for d in docs]).set_index("pid", drop=False)
