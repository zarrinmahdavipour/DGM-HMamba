"""
Class-name embeddings for the DSSPE semantic anchors (manuscript Sec. 3.4).

The manuscript initialises each anchor from the 300-d GloVe vector of the
class name, s_k = Proj(e_k). It does not say which GloVe release was used or
how multi-word / misspelt dataset names ("Corn-notill", "Vinyard untrained")
were turned into one vector. This module makes those choices explicit:

* a class name is lower-cased and split into word tokens (anything that is not
  a letter or a digit separates tokens);
* tokens listed in TOKEN_ALIASES are replaced first (dataset abbreviations and
  misspellings, e.g. "notill" -> "no till", "vinyard" -> "vineyard");
* the class vector is the mean of the GloVe vectors of the tokens that are in
  the vocabulary; tokens that are not are recorded as out-of-vocabulary (OOV).

`scripts/make_class_embeddings.py` builds the JSON file read here from a GloVe
text file. Replace TOKEN_ALIASES or the file itself if the authors' original
choices differ.

File format (JSON):
    {"embedding": "glove.6B.300d", "dim": 300,
     "method": "...", "aliases": {...},
     "classes": {"pavia_u": {"Asphalt": [300 floats], ...}, ...},
     "oov": {"pavia_u": {"Asphalt": []}, ...}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Dataset abbreviations and misspellings mapped to dictionary words. These are
# choices of this re-implementation, not taken from the manuscript.
TOKEN_ALIASES: Dict[str, str] = {
    "notill": "no till",
    "mintill": "minimum till",
    "vinyard": "vineyard",
    "brocoli": "broccoli",
    "4wk": "four weeks",
    "5wk": "five weeks",
    "6wk": "six weeks",
    "7wk": "seven weeks",
}

METHOD = ("lower-cased word tokens (split on non-alphanumerics), TOKEN_ALIASES applied, "
          "mean of the in-vocabulary token vectors")


def tokenize(name: str, aliases: Dict[str, str] = TOKEN_ALIASES) -> List[str]:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    out: List[str] = []
    for t in tokens:
        out.extend(aliases.get(t, t).split())
    return out


def phrase_vector(name: str, vectors: Dict[str, np.ndarray],
                  aliases: Dict[str, str] = TOKEN_ALIASES) -> Tuple[np.ndarray, List[str]]:
    """Mean of the word vectors of the tokens of `name`; returns (vector, oov tokens)."""
    toks = tokenize(name, aliases)
    found = [vectors[t] for t in toks if t in vectors]
    oov = [t for t in toks if t not in vectors]
    if not found:
        raise KeyError(f"no token of class name {name!r} is in the embedding vocabulary (tokens {toks})")
    return np.mean(found, axis=0).astype(np.float32), oov


def needed_tokens(class_names: Dict[str, Sequence[str]], aliases: Dict[str, str] = TOKEN_ALIASES) -> set:
    return {t for names in class_names.values() for n in names for t in tokenize(n, aliases)}


def read_glove(path: str | Path, keep: Iterable[str]) -> Dict[str, np.ndarray]:
    """Read only the rows of `keep` from a GloVe text file (word v1 v2 ...)."""
    keep = set(keep)
    out: Dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in keep:
                out[word] = np.array(rest.split(), dtype=np.float32)
                if len(out) == len(keep):
                    break
    return out


def build_embedding_file(vectors: Dict[str, np.ndarray], class_names: Dict[str, Sequence[str]],
                         embedding: str, out_path: str | Path) -> dict:
    dims = {v.shape[0] for v in vectors.values()}
    if len(dims) != 1:
        raise ValueError(f"inconsistent vector sizes {dims}")
    doc = {"embedding": embedding, "dim": dims.pop(), "method": METHOD, "aliases": TOKEN_ALIASES,
           "classes": {}, "oov": {}}
    for ds, names in class_names.items():
        doc["classes"][ds], doc["oov"][ds] = {}, {}
        for n in names:
            v, oov = phrase_vector(n, vectors)
            doc["classes"][ds][n] = [round(float(x), 6) for x in v]
            doc["oov"][ds][n] = oov
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return doc


def load_class_embeddings(path: str | Path, dataset: str, class_names: Sequence[str]) -> Tuple[np.ndarray, dict]:
    """Rows in the order of `class_names`; returns (array (n, dim), info)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    table = doc["classes"].get(dataset)
    if table is None:
        raise KeyError(f"{path}: no class embeddings for dataset {dataset!r}")
    missing = [n for n in class_names if n not in table]
    if missing:
        raise KeyError(f"{path}: no embedding for classes {missing} of {dataset!r}")
    arr = np.array([table[n] for n in class_names], dtype=np.float32)
    info = {"file": str(path), "embedding": doc.get("embedding"), "dim": doc.get("dim"),
            "method": doc.get("method"), "oov": {n: doc.get("oov", {}).get(dataset, {}).get(n, []) for n in class_names}}
    return arr, info
