"""
Build the class-name embedding file used for the DSSPE semantic anchors.

    python scripts/make_class_embeddings.py --glove glove.6B.300d.txt \
        --out data/class_embeddings/glove.6B.300d.json

The GloVe vectors are not distributed with this repository; download them from
https://nlp.stanford.edu/projects/glove/ (Public Domain Dedication and License).
Only the vectors of the tokens of the class names in dgm_hmamba/data/datasets.py
(CLASS_NAMES) are kept. See dgm_hmamba/data/class_embeddings.py for how a class
name is turned into one vector.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgm_hmamba.data.class_embeddings import build_embedding_file, needed_tokens, read_glove  # noqa: E402
from dgm_hmamba.data.datasets import CLASS_NAMES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glove", required=True, help="GloVe text file (word followed by its vector on each line)")
    p.add_argument("--name", default=None, help="name stored in the file (default: the GloVe file name)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    toks = needed_tokens(CLASS_NAMES)
    vectors = read_glove(args.glove, toks)
    print(f"{len(vectors)} of {len(toks)} tokens found in {args.glove}")
    missing = sorted(toks - set(vectors))
    if missing:
        print("not in the vocabulary:", ", ".join(missing))
    doc = build_embedding_file(vectors, CLASS_NAMES, args.name or Path(args.glove).stem, args.out)
    for ds, table in doc["oov"].items():
        partial = {n: o for n, o in table.items() if o}
        print(f"{ds}: {len(table)} classes" + (f"; names with out-of-vocabulary tokens: {partial}" if partial else ""))
    print(f"wrote {args.out} ({doc['dim']}-d)")


if __name__ == "__main__":
    main()
