"""TF-IDF similarity analysis of extant Greek drama.

Replaces the copy-pasted cells in tfidf-ancient-greek-analysis.ipynb with one
reusable pipeline. Each "experiment" is a named set of documents (authors,
one author's plays, all tragedy, and so on). Every experiment produces:

  results/<experiment>_cosine_mds.png      2D map (MDS on cosine distance)
  results/<experiment>_euclidean_mds.png   2D map (MDS on Euclidean distance)
  results/<experiment>_heatmap.png         pairwise cosine similarity
  results/<experiment>_dendrogram.png      Ward hierarchical clustering
  results/<experiment>_cosine.csv          similarity matrix
  results/<experiment>_top_terms.txt       most distinctive lemmas per document

Input is the folder of cleaned, lemmatized text written by preprocessing:
  <data_dir>/<author>/<play>_vocab.txt                 one file per play
  <data_dir>/<author>/<author>_vocab.txt               one file per author
  <data_dir>/<author>/<speaker>/<play>_<speaker>_vocab.txt   one file per character

Usage:
  python analysis.py --data-dir ~/cltk_data/tmp            # run every experiment
  python analysis.py --data-dir ~/cltk_data/tmp --only authors frogs_characters
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# --------------------------------------------------------------------------
# Corpus definition
# --------------------------------------------------------------------------

AUTHOR_PREFIX = {
    "aeschylus": "aesch",
    "sophocles": "soph",
    "euripides": "eur",
    "aristophanes": "aristoph",
}

# Perseus file code -> readable English title used as the chart label
PLAYS = {
    "aeschylus": {
        "ag": "Agamemnon", "lib": "Libation Bearers", "eum": "Eumenides",
        "pers": "Persians", "seven": "Seven Against Thebes", "supp": "Suppliants (A)",
        "pb": "Prometheus Bound",
    },
    "sophocles": {
        "aj": "Ajax", "ant": "Antigone", "el": "Electra (S)", "oc": "Oedipus at Colonus",
        "ot": "Oedipus Tyrannus", "phil": "Philoctetes", "trach": "Women of Trachis",
    },
    "euripides": {
        "ba": "Bacchae", "el": "Electra (E)", "hec": "Hecuba", "her": "Heracles",
        "ia": "Iphigenia at Aulis", "ion": "Ion", "it": "Iphigenia in Tauris",
        "orest": "Orestes", "phoen": "Phoenician Women", "rh": "Rhesus",
        "supp": "Suppliants (E)", "tro": "Trojan Women",
    },
    "aristophanes": {
        "ach": "Acharnians", "birds": "Birds", "cl": "Clouds", "eccl": "Assemblywomen",
        "frogs": "Frogs", "kn": "Knights", "lys": "Lysistrata", "peace": "Peace",
        "pl": "Wealth", "thes": "Thesmophoriazusae", "wasps": "Wasps",
    },
}

TRAGEDIANS = ["aeschylus", "sophocles", "euripides"]

COMEDIANS = ["aristophanes"]


def play_file(author: str, code: str) -> Path:
    return Path(author) / f"{AUTHOR_PREFIX[author]}.{code}_gk_vocab.txt"


def plays_of(*authors: str) -> dict[str, Path]:
    return {title: play_file(a, code) for a in authors for code, title in PLAYS[a].items()}


def frogs_character(speaker: str) -> Path:
    return Path("aristophanes") / speaker / f"aristoph.frogs_gk_{speaker}_vocab.txt"


EXPERIMENTS: dict[str, dict[str, Path]] = {
    "authors": {a.title(): Path(a) / f"{a}_vocab.txt" for a in PLAYS},
    "aeschylus_plays": plays_of("aeschylus"),
    "sophocles_plays": plays_of("sophocles"),
    "euripides_plays": plays_of("euripides"),
    "aristophanes_plays": plays_of("aristophanes"),
    "all_tragedy": plays_of(*TRAGEDIANS),
    "all_comedy": plays_of(*COMEDIANS),
    "all_drama": plays_of(*PLAYS),
    # Do Aristophanes's caricatures of the poets in Frogs sound like the poets?
    "frogs_characters": {
        **plays_of("aeschylus", "euripides"),
        "'Aeschylus' in Frogs": frogs_character("αἰσχύλος"),
        "'Euripides' in Frogs": frogs_character("εὐριπίδης"),
    },
}

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def load(data_dir: Path, files: dict[str, Path]) -> dict[str, str]:
    """Read each document, skipping (with a warning) any that are missing or empty."""
    docs = {}
    for label, rel in files.items():
        path = data_dir / rel
        if not path.is_file():
            print(f"  warning: missing {path}")
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs[label] = text
        else:
            print(f"  warning: empty {path}")
    return docs


def vectorize(docs: dict[str, str]):
    """TF-IDF on pre-tokenized, lemmatized text.

    Splits on whitespace (the text is already tokenized), keeps one-letter
    words, and uses sublinear term frequency so very frequent lemmas that got
    past the stopword list don't dominate.
    """
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                                 lowercase=False, sublinear_tf=True)
    labels = list(docs)
    matrix = vectorizer.fit_transform(docs[label] for label in labels)
    return labels, matrix, vectorizer


def mds_coords(dist: np.ndarray, seed: int = 42) -> np.ndarray:
    """2D layout whose distances approximate dist. Axes are arbitrary; negative values are normal."""
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        kwargs = {"metric": "precomputed", "init": "classical_mds"}
    else:
        kwargs = {"dissimilarity": "precomputed", "normalized_stress": "auto"}
    return MDS(n_components=2, random_state=seed, **kwargs).fit_transform(dist)


def author_of(label: str) -> str:
    for author, plays in PLAYS.items():
        if label in plays.values() or label.lower() == author:
            return author
    return "other"


COLORS = {"aeschylus": "#1b9e77", "sophocles": "#d95f02", "euripides": "#7570b3",
          "aristophanes": "#e7298a", "other": "#444444"}


def plot_mds(dist, labels, title, out):
    coords = mds_coords(dist)
    fig, ax = plt.subplots(figsize=(10, 8))
    seen = set()
    for (x, y), label in zip(coords, labels):
        author = author_of(label)
        legend = author.title() if author not in seen else None
        seen.add(author)
        ax.scatter(x, y, s=45, color=COLORS[author], label=legend)
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.margins(0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    if len(seen) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_heatmap(sim, labels, title, out):
    size = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dendrogram(dist, labels, title, out):
    # ward() needs the condensed (1-D) form of a square distance matrix
    linkage = ward(squareform(dist, checks=False))
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.3)))
    dendrogram(linkage, orientation="right", labels=labels, ax=ax, leaf_font_size=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_experiment(name: str, files: dict[str, Path], data_dir: Path, out_dir: Path) -> None:
    print(f"{name}:")
    docs = load(data_dir, files)
    if len(docs) < 3:
        print(f"  skipped: need at least 3 documents, found {len(docs)}")
        return
    labels, matrix, vectorizer = vectorize(docs)
    print(f"  {len(labels)} documents, {matrix.shape[1]} distinct lemmas")

    sim = cosine_similarity(matrix)
    cos_dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(cos_dist, 0)
    euc_dist = euclidean_distances(matrix)

    nice = name.replace("_", " ").title()
    plot_mds(cos_dist, labels, f"{nice}: TF-IDF cosine distance (MDS)", out_dir / f"{name}_cosine_mds.png")
    plot_mds(euc_dist, labels, f"{nice}: TF-IDF Euclidean distance (MDS)", out_dir / f"{name}_euclidean_mds.png")
    plot_heatmap(sim, labels, f"{nice}: cosine similarity", out_dir / f"{name}_heatmap.png")
    plot_dendrogram(cos_dist, labels, f"{nice}: Ward clustering", out_dir / f"{name}_dendrogram.png")

    with (out_dir / f"{name}_cosine.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, sim):
            writer.writerow([label] + [f"{v:.4f}" for v in row])

    vocab = vectorizer.get_feature_names_out()
    with (out_dir / f"{name}_top_terms.txt").open("w", encoding="utf-8") as f:
        for i, label in enumerate(labels):
            row = matrix[i].toarray().ravel()
            top = [vocab[j] for j in row.argsort()[::-1][:15] if row[j] > 0]
            f.write(f"{label}: {', '.join(top)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "cltk_data" / "tmp",
                        help="Folder written by preprocessing (default: ~/cltk_data/tmp)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="Run only these experiments")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or EXPERIMENTS:
        run_experiment(name, EXPERIMENTS[name], args.data_dir.expanduser(), args.out_dir)
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
