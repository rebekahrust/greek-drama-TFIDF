# Greek Tragedies TF-IDF

**Computational stylometry of extant ancient Greek drama: comparing the vocabulary of Aeschylus, Sophocles, Euripides, and Aristophanes at the author, play, and character level.**

![Python](https://img.shields.io/badge/python-3.11-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

<!-- TODO: Replace with your best chart, e.g. the play-similarity plot -->
![Play similarity plot](results/play_similarity_cosine.png)

## Research question

Do word-use patterns in Greek drama follow **genre** (tragedy vs. comedy), **author**, or something else, such as **generation**? And can vocabulary alone pick out plays whose authorship scholars already dispute?

## Key findings

> These results are provisional and exploratory (see [Limitations](#limitations)).
<!-- TODO: These findings come from the 2016 raw-count analysis. Re-run src/analysis.py and update anything that changes with TF-IDF. -->



- **Generation may matter more than genre.** By cosine similarity, Sophocles, Euripides, and Aristophanes (comedy) are closer to each other than any of them is to Aeschylus, who belongs to an older generation and is known for coining words.
- **Disputed plays stand out.** *Prometheus Bound* falls well away from Aeschylus's other plays, and *Rhesus* is an outlier among Euripides's plays. Both results agree with the scholarly doubts about who wrote them.
- **The Oresteia clusters together.** *Agamemnon*, *Libation Bearers*, and *Eumenides*, the only surviving trilogy, sit close together, which is a useful check that the method works.
- **Aristophanes's caricatures don't match the real authors.** The characters "Aeschylus" and "Euripides" in *Frogs* show no strong similarity to the real playwrights' vocabulary.

## Quickstart

```bash
git clone https://github.com/rebekahrust/greek-tragedies-TFIDF.git
cd greek-tragedies-TFIDF
pip install -r requirements.txt
python src/preprocess.py                              # download and clean the corpus
python src/analysis.py --data-dir ~/cltk_data/tmp     # TF-IDF, similarity, charts
```

Charts, similarity matrices (CSV), and each document's most distinctive words are written to `results/`.

## Method

| Stage | What happens |
|---|---|
| **Input** | Download the Greek corpus (Perseus TEI XML) with the [Classical Language Toolkit (CLTK)](https://github.com/cltk/cltk). |
| **Pre-processing** | Parse the XML with Python's DOM parser, split each play into speaker lines (keeping line numbers), convert Beta Code to Unicode, lemmatize, strip punctuation, and remove stopwords. Output is organized as `author / play / character / lines`. |
| **Processing** | Build TF-IDF vectors with scikit-learn's `TfidfVectorizer`. TF-IDF lowers the weight of words that are common across the whole corpus, which makes up for stopwords (pronouns, forms of *εἰμί*) that get past an imperfect Greek lemmatizer. |
| **Comparison** | Measure pairwise **cosine similarity**, the primary measure because it adjusts for document length, and **Euclidean distance**, for comparison. |
| **Output** | Project the distance matrices into 2D with multidimensional scaling (MDS), cluster them with Ward's method, and plot the results at three levels: author, play, and character. MDS axes have no inherent meaning, and negative coordinates are normal; only relative position matters. |

## Repository structure

```
├── archive/2016/       # original 2016 notebooks and code, preserved as written
├── src/
│   ├── preprocess.py   # XML parsing, Beta Code → Unicode, lemmatizing, stopwords
│   └── analysis.py     # TF-IDF, cosine/Euclidean similarity, MDS, clustering
├── results/            # generated charts and similarity matrices
├── requirements.txt
└── README.md
```

## Limitations

- **The evidence is fragmentary.** Surviving Greek drama is only a sliver of what was produced, and some surviving plays (several by Euripides, for example) aren't available in the open-source Perseus corpus.
- **Lemmatization is imperfect.** The Greek lemmatizer misses some inflected forms, and that noise can affect the results.
- **Similarity is not proof.** Distance measures are a way to question the data, not a basis for conclusions. As Drucker warns, visualizations can mislead humanists. These findings suggest patterns; they don't establish them.

## Applications beyond classics

The same pipeline (structured parsing → normalization → TF-IDF vectorization → similarity scoring) is used in authorship attribution, plagiarism detection, document deduplication, and security text analysis such as phishing classification and log clustering.

## Background and credits

Originally built in December 2016 as a digital humanities research project. The original version used raw word counts. The 2026 update adds true TF-IDF weighting, fixes clustering and plotting bugs, and replaces the notebooks with a reproducible pipeline. The original is preserved in [`archive/2016/`](archive/2016/) and as the [`v2016` release](https://github.com/rebekahrust/greek-tragedies-TFIDF/releases/tag/v2016).

- [Classical Language Toolkit](https://github.com/cltk/cltk): Kyle P. Johnson, Patrick J. Burns, et al.
- Allen Riddell's tutorials on text vectorization
- [Perseus Digital Library](http://www.perseus.tufts.edu/) for the source texts

### References

- Drucker, Johanna. "Humanities Approaches to Graphical Display." *Digital Humanities Quarterly* 5.1 (2011).
<!-- TODO: add your other footnote sources (6, 7) -->

## License

MIT
