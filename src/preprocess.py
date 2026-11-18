
"""Preprocess the Perseus Greek drama corpus for TF-IDF analysis.

Steps for each play:
  1. Parse the Perseus TEI XML and split it into speeches (<sp>), keeping line numbers.
  2. Convert Beta Code to Unicode Greek.
  3. Lemmatize, remove stopwords, and strip punctuation.
  4. Write the cleaned text at four levels, in the layout analysis.py expects:

     <out_dir>/<author>/<author>_vocab.txt                        whole author
     <out_dir>/<author>/<play>_vocab.txt                          whole play
     <out_dir>/<author>/<speaker>/<play>_<speaker>_vocab.txt      one character in one play
     <out_dir>/<author>/<speaker>/<play>_<speaker>_<start>_<end>.txt   one speech

This is the 2016 main.py and processing notebook combined, with bugs fixed.
It deliberately keeps the pre-1.0 CLTK API so results match the original
project (see requirements.txt).

Usage:
  python preprocess.py                       # uses ~/cltk_data
  python preprocess.py --cltk-data /path/to/cltk_data
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from xml.dom.minidom import parse

import bleach
from cltk.corpus.greek.beta_to_unicode import Replacer
from cltk.corpus.utils.formatter import phi5_plaintext_cleanup, tlg_plaintext_cleanup
from cltk.corpus.utils.importer import CorpusImporter
from cltk.stem.lemma import LemmaReplacer
from cltk.stop.greek.stops import STOPS_LIST
from nltk.tokenize.punkt import PunktLanguageVars

log = logging.getLogger("preprocess")

# Plays analyzed, by author (Perseus file names). Some Euripides plays are not
# in the open-source Perseus corpus and so are not included.
CORPUS = {
    "aeschylus": ["aesch.ag_gk.xml", "aesch.eum_gk.xml", "aesch.lib_gk.xml", "aesch.pers_gk.xml",
                  "aesch.seven_gk.xml", "aesch.supp_gk.xml", "aesch.pb_gk.xml"],
    "sophocles": ["soph.aj_gk.xml", "soph.ant_gk.xml", "soph.oc_gk.xml", "soph.ot_gk.xml",
                  "soph.phil_gk.xml", "soph.trach_gk.xml", "soph.el_gk.xml"],
    "euripides": ["eur.ba_gk.xml", "eur.el_gk.xml", "eur.hec_gk.xml", "eur.her_gk.xml",
                  "eur.ia_gk.xml", "eur.ion_gk.xml", "eur.it_gk.xml", "eur.phoen_gk.xml",
                  "eur.rh_gk.xml", "eur.supp_gk.xml", "eur.tro_gk.xml", "eur.orest_gk.xml"],
    "aristophanes": ["aristoph.ach_gk.xml", "aristoph.birds_gk.xml", "aristoph.cl_gk.xml",
                     "aristoph.eccl_gk.xml", "aristoph.frogs_gk.xml", "aristoph.kn_gk.xml",
                     "aristoph.lys_gk.xml", "aristoph.peace_gk.xml", "aristoph.pl_gk.xml",
                     "aristoph.thes_gk.xml", "aristoph.wasps_gk.xml"],
}

# Created once and reused; the 2016 version rebuilt these for every speech.
_replacer = Replacer()
_lemmatizer = None  # loaded lazily, after the models corpus is downloaded
_punkt = PunktLanguageVars()
_stops = set(STOPS_LIST)


@dataclass
class Speech:
    speaker: str
    start_line: int
    end_line: int
    text: str


# ---------------------------------------------------------------------------
# Corpus download
# ---------------------------------------------------------------------------

def ensure_corpus(cltk_data: Path) -> Path:
    """Download the Perseus texts and the Greek lemmatizer models if missing."""
    perseus = cltk_data / "greek" / "text" / "greek_text_perseus"
    importer = CorpusImporter("greek")
    if not perseus.is_dir():
        log.info("Downloading greek_text_perseus (one time)...")
        importer.import_corpus("greek_text_perseus")
    if not (cltk_data / "greek" / "model" / "greek_models_cltk").is_dir():
        log.info("Downloading greek_models_cltk for the lemmatizer (one time)...")
        importer.import_corpus("greek_models_cltk")
    return perseus


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(beta_code: str) -> str:
    """Beta Code -> Unicode -> lemmas, with stopwords and punctuation removed."""
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = LemmaReplacer("greek")

    text = bleach.clean(beta_code, strip=True).upper()
    # Periods and semicolons left in place stop some words from lemmatizing.
    text = text.replace(".", " ").replace(";", " ")
    unicode_text = _replacer.beta_code(text)

    lemmatized = _lemmatizer.lemmatize(unicode_text.lower(), return_string=True)
    words = [w for w in _punkt.word_tokenize(lemmatized) if w not in _stops]

    # Strip any punctuation that survived lemmatization.
    cleaned = phi5_plaintext_cleanup(" ".join(words), rm_punctuation=True, rm_periods=True)
    cleaned = tlg_plaintext_cleanup(cleaned, rm_punctuation=True, rm_periods=False)
    cleaned = cleaned.replace("·", " ")
    return " ".join(cleaned.split())  # collapse repeated whitespace


def safe_name(name: str) -> str:
    """Make a speaker name safe to use as a folder or file name."""
    name = re.sub(r'[\\/:*?"<>|]', "", name).strip()
    return name or "unknown"


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
# Perseus format:
#   <sp>
#     <speaker>*)aqh/na</speaker>
#     <l>a)ei\ me/n, w)= pai= *larti/ou, de/dorka/ se</l>
#     <l n="95">malakai=s a)do/loisi parhgori/ais,</l>
#   </sp>

def get_text(nodes) -> str:
    return "".join(n.data for n in nodes if n.nodeType == n.TEXT_NODE)


def parse_play(xml_path: Path) -> list[Speech]:
    """Split a play into speeches, tracking line numbers.

    Only every fifth <l> carries an explicit n="..." number, so we count the
    lines in between. The counter is local to each play (the 2016 version used
    globals that had to be reset by hand).
    """
    dom = parse(str(xml_path))
    line_number = 0
    speeches = []

    for sp in dom.getElementsByTagName("sp"):
        start_line = line_number + 1

        speaker_tags = sp.getElementsByTagName("speaker")
        if speaker_tags:
            # Same cleaning as the 2016 version, so speaker names (and the
            # folder names analysis.py looks for) stay identical.
            speaker = safe_name(clean_text(get_text(speaker_tags[0].childNodes)))
        else:
            speaker = "unknown"
            log.debug("Speech with no <speaker> tag in %s", xml_path.name)

        chunk = []
        for line in sp.getElementsByTagName("l"):
            digits = re.sub(r"[^0-9]", "", line.getAttribute("n"))
            line_number = int(digits) if digits else line_number + 1
            chunk.append(get_text(line.childNodes))

        text = clean_text(" ".join(chunk))
        speeches.append(Speech(speaker, start_line, line_number, text))

    return speeches


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_play(author_dir: Path, play: str, speeches: list[Speech]) -> str:
    """Write speech, character, and play files. Returns the play's full text."""
    by_speaker: dict[str, list[str]] = {}
    speech_files = []

    for s in speeches:
        speaker_dir = author_dir / s.speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        path = speaker_dir / f"{play}_{s.speaker}_{s.start_line}_{s.end_line}.txt"
        path.write_text(s.text, encoding="utf-8")
        speech_files.append(str(path))
        by_speaker.setdefault(s.speaker, []).append(s.text)

    for speaker, texts in by_speaker.items():
        vocab = author_dir / speaker / f"{play}_{speaker}_vocab.txt"
        vocab.write_text(" ".join(texts), encoding="utf-8")

    (author_dir / f"{play}_files.txt").write_text("\n".join(speech_files), encoding="utf-8")

    play_text = " ".join(s.text for s in speeches)
    (author_dir / f"{play}_vocab.txt").write_text(play_text, encoding="utf-8")
    return play_text


def run(cltk_data: Path, out_dir: Path) -> None:
    perseus = ensure_corpus(cltk_data)
    for author, files in CORPUS.items():
        log.info("Author: %s", author)
        author_dir = out_dir / author
        author_dir.mkdir(parents=True, exist_ok=True)
        author_texts = []

        for filename in files:
            xml_path = perseus / author / "opensource" / filename
            if not xml_path.is_file():
                log.warning("  Missing %s, skipping", xml_path)
                continue
            play = Path(filename).stem  # e.g. aesch.ag_gk
            speeches = parse_play(xml_path)
            author_texts.append(write_play(author_dir, play, speeches))
            log.info("  %s: %d speeches", play, len(speeches))

        (author_dir / f"{author}_vocab.txt").write_text(" ".join(author_texts), encoding="utf-8")

    log.info("Done. Cleaned text is in %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cltk-data", type=Path, default=Path.home() / "cltk_data",
                        help="CLTK data folder (default: ~/cltk_data)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write cleaned text (default: <cltk-data>/tmp)")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    cltk_data = args.cltk_data.expanduser()
    run(cltk_data, (args.out_dir or cltk_data / "tmp").expanduser())


if __name__ == "__main__":
    main()
