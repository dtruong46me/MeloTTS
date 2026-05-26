"""Preprocess raw metadata into cleaned phoneme lists and train/val splits.

This script reads a pipe-delimited metadata file, runs text normalisation and
grapheme-to-phoneme conversion (G2P) via
:func:`melo.text.cleaner.clean_text_bert`, writes a ``.cleaned`` file with
phoneme/tone/word2ph columns, then splits the result into ``train.list`` and
``val.list`` files and emits an updated ``config.json`` that includes the
speaker-ID map, number of languages, number of tones, and symbol vocabulary.

Usage::

    python -m melo.preprocess_text \\
        --metadata data/example/metadata.list \\
        --config_path configs/config.json

"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import json
import os
import sys
from collections import defaultdict
from random import shuffle
from typing import Optional

# ---------------------------------------------------------------------------
# third-party
# ---------------------------------------------------------------------------
import click
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: scripts/ is now under melo/.  Inserting the project root
# (3 levels up) onto sys.path makes the ``melo`` package importable when
# this script is run directly (e.g. ``python melo/scripts/preprocess_text.py``).
# ---------------------------------------------------------------------------
__root__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __root__ not in sys.path:
    sys.path.insert(0, __root__)

# ---------------------------------------------------------------------------
# local – must come *after* the sys.path fix above
# ---------------------------------------------------------------------------
from melo.text.cleaner import clean_text_bert  # noqa: E402
from melo.text.symbols import num_languages, num_tones, symbols  # noqa: E402


@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the raw pipe-delimited metadata list file.",
)
@click.option(
    "--cleaned-path",
    default=None,
    help="Output path for the cleaned metadata file.  Defaults to "
    "<metadata>.cleaned.",
)
@click.option(
    "--train-path",
    default=None,
    help="Output path for the training split list.  Defaults to "
    "<metadata_dir>/train.list.",
)
@click.option(
    "--val-path",
    default=None,
    help="Output path for the validation split list.  Defaults to "
    "<metadata_dir>/val.list.",
)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Base config JSON file.  An updated copy is written alongside the "
    "metadata file.",
)
@click.option(
    "--val-per-spk",
    default=4,
    help="Maximum number of validation utterances per speaker.",
)
@click.option(
    "--max-val-total",
    default=8,
    help="Hard cap on the total number of validation utterances across all "
    "speakers.",
)
@click.option(
    "--clean/--no-clean",
    default=True,
    help="Whether to run G2P cleaning.  Pass --no-clean to skip and use an "
    "already-cleaned metadata file.",
)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: Optional[str],
    val_path: Optional[str],
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
) -> None:
    """Preprocess metadata: clean text, extract BERT features, and split data.

    When ``--clean`` is active (the default) each line in *metadata* is
    parsed as ``utt|spk|language|text``, normalised, converted to phonemes,
    and written to *cleaned_path* with the additional columns
    ``norm_text|phones|tones|word2ph``.  BERT features are saved alongside
    each WAV as ``<utt_stem>.bert.pt``.

    The cleaned file is then split into *train_path* / *val_path* lists, and
    *config_path* is updated with speaker-ID mapping, ``num_languages``,
    ``num_tones``, ``symbols``, and the paths to the generated split files.

    Args:
        metadata: Path to raw pipe-delimited metadata list.
        cleaned_path: Destination for the cleaned metadata file.
        train_path: Destination for the training split list.
        val_path: Destination for the validation split list.
        config_path: Source config JSON that will be augmented and saved.
        val_per_spk: Max validation utterances per speaker.
        max_val_total: Hard cap on total validation utterances.
        clean: If ``True``, run G2P and BERT extraction; otherwise read an
            already-cleaned *metadata* file directly.
    """
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), "train.list")
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), "val.list")
    out_config_path = os.path.join(os.path.dirname(metadata), "config.json")

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                utt, spk, language, text = line.strip().split("|")
                norm_text, phones, tones, word2ph, bert = clean_text_bert(
                    text, language, device="cuda:0"
                )
                for ph in phones:
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print("update!, now symbols:")
                        print(new_symbols)
                        with open(f"{language}_symbol.txt", "w") as f:
                            f.write(f"{new_symbols}")

                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                bert_path = utt.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        metadata = cleaned_path

    spk_utt_map: dict[str, list[str]] = defaultdict(list)
    spk_id_map: dict[str, int] = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list: list[str] = []
    val_list: list[str] = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
