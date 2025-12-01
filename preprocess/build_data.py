#!/usr/bin/env python3
"""CLI utility to convert raw datasets into TPLinker training files.

This file is a direct script counterpart of ``BuildData.ipynb`` so that
the preprocessing pipeline can run without Jupyter. Pass the same
configuration that the notebook expects (see ``build_data_config.yaml``)
and this script will emit ``train_data.json`` / ``valid_data.json`` /
``test_data.json`` and relevant metadata under ``data_out_dir``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm
from transformers import BertTokenizerFast

from common.utils import Preprocessor


LOGGER = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        try:
            from yaml import CLoader as Loader
        except ImportError:  # pragma: no cover - fallback path
            from yaml import Loader  # type: ignore
        return yaml.load(fp, Loader=Loader)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def _get_tokenizer(config: Dict[str, Any]):
    encoder = config.get("encoder", "BERT")
    if encoder == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(
            config["bert_path"], add_special_tokens=False, do_lower_case=False
        )
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(  # noqa: E731
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]
    elif encoder == "BiLSTM":
        tokenize = lambda text: text.split(" ")  # noqa: E731

        def get_tok2char_span_map(text: str):
            tokens = tokenize(text)
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    return tokenize, get_tok2char_span_map


def _check_tok_span(data, get_tok2char_span_map, tokenize):
    def extr_ent(text, tok_span, tok2char_span):
        char_span_list = tok2char_span[tok_span[0] : tok_span[1]]
        char_span = (char_span_list[0][0], char_span_list[-1][1])
        return text[char_span[0] : char_span[1]]

    span_error_memory = set()
    for sample in tqdm(data, desc="check tok spans"):
        text = sample["text"]
        tok2char_span = get_tok2char_span_map(text)
        for ent in sample["entity_list"]:
            tok_span = ent["tok_span"]
            if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                span_error_memory.add(
                    f"extr ent: {extr_ent(text, tok_span, tok2char_span)}---gold ent: {ent['text']}"
                )

        for rel in sample["relation_list"]:
            subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
            if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                span_error_memory.add(
                    f"extr: {extr_ent(text, subj_tok_span, tok2char_span)}---gold: {rel['subject']}"
                )
            if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                span_error_memory.add(
                    f"extr: {extr_ent(text, obj_tok_span, tok2char_span)}---gold: {rel['object']}"
                )

    return span_error_memory


def build_dataset(config: Dict[str, Any], base_dir: Path | None = None) -> Dict[str, Any]:
    """Run the preprocessing pipeline with the provided configuration."""

    cfg = copy.deepcopy(config)
    base_dir = base_dir or Path.cwd()

    data_in_dir = _resolve_path(base_dir, cfg["data_in_dir"])
    data_out_dir = _resolve_path(base_dir, cfg["data_out_dir"])
    if "bert_path" in cfg:
        cfg["bert_path"] = str(_resolve_path(base_dir, cfg["bert_path"]))

    exp_name = cfg["exp_name"]
    data_in_dir = data_in_dir / exp_name
    data_out_dir = (data_out_dir / exp_name).resolve()
    data_out_dir.mkdir(parents=True, exist_ok=True)

    file_name2data: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in data_in_dir.glob("*.json"):
        file_name = re.match(r"(.*?)\.json", file_path.name).group(1)
        with file_path.open("r", encoding="utf-8") as fp:
            file_name2data[file_name] = json.load(fp)

    tokenize, get_tok2char_span_map = _get_tokenizer(cfg)
    preprocessor = Preprocessor(
        tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map
    )

    ori_format = cfg["ori_data_format"]
    if ori_format != "tplinker":
        for file_name, data in file_name2data.items():
            data_type = "train"
            if "valid" in file_name:
                data_type = "valid"
            elif "test" in file_name:
                data_type = "test"
            file_name2data[file_name] = preprocessor.transform_data(
                data, ori_format=ori_format, dataset_type=data_type, add_id=True
            )

    rel_set = set()
    ent_set = set()
    error_statistics: Dict[str, Dict[str, Any]] = {}
    for file_name, data in file_name2data.items():
        if not data:
            LOGGER.warning("%s is empty, skip span checks", file_name)
            continue
        if "relation_list" in data[0]:
            data = preprocessor.clean_data_wo_span(
                data, separate=cfg["separate_char_by_white"]
            )
            error_statistics[file_name] = {}

            if cfg["add_char_span"]:
                data, miss_samples = preprocessor.add_char_span(
                    data, cfg["ignore_subword"]
                )
                error_statistics[file_name]["miss_samples"] = len(miss_samples)

            for sample in tqdm(
                data, desc="building relation type set and entity type set"
            ):
                if "entity_list" not in sample:
                    ent_list = []
                    for rel in sample["relation_list"]:
                        ent_list.append(
                            {
                                "text": rel["subject"],
                                "type": "DEFAULT",
                                "char_span": rel["subj_char_span"],
                            }
                        )
                        ent_list.append(
                            {
                                "text": rel["object"],
                                "type": "DEFAULT",
                                "char_span": rel["obj_char_span"],
                            }
                        )
                    sample["entity_list"] = ent_list

                for ent in sample["entity_list"]:
                    ent_set.add(ent["type"])

                for rel in sample["relation_list"]:
                    rel_set.add(rel["predicate"])

            data = preprocessor.add_tok_span(data)

            if cfg["check_tok_span"]:
                span_error_memory = _check_tok_span(
                    data, get_tok2char_span_map, tokenize
                )
                error_statistics[file_name]["tok_span_error"] = len(
                    span_error_memory
                )

            file_name2data[file_name] = data

    logging.info("error statistics: %s", error_statistics)

    rel_set = sorted(rel_set)
    rel2id = {rel: ind for ind, rel in enumerate(rel_set)}
    ent_set = sorted(ent_set)
    ent2id = {ent: ind for ind, ent in enumerate(ent_set)}

    data_statistics: Dict[str, Any] = {
        "relation_type_num": len(rel2id),
        "entity_type_num": len(ent2id),
    }

    for file_name, data in file_name2data.items():
        data_path = data_out_dir / f"{file_name}.json"
        with data_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False)
        logging.info("%s is output to %s", file_name, data_path)
        data_statistics[file_name] = len(data)

    rel2id_path = data_out_dir / "rel2id.json"
    with rel2id_path.open("w", encoding="utf-8") as fp:
        json.dump(rel2id, fp, ensure_ascii=False)

    ent2id_path = data_out_dir / "ent2id.json"
    with ent2id_path.open("w", encoding="utf-8") as fp:
        json.dump(ent2id, fp, ensure_ascii=False)

    data_statistics_path = data_out_dir / "data_statistics.txt"
    with data_statistics_path.open("w", encoding="utf-8") as fp:
        json.dump(data_statistics, fp, ensure_ascii=False, indent=4)

    logging.info("data statistics: %s", data_statistics)

    if cfg.get("encoder") == "BiLSTM":
        all_data: List[Dict[str, Any]] = []
        for dataset in file_name2data.values():
            all_data.extend(dataset)

        tokenize_func = tokenize
        token2num = {}
        for sample in tqdm(all_data, desc="Tokenizing"):
            for tok in tokenize_func(sample["text"]):
                token2num[tok] = token2num.get(tok, 0) + 1

        token2num = dict(sorted(token2num.items(), key=lambda x: x[1], reverse=True))
        token_set = set()
        for tok, num in tqdm(token2num.items(), desc="Filter uncommon words"):
            if num < 3:
                continue
            token_set.add(tok)
            if len(token_set) == 50000:
                break

        token2idx = {tok: idx + 2 for idx, tok in enumerate(sorted(token_set))}
        token2idx["<PAD>"] = 0
        token2idx["<UNK>"] = 1

        dict_path = data_out_dir / "token2idx.json"
        with dict_path.open("w", encoding="utf-8") as fp:
            json.dump(token2idx, fp, ensure_ascii=False, indent=4)
        logging.info(
            "token2idx is output to %s, total token num: %s",
            dict_path,
            len(token2idx),
        )

    return {
        "data_out_dir": str(data_out_dir),
        "rel2id_path": str(rel2id_path),
        "ent2id_path": str(ent2id_path),
        "data_statistics_path": str(data_statistics_path),
    }


def run_from_config_file(config_path: Path, overrides: Dict[str, Any] | None = None):
    config_dir = config_path.parent
    config = _load_config(config_path)
    overrides = overrides or {}
    config.update(overrides)
    return build_dataset(config, base_dir=config_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TPLinker datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("build_data_config.yaml"),
        help="Path to build_data_config.yaml",
    )
    parser.add_argument(
        "--exp-name",
        help="Override exp_name from the config file (optional)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    overrides = {}
    if args.exp_name:
        overrides["exp_name"] = args.exp_name
    res = run_from_config_file(args.config, overrides=overrides)
    LOGGER.info("Build finished: %s", res)


if __name__ == "__main__":
    main()
