#!/usr/bin/env python3
"""One-click preparation pipeline for the MDIEC dataset.

Steps performed:
1. Download and extract the official MDIEC archive from Hugging Face.
2. Convert the BRAT ``.txt/.ann`` files into TPLinker JSON format.
3. Invoke the standard build-data pipeline to produce inputs under ``data4bert``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_data_module():
    module_path = REPO_ROOT / "preprocess" / "build_data.py"
    spec = importlib.util.spec_from_file_location("build_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load build_data module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_data_mod = _load_build_data_module()
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "NER&RE_model" / "InputsAndOutputs" / "data" / "dataset" / "MDIEC"
)
DEFAULT_ORI_OUTPUT = REPO_ROOT / "preprocess" / "ori_data" / "mdiec"
DEFAULT_CONFIG = REPO_ROOT / "preprocess" / "build_data_config.yaml"
DEFAULT_BERT = REPO_ROOT / "pretrained_models" / "bert-base-cased"


@dataclass
class Entity:
    ent_id: str
    ent_type: str
    start: int
    end: int
    text: str


@dataclass
class Relation:
    rel_type: str
    arg1: str
    arg2: str


def run_fetch_script(overwrite: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "fetch_official_data.py")]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def parse_ann_file(ann_path: Path) -> Tuple[Dict[str, Entity], List[Relation]]:
    entities: Dict[str, Entity] = {}
    relations: List[Relation] = []
    with ann_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith("T"):
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                ent_id = parts[0]
                type_and_span = parts[1].split()
                ent_type = type_and_span[0]
                start, end = map(int, type_and_span[1:3])
                entities[ent_id] = Entity(
                    ent_id=ent_id,
                    ent_type=ent_type,
                    start=start,
                    end=end,
                    text=parts[2],
                )
            elif line.startswith("R"):
                parts = line.split()
                if len(parts) < 4:
                    continue
                rel_type = parts[1]
                arg1 = parts[2].split(":", 1)[1]
                arg2 = parts[3].split(":", 1)[1]
                relations.append(Relation(rel_type=rel_type, arg1=arg1, arg2=arg2))
    return entities, relations


def convert_sample(doc_id: str, dataset_dir: Path) -> Dict[str, object]:
    txt_path = dataset_dir / f"{doc_id}.txt"
    ann_path = dataset_dir / f"{doc_id}.ann"
    text = txt_path.read_text(encoding="utf-8")
    entities, relations = parse_ann_file(ann_path)

    entity_list = [
        {
            "text": ent.text,
            "type": ent.ent_type,
            "char_span": [ent.start, ent.end],
        }
        for ent in sorted(entities.values(), key=lambda e: (e.start, e.end))
    ]

    rel_memory = set()
    relation_list = []
    for rel in relations:
        subj = entities.get(rel.arg1)
        obj = entities.get(rel.arg2)
        if not subj or not obj:
            continue
        rel_key = (subj.ent_id, obj.ent_id, rel.rel_type)
        if rel_key in rel_memory:
            continue
        rel_memory.add(rel_key)
        relation_list.append(
            {
                "subject": subj.text,
                "object": obj.text,
                "predicate": rel.rel_type,
                "subj_char_span": [subj.start, subj.end],
                "obj_char_span": [obj.start, obj.end],
            }
        )

    return {
        "text": text,
        "id": doc_id,
        "entity_list": entity_list,
        "relation_list": relation_list,
    }


def convert_dataset(dataset_dir: Path) -> List[Dict[str, object]]:
    doc_ids = []
    for txt_path in sorted(dataset_dir.glob("*.txt")):
        if not txt_path.with_suffix(".ann").exists():
            continue
        doc_ids.append(txt_path.stem)
    samples = [convert_sample(doc_id, dataset_dir) for doc_id in doc_ids]
    return samples


def split_samples(
    samples: List[Dict[str, object]], valid_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Dict[str, object]]]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    total = len(samples)
    n_valid = int(round(total * valid_ratio))
    n_test = int(round(total * test_ratio))
    n_train = max(total - n_valid - n_test, 0)
    if total > 0 and n_train == 0:
        if n_valid > n_test and n_valid > 0:
            n_valid -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = min(1, total)

    train = samples[:n_train]
    valid = samples[n_train : n_train + n_valid]
    test = samples[n_train + n_valid :]
    return {"train": train, "valid": valid, "test": test}


def dump_ori_data(split_data: Dict[str, List[Dict[str, object]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in split_data.items():
        path = output_dir / f"{split_name}_data.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False)
        print(f"Wrote {len(data)} samples to {path}")


def run_build_data(exp_name: str, data_in_root: Path, data_out_root: Path, bert_path: Path):
    overrides = {
        "exp_name": exp_name,
        "ori_data_format": "tplinker",
        "data_in_dir": str(data_in_root),
        "data_out_dir": str(data_out_root),
        "encoder": "BERT",
        "bert_path": str(bert_path),
    }
    build_data_mod.run_from_config_file(DEFAULT_CONFIG, overrides=overrides)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MDIEC for TPLinker")
    parser.add_argument("--exp-name", default="mdiec", help="Experiment name / folder")
    parser.add_argument(
        "--valid-ratio", type=float, default=0.1, help="Validation set proportion"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set proportion"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed for split")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory with MDIEC .txt/.ann files",
    )
    parser.add_argument(
        "--ori-output-dir",
        type=Path,
        default=DEFAULT_ORI_OUTPUT,
        help="Where to place intermediate TPLinker JSON",
    )
    parser.add_argument(
        "--data-out-root",
        type=Path,
        default=REPO_ROOT / "data4bert",
        help="Root directory for processed data (matching config)",
    )
    parser.add_argument(
        "--bert-path",
        type=Path,
        default=DEFAULT_BERT,
        help="Path to pretrained BERT weights",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing MDIEC download and skip huggingface fetch",
    )
    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download of the MDIEC archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.valid_ratio < 1 or not 0 <= args.test_ratio < 1:
        raise ValueError("valid/test ratios must be within [0, 1)")
    if args.valid_ratio + args.test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be < 1")

    if not args.skip_download:
        run_fetch_script(overwrite=args.overwrite_download)

    samples = convert_dataset(args.dataset_dir)
    splits = split_samples(samples, args.valid_ratio, args.test_ratio, args.seed)
    dump_ori_data(splits, args.ori_output_dir)

    run_build_data(
        exp_name=args.exp_name,
        data_in_root=args.ori_output_dir.parent,
        data_out_root=args.data_out_root,
        bert_path=args.bert_path,
    )


if __name__ == "__main__":
    main()
