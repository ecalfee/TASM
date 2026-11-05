# Copyright 2026 Adaptive Biotechnologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare OpenFold input files from a samples CSV.

Provides two operations:
  1. Write per-structure 3-part FASTA files for inference input.
  2. Write per-chain single-sequence FASTA files for alignment precomputation.

Usage:
  python scripts/prepare_input_from_csv.py --samples_csv samples.csv \\
      --output_fastas_dir input_data/ --output_chains_dir chains_for_alignment/
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_unique_chains(
    samples_csv_file: str | Path | None = None,
    chains_list_csv_file: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Extract unique (chain_name, chain_seq) pairs from one of two input sources.

    Exactly one of the two arguments must be provided. Returns a sorted
    list of (chain_name, chain_seq) tuples, deduplicated and validated so
    each chain_name maps to exactly one chain_seq.
    """
    if sum(x is not None for x in [samples_csv_file, chains_list_csv_file]) != 1:
        raise ValueError(
            "Provide exactly one of samples_csv_file or chains_list_csv_file."
        )

    if samples_csv_file is not None:
        df = pd.read_csv(samples_csv_file)

        required_cols = [
            "tcra_name", "tcra_seq",
            "tcrb_name", "tcrb_seq",
            "peptide",
            "mhca_name", "mhca_seq",
            "pmhc_name", "pmhc_seq",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"samples_csv missing required columns: {missing}. "
                f"Required columns are: {required_cols}. "
                f"Optional columns: ['mhcb_name', 'mhcb_seq'] (for class II). "
                f"Columns found: {list(df.columns)}"
            )

        chain_pairs: set[tuple[str, str]] = set()
        has_mhcb = "mhcb_name" in df.columns and "mhcb_seq" in df.columns
        for _, row in df.iterrows():
            chain_pairs.add((row["tcra_name"], row["tcra_seq"]))
            chain_pairs.add((row["tcrb_name"], row["tcrb_seq"]))
            chain_pairs.add((row["mhca_name"], row["mhca_seq"]))
            chain_pairs.add((row["peptide"], row["peptide"]))
            chain_pairs.add((row["pmhc_name"], row["pmhc_seq"]))
            if has_mhcb:
                mhcb_name = row["mhcb_name"]
                mhcb_seq = row["mhcb_seq"]
                name_empty = pd.isna(mhcb_name) or mhcb_name == ""
                seq_empty = pd.isna(mhcb_seq) or mhcb_seq == ""
                if not name_empty and not seq_empty:
                    chain_pairs.add((mhcb_name, mhcb_seq))
                elif name_empty != seq_empty:
                    raise ValueError(
                        f"Row has mhcb_name={mhcb_name!r} but mhcb_seq={mhcb_seq!r}. "
                        "Class II structures must have both or neither."
                    )
        unique_chains = list(sorted(chain_pairs))

    elif chains_list_csv_file is not None:
        df = pd.read_csv(chains_list_csv_file)
        if "chain_name" not in df.columns or "chain_seq" not in df.columns:
            raise ValueError(
                "chains_list_csv_file must have columns 'chain_name' and 'chain_seq'."
            )
        unique_chains = list(sorted(set(
            zip(df["chain_name"].tolist(), df["chain_seq"].tolist())
        )))
        if len(unique_chains) != len(set(name for name, seq in unique_chains)):
            raise ValueError(
                "chains_list_csv_file must not have duplicate chain_name entries "
                "with different chain_seq values."
            )

    else:
        raise ValueError(
            "Must provide one of samples_csv_file or chains_list_csv_file."
        )

    for name, seq in unique_chains:
        if pd.isna(name) or name == "":
            raise ValueError(f"Found chain with empty name: seq={seq!r}")
        if pd.isna(seq) or seq == "":
            raise ValueError(f"Found chain with empty sequence: name={name!r}")

    from collections import Counter

    name_counts = Counter(name for name, seq in unique_chains)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    if duplicates:
        dup_details = []
        for name in sorted(duplicates):
            seqs = [seq for n, seq in unique_chains if n == name]
            dup_details.append(f"  {name}: {seqs}")
        raise ValueError(
            "Multiple different sequences found for the same chain_name. "
            "Each chain_name must map to exactly one chain_seq.\n"
            + "\n".join(dup_details)
        )

    logger.info("Found %d unique chains", len(unique_chains))
    return unique_chains


def write_single_chain_fasta(
    fasta_dir: str,
    chains: list[tuple[str, str]],
) -> None:
    """Write one single-chain FASTA file per (name, seq) pair for alignment precomputation."""
    os.makedirs(fasta_dir, exist_ok=True)
    for name, seq in chains:
        with open(os.path.join(fasta_dir, f"{name}.fasta"), "w") as fh:
            fh.write(f">{name}\n{seq}")


def write_openfold_input_fastas(
    output_dir: str,
    samples_csv_file: str | Path,
) -> None:
    """Write 3-part FASTA files (tcra, tcrb, pmhc) for OpenFold inference input.

    Each file is named {structure_id}.fasta and contains three sequences.
    """
    df = pd.read_csv(samples_csv_file)

    required_cols = [
        "structure_id", "tcra_name", "tcrb_name", "pmhc_name",
        "tcra_seq", "tcrb_seq", "pmhc_seq",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"samples_csv missing required columns: {missing}")

    if df["structure_id"].nunique() != len(df):
        dupes = df[df["structure_id"].duplicated(keep=False)]["structure_id"].unique()
        raise ValueError(
            f"structure_id column must have all unique values. "
            f"Duplicates: {list(dupes[:10])}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        fasta = f">{row['tcra_name']}\n{row['tcra_seq']}\n"
        fasta += f">{row['tcrb_name']}\n{row['tcrb_seq']}\n"
        fasta += f">{row['pmhc_name']}\n{row['pmhc_seq']}\n"
        fasta_path = os.path.join(output_dir, f"{row['structure_id']}.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta)

    logger.info("Wrote %d input FASTA files to %s", len(df), output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare OpenFold input files from a samples CSV."
    )
    parser.add_argument(
        "--samples_csv", required=True,
        help="Path to samples CSV file.",
    )
    parser.add_argument(
        "--output_fastas_dir", default=None,
        help="Output directory for per-structure 3-part FASTA files (inference input).",
    )
    parser.add_argument(
        "--output_chains_dir", default=None,
        help="Output directory for per-chain single-sequence FASTA files (alignment precomputation input).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.output_fastas_dir is None and args.output_chains_dir is None:
        parser.error("Provide at least one of --output_fastas_dir or --output_chains_dir.")

    if args.output_chains_dir is not None:
        chains = get_unique_chains(samples_csv_file=args.samples_csv)
        write_single_chain_fasta(args.output_chains_dir, chains)
        logger.info("Wrote %d single-chain FASTA files to %s", len(chains), args.output_chains_dir)

    if args.output_fastas_dir is not None:
        write_openfold_input_fastas(args.output_fastas_dir, args.samples_csv)
