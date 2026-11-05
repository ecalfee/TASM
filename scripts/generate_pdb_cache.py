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
import os
import pandas as pd
import json
import argparse
from functools import partial
from multiprocessing import Pool

import sys
sys.path.append(".") # an innocent hack to get this to run from the top level

from tqdm import tqdm

from openfold.data import parsers


def parse_file(file_id, args):
    # check that all necessary files are present
    fasta_path = os.path.join(args.data_dir, f"{file_id}.fasta")
    label_path = os.path.join(args.data_dir, f"{file_id}.label")

    if ((not os.path.exists(fasta_path)) or 
        (not os.path.exists(label_path))):
        raise FileNotFoundError(
            f"Failed to generate cache for {file_id}. Need fasta and label files."
        )

    local_data = {}
    # fasta features
    # read fasta seqs and chain ids
    with open(fasta_path, 'r') as f:
        fasta_str = f.read()
    seqs, tags = parsers.parse_fasta(fasta_str)

    # convert to 4-piece fasta
    fasta_chain_ids, fasta_seqs, = [], []
    for t, s in zip(tags, seqs):
        if ":" in s:
            t1, t2 = t.split("z", 1)
            s1, s2 = s.split(":")
            fasta_chain_ids.extend([t1, t2])
            fasta_seqs.extend([s1, s2])
        else:
            fasta_chain_ids.append(t)
            fasta_seqs.append(s)

    local_data["fasta_chain_ids"] = fasta_chain_ids
    local_data["seqs"] = fasta_seqs

    # pdb features
    # assume D=alpha, E=beta, C=pep, A=mhc
    local_data["chain_ids"] = ["D", "E", "C", "A"]
    local_data["no_chains"] = len(local_data["chain_ids"])

    # label features
    labels = pd.read_csv(label_path, sep="\t")
    local_data["binding_status"] = int(labels["binding_status"].iloc[0])
    local_data["has_predicted_structure"] = int(labels["has_predicted_structure"].iloc[0])
    local_data["has_true_structure"] = int(labels["has_true_structure"].iloc[0])
    local_data["resolution"] = labels["resolution"].iloc[0]

    return {file_id: local_data}


def main(args):
    files = [os.path.splitext(f)[0] for f in os.listdir(args.data_dir) if ".fasta" in f]
    fn = partial(parse_file, args=args)
    data = {}
    with Pool(processes=args.no_workers) as p:
        with tqdm(total=len(files)) as pbar:
            for d in p.imap_unordered(fn, files, chunksize=args.chunksize):
                data.update(d)
                pbar.update()

    with open(args.output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="Directory containing fasta, pdb, label files"
    )
    parser.add_argument(
        "output_path", type=str, help="Path for .json output"
    )
    parser.add_argument(
        "--no_workers", type=int, default=4,
        help="Number of workers to use for parsing"
    )
    parser.add_argument(
        "--chunksize", type=int, default=10,
        help="How many files should be distributed to each worker at a time"
    )

    args = parser.parse_args()

    main(args)
