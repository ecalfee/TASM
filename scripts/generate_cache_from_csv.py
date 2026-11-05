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
import argparse
import json
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

def get_cache(file_id):
    """Create cache for a single csv file.
    """
    file_path = os.path.join(args.csv_dir, f"{file_id}.csv")
    df = pd.read_csv(file_path)

    # check that df doesn't have samples with has_true_structure=0 and resolution!=None
    if df[(df['resolution'].notnull()) & (df['has_true_structure'] == 0)].shape[0] > 0:
        raise ValueError(
            f"{file_path} has samples with has_true_structure=0 and resolution!=None. "
            "Samples without true structure are expected to have resolution set to None."
        )

    df["resolution"] = df["resolution"].fillna(0.0)  # replace None or NaN with 0.0

    local_cache = {}
    for _, row in df.iterrows():
        file_id = row["structure_id"]
        fasta_chain_ids = [row["tcra_name"], row["tcrb_name"], row["peptide"], row["mhca_name"]]
        seqs = [row["tcra_seq"], row["tcrb_seq"], row["peptide"], row["mhca_seq"]]
        chain_ids = ["D", "E", "C", "A"] # assume D=alpha, E=beta, C=pep, A=mhc
        no_chains = len(chain_ids)
        binding_status = row["binding_status"]
        has_true_structure = row["has_true_structure"]
        has_predicted_structure = row["has_predicted_structure"]
        resolution = row["resolution"]

        local_cache[file_id] = {
            "fasta_chain_ids": fasta_chain_ids,
            "seqs": seqs,
            "chain_ids": chain_ids,
            "no_chains": no_chains,
            "binding_status": binding_status,
            "has_predicted_structure": has_predicted_structure,
            "has_true_structure": has_true_structure,
            "resolution": resolution,
        }

        # Need to check since validation_groups is not present train input csvs
        if "validation_groups" in row:
            local_cache[file_id]["validation_groups"] = row["validation_groups"]

    return local_cache

def main(args):
    # get csv names (e.g., 'A0201_negative_structures_set49')
    files = [os.path.splitext(f)[0] for f in os.listdir(args.csv_dir) if ".csv" in f]

    data = {}
    with Pool(processes=args.num_workers) as p:
        with tqdm(total=len(files)) as pbar:
            for d in p.imap_unordered(get_cache, files, chunksize=args.chunksize):
                data.update(d)
                pbar.update()

    # Assign a unique index to each sample
    for i, key in enumerate(data.keys()):
        data[key]['sample_index'] = i

    with open(args.output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_dir", type=str, help="Directory containing summary csv files"
    )
    parser.add_argument(
        "output_path", type=str, help="E.g., train_data_cache.json"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers to use for parsing"
    )
    parser.add_argument(
        "--chunksize", type=int, default=10,
        help="How many files should be distributed to each worker at a time"
    )

    args = parser.parse_args()

    main(args)