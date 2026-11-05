# Modified by Adaptive Biotechnologies
import argparse
from functools import partial
import json
import logging
import os
import threading
from multiprocessing import cpu_count
import shutil
import tempfile

import openfold.data.mmcif_parsing as mmcif_parsing
from openfold.data.data_pipeline import AlignmentRunner
from openfold.data.parsers import parse_fasta
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein, residue_constants

from utils import add_data_args


logging.basicConfig(level=logging.WARNING)


def run_seq_group_alignments(seq_groups, alignment_runner, args,
                             completed_chains, errored_chains,
                             skipped_chains):
    dirs = set(os.listdir(args.output_dir)) if os.path.isdir(args.output_dir) else set()
    for seq, names in seq_groups:
        first_name = names[0]
        alignment_dir = os.path.join(args.output_dir, first_name)

        if first_name in dirs:
            logging.warning(f"Directory for {first_name} already exists. Skipping...")
            skipped_chains.append(first_name)
            continue

        os.makedirs(alignment_dir)

        fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
        with os.fdopen(fd, 'w') as fp:
            fp.write(f'>query\n{seq}')

        try:
            alignment_runner.run(
                fasta_path, alignment_dir, seq
            )
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Failed to run alignments for {first_name}. Skipping...")
            os.remove(fasta_path)
            shutil.rmtree(alignment_dir, ignore_errors=True)
            errored_chains.append(f"{first_name}\t{e}")
            continue

        os.remove(fasta_path)
        completed_chains.append(first_name)

        for name in names[1:]:
            if(name in dirs):
                logging.warning(
                    f'{name} has already been processed. Skipping...'
                )
                skipped_chains.append(name)
                continue

            cp_dir = os.path.join(args.output_dir, name)
            os.makedirs(cp_dir, exist_ok=True)
            for f in os.listdir(alignment_dir):
                shutil.copyfile(os.path.join(alignment_dir, f), os.path.join(cp_dir, f))
            completed_chains.append(name)


def parse_and_align(files, alignment_runner, args,
                    completed_chains, errored_chains,
                    skipped_chains):
    for f in files:
        path = os.path.join(args.input_dir, f)
        file_id = os.path.splitext(f)[0]
        seq_group_dict = {}  
        if(f.endswith('.cif')):
            with open(path, 'r') as fp:
                mmcif_str = fp.read()
            mmcif = mmcif_parsing.parse(
                file_id=file_id, mmcif_string=mmcif_str
            )
            if(mmcif.mmcif_object is None):
                logging.warning(f'Failed to parse {f}...')
                if(args.raise_errors):
                    raise list(mmcif.errors.values())[0]
                else:
                    continue
            mmcif = mmcif.mmcif_object
            for chain_letter, seq in mmcif.chain_to_seqres.items():
                chain_id = '_'.join([file_id, chain_letter])
                l = seq_group_dict.setdefault(seq, [])
                l.append(chain_id)
        elif(f.endswith('.fasta') or f.endswith('.fa')):
            with open(path, 'r') as fp:
                fasta_str = fp.read()
            input_seqs, _ = parse_fasta(fasta_str)
            if len(input_seqs) != 1: 
                msg = f'More than one input_sequence found in {f}'
                if(args.raise_errors):
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            input_sequence = input_seqs[0]
            seq_group_dict[input_sequence] = [file_id]
        elif(f.endswith('.core')):
            with open(path, 'r') as fp:
                core_str = fp.read()
            core_prot = protein.from_proteinnet_string(core_str)
            aatype = core_prot.aatype
            seq = ''.join([
                residue_constants.restypes_with_x[aatype[i]] 
                for i in range(len(aatype))
            ])
            seq_group_dict[seq] = [file_id]
        else:
            continue

        seq_group_tuples = [(k,v) for k,v in seq_group_dict.items()]
        run_seq_group_alignments(seq_group_tuples, alignment_runner, args,
                                completed_chains, errored_chains,
                                skipped_chains)


def main(args):
    # Build the alignment tool runner
    if args.hmmsearch_binary_path is not None and args.pdb_seqres_database_path is not None:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=args.hmmsearch_binary_path,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            database_path=args.pdb_seqres_database_path,
        )
    elif args.hhsearch_binary_path is not None and args.pdb70_database_path is not None:
        template_searcher = hhsearch.HHSearch(
            binary_path=args.hhsearch_binary_path,
            databases=[args.pdb70_database_path],
        )
    else:
        template_searcher = None

    alignment_runner = AlignmentRunner(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniref30_database_path=args.uniref30_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        uniprot_database_path=args.uniprot_database_path,
        pmhc_templates_dir=args.pmhc_templates_dir,
        template_searcher=template_searcher,
        use_small_bfd=args.use_small_bfd,
        no_cpus=args.cpus_per_task,
        use_pmhc_template_filter=args.use_pmhc_template_filter,
    )

    files = list(os.listdir(args.input_dir))

    # Do some filtering
    if(args.mmcif_cache is not None):
        with open(args.mmcif_cache, "r") as fp:
            cache = json.load(fp)
    else:
        cache = None

    dirs = []
    if(cache is not None and args.filter):
        dirs = set(os.listdir(args.output_dir)) if os.path.isdir(args.output_dir) else set()
        def prot_is_done(f):
            prot_id = os.path.splitext(f)[0]
            if(prot_id in cache):
                chain_ids = cache[prot_id]["chain_ids"]
                for c in chain_ids:
                    full_name = prot_id + "_" + c
                    if(not full_name in dirs):
                        return False
            else:
                return False

            return True

        files = [f for f in files if not prot_is_done(f)]

    def split_up_arglist(arglist):
        # Split up the survivors
        if(os.environ.get("SLURM_JOB_NUM_NODES", 0)):
            num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
            if(num_nodes > 1):
                node_id = int(os.environ["SLURM_NODEID"])
                logging.warning(f"Num nodes: {num_nodes}")
                logging.warning(f"Node ID: {node_id}")
                arglist = arglist[node_id::num_nodes]

        t_arglist = []
        for i in range(args.no_tasks):
            t_arglist.append(arglist[i::args.no_tasks])

        return t_arglist

    if(cache is not None and "seqs" in next(iter(cache.values()))):
        seq_group_dict = {}
        for f in files:
            prot_id = os.path.splitext(f)[0]
            if(prot_id in cache):
                prot_cache = cache[prot_id]
                chains_seqs = zip(
                    prot_cache["chain_ids"], prot_cache["seqs"]
                )
                for chain, seq in chains_seqs:
                    chain_name = prot_id + "_" + chain
                    if(chain_name not in dirs):
                        l = seq_group_dict.setdefault(seq, [])
                        l.append(chain_name)
       
        completed_chains = []
        errored_chains = []
        skipped_chains = []
        func = partial(run_seq_group_alignments,
            alignment_runner=alignment_runner,
            args=args,
            completed_chains=completed_chains,
            errored_chains=errored_chains,
            skipped_chains=skipped_chains,
        )

        seq_groups = [(k,v) for k,v in seq_group_dict.items()]

        # Sort them by group length so the tasks are approximately balanced
        seq_groups = sorted(seq_groups, key=lambda x: len(x[1]))

        task_arglist = [[a] for a in split_up_arglist(seq_groups)]
    else:
        completed_chains = []
        errored_chains = []
        skipped_chains = []
        func = partial(parse_and_align,
            alignment_runner=alignment_runner,
            args=args,
            completed_chains=completed_chains,
            errored_chains=errored_chains,
            skipped_chains=skipped_chains,
        )
        task_arglist = [[a] for a in split_up_arglist(files)]

    threads = []
    for i, task_args in enumerate(task_arglist):
        print(f"Started thread {i}...")
        t = threading.Thread(target=func, args=task_args)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    status = {
        "completed": completed_chains,
        "errored": errored_chains,
        "skipped": skipped_chains,
    }
    with open(os.path.join(args.output_dir, ".status.json"), 'w') as f:
        json.dump(status, f)
    print(f"{len(completed_chains)} completed, "
          f"{len(skipped_chains)} skipped (alignment dir already exists), "
          f"{len(errored_chains)} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir", type=str,
        help="""Path to directory containing mmCIF, FASTA and/or ProteinNet
                .core files"""
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Directory in which to output alignments"
    )
    add_data_args(parser)
    parser.add_argument(
        "--raise_errors", action="store_true", default=False,
        help="Whether to crash on parsing errors"
    )
    parser.add_argument(
        "--cpus_per_task", type=int, default=cpu_count(),
        help="Number of CPUs to use"
    )
    parser.add_argument(
        "--mmcif_cache", type=str, default=None,
        help="Path to mmCIF cache. Used to filter files to be parsed"
    )
    parser.add_argument(
        "--no_tasks", type=int, default=1,
    )
    parser.add_argument(
        "--filter", type=bool, default=True,
    )
    parser.add_argument(
        "--use_small_bfd", action="store_true", default=False,
    )
    parser.add_argument(
        "--pmhc_templates_dir", type=str, default=None,
        help="""Name of the directory containing pMHC /pdb and nr_cls_I_complexes.txt""",
    )
    parser.add_argument(
        "--use_pmhc_template_filter", action="store_true", default=False,
    )
    args = parser.parse_args()

    main(args)
