# TASM: T-cell Activation via Structural Modeling

TASM extends the OpenFold structure model with a new classification layer to predict TCR activation and modifies OpenFold's training procedure to learn directly from activation-labeled TCR-pMHCs sequences without ground truth structures. TASM was trained on a large A*02:01 dataset of experimentally-labeled activating and non-activating TCR-pMHC sequences. TASM additionally incorporates TCRmodel2's TCR-pMHC specific pMHC templating and faster MSA searches.

The baseline model, OF-TCRmodel2, uses the same OpenFold + tcrmodel2 architecture but is run with AlphaFold2 Multimer pretrained parameters and no additional training on activation data. OF-TCRmodel2 inference outputs TCR-pMHC structure predictions and the TCR-pMHC ipTM scores that were used to predict activation in the TASM paper.

## Installation

Follow the OpenFold installation instructions in [OpenFold_README.md](OpenFold_README.md). Key dependencies include PyTorch, DeepSpeed, and OpenMM.

## Reference Databases

### AlphaFold2 Parameters and PDB Templates

Download AF2 model parameters and PDB mmCIF template files following the instructions in the [AlphaFold repository](https://github.com/google-deepmind/alphafold) from Google DeepMind ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)):

- **AF2 parameters**: Download `params_model_1_multimer_v3.npz` and place in `openfold/resources/params/`.
- **PDB mmCIF files**: Used as structural templates during inference. See AlphaFold's `scripts/download_all_data.sh` for download instructions.

### MSA and Template Search Databases

TASM and OF-tcrmodel2 use the reduced TCR-pMHC-specific databases from TCRmodel2. Download the database bundle from the [TCRmodel2 repository](https://github.com/piercelab/tcrmodel2). This includes:

- `uniref90.tcrmhc.fasta` — for MSA search (jackhmmer)
- `mgnify.fasta` — for MSA search (jackhmmer)
- `small_bfd.tcrmhc.fasta` — for MSA search (jackhmmer)
- `uniprot.tcrmhc.fasta` — for MSA search (jackhmmer)
- `pdb_seqres.txt` — for template search (hmmsearch)
- `pmhc_templates/` — joint pMHC template PDBs (contains `pdb/` and `nr_cls_I_complexes.txt`)

**Required tools:** `jackhmmer`, `hhblits`, `hmmsearch`, `hmmbuild`, `kalign`

## Input Data Format

All workflows (inference, alignment precomputation, and training) use a **samples CSV** file. Prepare your data as a CSV with the columns described below.

### Required Columns (Inference and Alignment Precomputation)

| Column | Description | Example |
|--------|-------------|---------|
| `structure_id` | Unique identifier for each TCR-pMHC complex | `sample_001` |
| `tcra_name` | Unique name for the TCR alpha chain | `TCRA_hash1234` |
| `tcra_seq` | TCR alpha chain amino acid sequence, trimmed to the variable domain region | `AQTVTQSQPEMSVQEAE...` |
| `tcrb_name` | Unique name for the TCR beta chain | `TCRB_hash5678` |
| `tcrb_seq` | TCR beta chain amino acid sequence, trimmed to the variable domain region | `ETGVTQSPTHLIKTRGQ...` |
| `peptide` | Peptide amino acid sequence (used as both name and sequence) | `GILGFVFTL` |
| `mhca_name` | MHC alpha chain name | `A0201` |
| `mhca_seq` | MHC alpha chain amino acid sequence | `GSHSMRYFFTSVSRPGR...` |
| `pmhc_name` | pMHC complex name: `{peptide}z{mhca_name}` | `GILGFVFTLzA0201` |
| `pmhc_seq` | pMHC complex sequence: `{peptide}:{mhca_seq}` (colon-separated) | `GILGFVFTL:GSHSMRY...` |

**Naming and sequence conventions:**
- **TCR sequences must be pre-trimmed** to the variable domain region before input. The pipeline does not perform trimming.
- `pmhc_name` uses `z` as the separator between peptide and MHC name (e.g., `GILGFVFTLzA0201`).
- `pmhc_seq` uses `:` as the separator between peptide and MHC sequences (e.g., `GILGFVFTL:GSHSMRYF...`). The `:` in the sequence is what triggers the pMHC-specific template alignment path during preprocessing (see below).
- **`z` and `:` are reserved characters.** Lowercase `z` must not appear in any other chain names, and `:` must not appear in any other chain sequences. These characters are used internally to identify and split pMHC complex names and sequences.
- Each unique chain sequence must have a unique chain name, because alignments are saved and looked up by chain name.
- `structure_id` values must be unique across all rows.

### Additional Columns Required for Training

| Column | Description |
|--------|-------------|
| `binding_status` | Activation label from the MIRA assay (int: 0 = non-activating, 1 = activating). Despite the column name, TASM is trained on MIRA-derived activation labels, not binding measurements. |
| `has_true_structure` | Whether a crystal structure is provided for this sample (int: 0 or 1) |
| `has_predicted_structure` | Whether a predicted structure is provided for this sample (int: 0 or 1) |
| `resolution` | Crystal structure resolution in angstroms (float). Must be empty/None for samples without a crystal structure. |

For validation CSVs, also include:
| Column | Description |
|--------|-------------|
| `validation_groups` | Comma-separated group labels for validation subsets (e.g., `"A0201"`, `"A0201,crystal"`) |

Training and validation input directories must each contain:
- A samples CSV file with the columns above.
- A `.pdb` file for each crystal structure referenced in the CSV (i.e., rows where `has_true_structure=1`), named `{structure_id}.pdb`.

Use `scripts/generate_cache_from_csv.py` to generate the data cache JSON files required by the training script:
```bash
python scripts/generate_cache_from_csv.py <csv_dir> <output_cache_path.json>
```

## Preparing Input Files from CSV

Use `scripts/prepare_input_from_csv.py` to convert a samples CSV into the FASTA files needed by OpenFold:

```bash
# Generate both inference input FASTAs and single-chain FASTAs for alignment precomputation:
python scripts/prepare_input_from_csv.py \
    --samples_csv your_samples.csv \
    --output_fastas_dir input_data/ \
    --output_chains_dir chains_for_alignment/
```

This produces:
- `input_data/`: One 3-part FASTA per structure (TCR-alpha, TCR-beta, pMHC) for inference input.
- `chains_for_alignment/`: One single-chain FASTA per unique chain for alignment precomputation.

## Preprocessing: MSA and Template Alignments

Before running inference or training, precompute MSA and template alignments for all unique chains. The alignment pipeline handles different chain types automatically:

- **Individual chains** (TCRA, TCRB, peptide, MHCA): Full MSA search via jackhmmer against uniref90, mgnify, small_bfd, and uniprot databases, plus template search via hmmsearch against pdb_seqres.
- **pMHC chains** (detected by the `:` in the sequence): Structural template alignment against TCRmodel2's joint pMHC template database. No MSA search is performed for pMHC chains.

```bash
python scripts/precompute_alignments.py \
    chains_for_alignment/ \
    alignments/ \
    --uniref90_database_path <path_to_databases>/uniref90.tcrmhc.fasta \
    --mgnify_database_path <path_to_databases>/mgnify.fasta \
    --pdb_seqres_database_path <path_to_databases>/pdb_seqres.txt \
    --uniprot_database_path <path_to_databases>/uniprot.tcrmhc.fasta \
    --bfd_database_path <path_to_databases>/small_bfd.tcrmhc.fasta \
    --use_small_bfd \
    --hmmsearch_binary_path hmmsearch \
    --hmmbuild_binary_path hmmbuild \
    --jackhmmer_binary_path jackhmmer \
    --kalign_binary_path kalign \
    --cpus_per_task 8
```

See `python scripts/precompute_alignments.py --help` for all options.

**Precomputed Alignment Outputs:** One directory per chain name under the output directory:
- Individual chains: `uniref90_hits.sto`, `mgnify_hits.sto`, `small_bfd_hits.sto`, `uniprot_hits.sto`, plus hmmsearch template hits.
- pMHC chains: `pmhc_alignment.tsv`.

## Inference

OF-tcrmodel2 baseline model can be run with AlphaFold-Multimer's pretrained weights ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)) downloaded from the [AlphaFold](https://github.com/google-deepmind/alphafold) repository.


TASM model weights are not released.

```bash
python run_pretrained_openfold.py \
    input_data/ \
    <path_to_pdb_mmcif>/mmcif_files/ \
    --use_precomputed_alignments alignments/ \
    --config_preset model_1_multimer_v3 \
    --output_dir output/ \
    --model_device cuda:0 \
    --data_random_seed <seed> \
    --pmhc_templates_dir <path_to_pmhc_templates>/ \
    --jax_param_path openfold/resources/params/params_model_1_multimer_v3.npz \
    --skip_relaxation \
    --save_template_names \
    --use_deepspeed_evoformer_attention
```

## Training

Training scripts and commands are provided for review. Training data is not released.

All training experiments started from AlphaFold2 Multimer v3 parameter set 1, except TASM phase 2 which resumed from a TASM phase 1 checkpoint.

Training requires a directory structure with separate train/validation subdirectories for input data and alignments. Use `scripts/generate_cache_from_csv.py` to generate data cache JSON files from your samples CSVs.

**Optional flags** (not shown in commands below):
- `--wandb --wandb_project <project> --wandb_entity <entity>` (logging: enable Weights & Biases experiment tracking; requires `pip install wandb`)
- `--kubeflow` (infrastructure: required when running on Kubeflow-managed multi-node clusters)

The `--gpus` and `--num_nodes` values shown below reflect our training setup; adjust to match your infrastructure.

### TASM Phase 1 (Pilot)

Training command and yaml provided for reference.

```bash
python train_openfold.py \
    <train_data_dir> \
    <train_alignment_dir> \
    <path_to_pdb_mmcif>/mmcif_files/ \
    <output_dir> \
    2021-10-10 \
    --val_data_dir <val_data_dir> \
    --val_alignment_dir <val_alignment_dir> \
    --train_mmcif_data_cache_path <train_data_cache>.json \
    --val_mmcif_data_cache_path <val_data_cache>.json \
    --template_release_dates_cache_path <template_data_cache>.json \
    --obsolete_pdbs_file_path <path_to_pdb_mmcif>/obsolete.dat \
    --config_preset model_1_multimer_v3 \
    --kalign_binary_path kalign \
    --seed 123 \
    --gpus 4 --num_nodes 5 \
    --max_epochs 20 \
    --train_epoch_len 20000 \
    --num_pos_crystal_per_train_batch 2 \
    --num_pos_predicted_per_train_batch 9 \
    --num_neg_predicted_per_train_batch 9 \
    --val_check_interval 100 \
    --checkpoint_every_validation \
    --num_samples_per_validation 500 \
    --num_pos_crystal_per_val_batch 2 \
    --num_pos_predicted_per_val_batch 9 \
    --num_neg_predicted_per_val_batch 9 \
    --use_custom_sampler \
    --resume_from_jax_params openfold/resources/params/params_model_1_multimer_v3.npz \
    --experiment_config_json training_configs/TASM_phase1.json \
    --deepspeed_config_path training_configs/deepspeed_config_bfloat16.json \
    --log_lr --log_every_n_steps 1
```

### TASM Phase 2

Training command and yaml provided for reference. Resumed from a TASM phase 1 checkpoint (model weights only).

```bash
python train_openfold.py \
    <train_data_dir> \
    <train_alignment_dir> \
    <path_to_pdb_mmcif>/mmcif_files/ \
    <output_dir> \
    2021-10-10 \
    --val_data_dir <val_data_dir> \
    --val_alignment_dir <val_alignment_dir> \
    --train_crystal_mmcif_data_cache_path <train_crystal_data_cache>.json \
    --train_pos_mmcif_data_cache_path <train_positive_data_cache>.json \
    --train_neg_mmcif_data_cache_path <train_negative_data_cache>.json \
    --val_mmcif_data_cache_path <val_data_cache>.json \
    --template_release_dates_cache_path <template_data_cache>.json \
    --obsolete_pdbs_file_path <path_to_pdb_mmcif>/obsolete.dat \
    --config_preset model_1_multimer_v3 \
    --kalign_binary_path kalign \
    --seed 46 \
    --gpus 4 --num_nodes 10 \
    --max_epochs 20 \
    --train_epoch_len 139440 \
    --num_pos_crystal_per_train_batch 2 \
    --num_pos_predicted_per_train_batch 0 \
    --num_pos_no_structure_per_train_batch 19 \
    --num_neg_no_structure_per_train_batch 19 \
    --val_check_interval 100 \
    --checkpoint_every_validation \
    --num_samples_per_validation 480 \
    --num_pos_crystal_per_val_batch 2 \
    --num_pos_predicted_per_val_batch 0 \
    --num_pos_no_structure_per_val_batch 19 \
    --num_neg_no_structure_per_val_batch 19 \
    --use_custom_sampler \
    --resume_from_ckpt <path_to_phase1_checkpoint>.pt \
    --resume_model_weights_only True \
    --experiment_config_json training_configs/TASM_phase2.json \
    --deepspeed_config_path training_configs/deepspeed_config_bfloat16.json \
    --log_lr --log_every_n_steps 1
```

## Training Directory Structure

```
dataset/
  input_data/
    train/
      samples_train.csv
      1ao7.pdb            # crystal structure PDB files (if any)
    validation/
      samples_validation.csv
      ...
  alignments/
    train/
      <chain_name>/       # one directory per unique chain
        ...               # alignment output files
    validation/
      <chain_name>/
        ...
  data_cache/
    train_crystal_data_cache.json
    train_positive_data_cache.json
    train_negative_data_cache.json
    validation_data_cache.json
    template_data_cache.json
```

Use `scripts/generate_cache_from_csv.py` to generate data cache JSON files from your samples CSVs.

## Attribution

TASM builds on the following openly licensed projects:

- **[OpenFold](https://github.com/aqlaboratory/openfold)** ([Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)) — We extend and train on the OpenFold model code. Modified files carry prominent notices per Apache 2.0 Section 4(b).
  Ahdritz, G. et al. "OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization." *Nature Methods* (2024).

- **[AlphaFold-Multimer](https://github.com/google-deepmind/alphafold)** v2.3 parameter set 1 model weights ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)) — We initialize TASM with and fine-tune the pretrained AlphaFold-Multimer weights provided by Google DeepMind.
  Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* (2021).
  Evans, R. et al. "Protein complex prediction with AlphaFold-Multimer." *bioRxiv* (2022).

- **[TCRmodel2](https://github.com/piercelab/tcrmodel2)** ([Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)) — TASM uses the TCR-pMHC-specific MSA/template search databases and pMHC template alignment pipeline from TCRmodel2. Interface pLDDT calculation and pMHC template search code are adapted from TCRmodel2.
  Yin, R. et al. "TCRmodel2: high-resolution modeling of T cell receptor recognition using deep learning." *Nucleic Acids Research* (2023).

- **[alphafold_finetune](https://github.com/phbradley/alphafold_finetune)** ([Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)) — Custom template processing code is adapted from alphafold_finetune.
  Motmaen, A. et al. "Peptide-binding specificity prediction using fine-tuned protein structure prediction networks." *PNAS* (2023).

## Citation

If you use TASM in your research, please cite:

> Calfee, E.\*, Ko, A., Mattox, D., Al-Asadi, H., Osborne, E., Hayes, T., Klinger, M., Noakes, M.T.\*, Howie, B., Robins, H. "TASM: Predicting TCR specificity by supervising a structural model with experimental activation data." Manuscript submitted (2026).
>
> \*Corresponding authors: ecalfee@adaptivebiotech.com, mnoakes@adaptivebiotech.com

## License

TASM source code is licensed under the Apache License, Version 2.0 (see the [LICENSE](LICENSE) file in this repository).

## Data and code availability

This code was developed and is run inside Adaptive Biotechnologies' internal research computing environment, and depends on internal libraries, infrastructure, and proprietary datasets that are not publicly available. As a result, this repo cannot be run standalone; it is provided to enable understanding and reimplementation of the algorithms.
 
## Intellectual Property Notice

Copyright © 2026 Adaptive Biotechnologies Corp. The availability of this source code should not be interpreted as a dedication to the public, disclosure, waiver, or license of any intellectual property rights other than the specific rights expressly granted under the applicable software license. Any descriptions, examples, test data, workflows, or documentation included in this repository are provided solely to support use of the software as permitted by the applicable license and should not be construed as granting any additional rights. The code, data, documentation, and other materials made available through this repository are intended solely for research uses as permitted under the licenses in this repository. They may not be used, directly or indirectly, to develop, validate, support, market, sell, license, or commercialize any product or service without prior written permission from Adaptive Biotechnologies, Corp.
 
## Questions

For questions about the methods or manuscript, please contact the corresponding author(s) listed in the preprint.
