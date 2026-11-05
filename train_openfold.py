# Modified by Adaptive Biotechnologies
import argparse
import logging
import os
import sys
import json
import traceback
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning import seed_everything
import torch
try:
    import wandb
except ImportError:
    wandb = None
from deepspeed.utils import zero_to_fp32
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from sklearn.metrics import roc_auc_score

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule, OpenFoldMultimerDataModule
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.argparse_utils import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose, ModelCheckpointWandb
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
    get_num_atomic_contacts,
)
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)
from openfold.utils.logger import PerformanceLoggingCallback
from openfold.utils.environments import KubeflowEnvironment


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, args):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.is_multimer = self.config.globals.is_multimer

        self.loss = AlphaFoldLoss(config.loss)

        if config.model.freeze_trunk:
            if not config.model.heads.tm.enabled:
                raise ValueError("freeze_trunk requires tm head to be enabled")
            if not config.model.classifier.use_classifier:
                raise ValueError("freeze_trunk requires classifier to be enabled (use_classifier=True)")

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.aux_heads.tm.parameters():
                param.requires_grad = True
            for param in self.model.classifier.parameters():
                param.requires_grad = True

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                total = sum(p.numel() for p in self.model.parameters())
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                frozen = total - trainable
                print(f"freeze_trunk enabled: {trainable:,} trainable params, {frozen:,} frozen params (total: {total:,})")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(f"  trainable: {name} ({param.numel():,} params)")

        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )

        self.cached_weights = None
        self.last_lr_step = -1
        self.save_hyperparameters()

        # For computing AUROC on validation set
        self.preds = []  # gather predicted probs across GPUs and batches
        self.labels = []  # gather labels across GPUs and batches
        self.is_miras = []  # gather whether the sample is MIRA or not
        self.sample_indices = [] # gather sample indices; used to compute AUROC by validation group

        # Will be used to compute validation metrics by sample type
        self.num_pos_crystal_per_val_batch = args.num_pos_crystal_per_val_batch
        self.num_pos_predicted_per_val_batch = args.num_pos_predicted_per_val_batch
        self.num_pos_no_structure_per_val_batch = args.num_pos_no_structure_per_val_batch
        self.num_neg_no_structure_per_val_batch = args.num_neg_no_structure_per_val_batch
        self.val_sample_types = []
        if self.num_pos_crystal_per_val_batch:
            self.val_sample_types.append("positive_crystal")
        if self.num_pos_predicted_per_val_batch:
            self.val_sample_types.append("positive_predicted")
        if self.num_pos_no_structure_per_val_batch:
            self.val_sample_types.append("positive_no_structure")
        if self.num_neg_no_structure_per_val_batch:
            self.val_sample_types.append("negative_no_structure")

        # Create a mapping from validation group to sample indices (e.g. "A0201" -> [0,3,4])
        self.val_group_to_sample_indices = {}
        if args.val_mmcif_data_cache_path:
            val_data_cache = json.load(open(args.val_mmcif_data_cache_path, 'rb'))
            n_missing = 0
            for _, sample_info in val_data_cache.items():
                if "validation_groups" not in sample_info:
                    n_missing += 1
                    continue
                val_groups = sample_info["validation_groups"].split(",")
                for val_group in val_groups:
                    if val_group not in self.val_group_to_sample_indices:
                        self.val_group_to_sample_indices[val_group] = []
                    self.val_group_to_sample_indices[val_group].append(
                        sample_info["sample_index"]
                    )
            if n_missing > 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"{n_missing}/{len(val_data_cache)} validation cache entries missing "
                    f"'validation_groups'. Only overall AUROC (val/auroc_mira_only_all) "
                    f"will be logged. To enable per-group AUROC, add a 'validation_groups' "
                    f"field to each MIRA sample in the validation data cache JSON, e.g.: "
                    f'{{"sample_index": 0, "validation_groups": "A0201,all", ...}}'
                )

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                prog_bar=(loss_name == 'loss'),
                on_step=train,
                on_epoch=(not train),
                logger=True,
                sync_dist=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k,v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                prog_bar = (k == 'loss'),
                on_step=train,
                on_epoch=(not train),
                logger=True,
                sync_dist=True,
            )

        # Log fape, lddt_ca, num contacts by sample type in validation
        if phase == "val":
            # Get contact metrics (e.g. num contacts between TCR and peptide)
            contact_metrics = self.compute_contact_metrics(batch, outputs)

            for sample_type in self.val_sample_types:
                # log contact metrics
                for metric_name in contact_metrics.keys():
                    self.log_val_metric_by_sample_type(
                        batch=batch,
                        metric_dict=contact_metrics,
                        metric_name=metric_name,
                        sample_type=sample_type,
                    )

                # log fape and lddt_ca for positives with ground truth structure
                if sample_type in ["positive_crystal", "positive_predicted"]:
                    self.log_val_metric_by_sample_type(
                        batch=batch,
                        metric_dict=other_metrics,
                        metric_name="lddt_ca",
                        sample_type=sample_type,
                    )
                    self.log_val_metric_by_sample_type(
                        batch=batch,
                        metric_dict=loss_breakdown,
                        metric_name="fape",
                        sample_type=sample_type,
                    )


    def compute_contact_metrics(self, batch, outputs, dist_thresh=4.0):
        """Compute the number of atomic contacts between TCR and peptide."""
        coords = outputs["final_atom_positions"]
        atom_mask = batch["all_atom_mask"]
        asym_id = batch["asym_id_tcra_tcrb_pep_mhc"]

        tcr_asym_ids = [1, 2]
        peptide_asym_ids = [3]
        mhc_asym_ids = [4]
        pmhc_asym_ids = [3, 4]

        contact_metrics = {}
        num_contacts_tcr_peptide = get_num_atomic_contacts(
            coords,
            atom_mask,
            asym_id,
            tcr_asym_ids,
            peptide_asym_ids,
            dist_thresh=dist_thresh,
        )
        num_contacts_tcr_mhc = get_num_atomic_contacts(
            coords,
            atom_mask,
            asym_id,
            tcr_asym_ids,
            mhc_asym_ids,
            dist_thresh=dist_thresh,
        )
        num_contacts_tcr_pmhc = get_num_atomic_contacts(
            coords,
            atom_mask,
            asym_id,
            tcr_asym_ids,
            pmhc_asym_ids,
            dist_thresh=dist_thresh,
        )
        num_contacts_peptide_mhc = get_num_atomic_contacts(
            coords,
            atom_mask,
            asym_id,
            peptide_asym_ids,
            mhc_asym_ids,
            dist_thresh=dist_thresh,
        )

        contact_metrics["num_contacts_tcr_peptide"] = num_contacts_tcr_peptide
        contact_metrics["num_contacts_tcr_mhc"] = num_contacts_tcr_mhc
        contact_metrics["num_contacts_tcr_pmhc"] = num_contacts_tcr_pmhc
        contact_metrics["num_contacts_peptide_mhc"] = num_contacts_peptide_mhc
        contact_metrics["has_contact_tcr_peptide"] = (num_contacts_tcr_peptide > 0).float()
        contact_metrics["has_contact_tcr_mhc"] = (num_contacts_tcr_mhc > 0).float()
        contact_metrics["has_contact_peptide_mhc"] = (num_contacts_peptide_mhc > 0).float()

        return contact_metrics

    def log_val_metric_by_sample_type(
            self,
            batch,
            metric_dict,
            metric_name,
            sample_type,
        ):
        """Log the mean of the metric across all target sample types in validation.
        Args:
            batch: the current batch
            metric_dict: a dictionary containing the metric values
            metric_name: the name of the metric to log
            sample_type: 'positive_crystal', 'positive_predicted',
                'positive_no_structure', 'negative_no_structure'
        """
        if sample_type == "positive_crystal":
            target_binding_status = 1
            target_has_predicted_structure = 0
            target_has_true_structure = 1
            num_target_samples = self.num_pos_crystal_per_val_batch * self.num_val_batches
        elif sample_type == "positive_predicted":
            target_binding_status = 1
            target_has_predicted_structure = 1
            target_has_true_structure = 0
            num_target_samples = self.num_pos_predicted_per_val_batch * self.num_val_batches
        elif sample_type == "positive_no_structure":
            target_binding_status = 1
            target_has_predicted_structure = 0
            target_has_true_structure = 0
            num_target_samples = self.num_pos_no_structure_per_val_batch * self.num_val_batches
        elif sample_type == "negative_no_structure":
            target_binding_status = 0
            target_has_predicted_structure = 0
            target_has_true_structure = 0
            num_target_samples = self.num_neg_no_structure_per_val_batch * self.num_val_batches
        else:
            raise ValueError("Invalid target sample type")

        if (batch["binding_status"] == target_binding_status and
            batch["has_predicted_structure"] == target_has_predicted_structure and
            batch["has_true_structure"] == target_has_true_structure):
            # Divide metric by number of target samples in the validation set.
            # v will be summed across GPUs in self.log() to compute the mean metric.
            v = torch.mean(metric_dict[metric_name].float()) / max(num_target_samples, 1)
        else:
            # Exclude metric from sum if the structure type is not the target type
            v = 0

        self.log(
            f"val/{metric_name}_{sample_type}",
            v,
            on_step=False,
            on_epoch=True,  # metric is aggregated across the entire validation set
            logger=True,
            sync_dist=True,
            reduce_fx='sum',
        )

    def on_fit_start(self):
        # Get the number of batches in each validation
        # Will be used to compute val metrics by structure type
        self.num_val_batches = len(self.trainer.datamodule.val_dataloader())
        if self.trainer.global_rank == 0:
            print(f"Number of batches in each validation: {self.num_val_batches}")

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # convert asym_id to 4-piece for loss computation
        batch['asym_id'] = batch['asym_id_tcra_tcrb_pep_mhc'].clone()
        ground_truth['asym_id'] = ground_truth['asym_id_tcra_tcrb_pep_mhc'].clone()
        outputs['asym_id'] = outputs['asym_id_tcra_tcrb_pep_mhc'].clone()
        # convert entity_id to 4-piece
        batch['entity_id'] = batch['asym_id_tcra_tcrb_pep_mhc'].clone()
        ground_truth['entity_id'] = ground_truth['asym_id_tcra_tcrb_pep_mhc'].clone()
        # convert residue_index to 4-piece (i.e. mhc should start at index 0)
        # TODO mhc index starts from peptide_len + 200. Not sure about class II. 
        mhc_start_idx = torch.sum(ground_truth["asym_id_tcra_tcrb_pep_mhc"] == 3) + 200
        batch['residue_index'][batch['asym_id_tcra_tcrb_pep_mhc'] >= 4] -= mhc_start_idx

        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        if args.in_debug_mode and self.trainer.global_rank == 0:
            import pickle
            debug_output_dir = os.path.join(os.getcwd(), "debug_files")
            os.makedirs(debug_output_dir, exist_ok=True)
            with open(os.path.join(debug_output_dir, "batch.pkl"), 'wb') as f:
                pickle.dump(batch, f, protocol=4)
            with open(os.path.join(debug_output_dir, "ground_truth.pkl"), "wb") as f:
                pickle.dump(ground_truth, f, protocol=4)
            with open(os.path.join(debug_output_dir, "outputs.pkl"), 'wb') as f:
                pickle.dump(outputs, f, protocol=4)
            with open(os.path.join(debug_output_dir, "loss.pkl"), 'wb') as f:
                pickle.dump(loss, f, protocol=4)
            with open(os.path.join(debug_output_dir, "loss_breakdown.pkl"), 'wb') as f:
                pickle.dump(loss_breakdown, f, protocol=4)
            print(f"Saved debug files to {debug_output_dir}")
            sys.exit()

        # TODO may not be necessary to convert back to 3-piece
        batch['asym_id'][batch['asym_id'] >= 4] = 3
        ground_truth['asym_id'][ground_truth['asym_id'] >= 4] = 3
        outputs['asym_id'][outputs['asym_id'] >= 4] = 3
        batch['entity_id'][batch['entity_id'] >= 4] = 3
        ground_truth['entity_id'][ground_truth['entity_id'] >= 4] = 3
        batch['residue_index'][batch['asym_id_tcra_tcrb_pep_mhc'] >= 4] += mhc_start_idx

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        '''Called after optimization step but before zero_grad() is called'''
        self.ema.update(self.model)

        # Log classifier params after optimization step
        for name, param in self.trainer.strategy.model.named_parameters():
            if "classifier" in name:
                self.log(f"classifier_params/{name}", param.item(), on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        batch["use_clamped_fape"] = 0.

        # convert asym_id to 4-piece for loss computation
        batch['asym_id'] = batch['asym_id_tcra_tcrb_pep_mhc'].clone()
        ground_truth['asym_id'] = ground_truth['asym_id_tcra_tcrb_pep_mhc'].clone()
        outputs['asym_id'] = outputs['asym_id_tcra_tcrb_pep_mhc'].clone()
        # convert entity_id to 4-piece
        batch['entity_id'] = batch['asym_id_tcra_tcrb_pep_mhc'].clone()
        ground_truth['entity_id'] = ground_truth['asym_id_tcra_tcrb_pep_mhc'].clone()
        # convert residue_index to 4-piece (i.e. mhc should start at index 0)
        # TODO mhc index starts from peptide_len + 200. Not sure about class II. 
        mhc_start_idx = torch.sum(ground_truth["asym_id_tcra_tcrb_pep_mhc"] == 3) + 200
        batch['residue_index'][batch['asym_id_tcra_tcrb_pep_mhc'] >= 4] -= mhc_start_idx


        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Gather predictions and labels for AUROC
        pred = torch.sigmoid(outputs["binding_logits"]).squeeze(0)
        is_mira = batch["has_true_structure"] == 0
        self.preds.append(pred)
        self.labels.append(batch["binding_status"])
        self.is_miras.append(is_mira)
        if "sample_index" in batch:
            self.sample_indices.append(batch["sample_index"])
        else:  # If no sample_index, append -1 (for backward compatibility with datasets with no sample_index)
            self.sample_indices.append(torch.tensor([-1], device=batch["aatype"].device))

        # TODO may not be necessary to convert back to 3-piece
        batch['asym_id'][batch['asym_id'] >= 4] = 3
        ground_truth['asym_id'][ground_truth['asym_id'] >= 4] = 3
        outputs['asym_id'][outputs['asym_id'] >= 4] = 3
        batch['entity_id'][batch['entity_id'] >= 4] = 3
        ground_truth['entity_id'][ground_truth['entity_id'] >= 4] = 3
        batch['residue_index'][batch['asym_id_tcra_tcrb_pep_mhc'] >= 4] += mhc_start_idx

        self._log(loss_breakdown, batch, outputs, train=False)

    def log_auroc(self, preds, labels, sample_indices, target_val_group):
        """Log AUROC for the target group id.
        Args:
            preds: tensor of predicted probabilities
            labels: tensor of ground truth labels
            sample_indices: tensor of sample indices
            target_val_group: validation group to compute AUROC for
        """
        # True if sample_index is in the target validation group
        mask = [
            sample_index in self.val_group_to_sample_indices[target_val_group]
            for sample_index in sample_indices
        ]
        preds_subset = preds[mask]
        labels_subset = labels[mask]

        # Require at least 10 positive and 10 negatives
        if (labels_subset == 1).sum() >= 10 and (labels_subset == 0).sum() >= 10:
            auroc = roc_auc_score(
                labels_subset.to(torch.float32).cpu().numpy(),
                preds_subset.to(torch.float32).cpu().numpy()
            )
        else:  # if no samples meet target criteria, log AUROC to be -1
            logging.warning(
                (
                    f"{target_val_group} has less than 10 positive or negative samples "
                    "in this validation set. Logging AUROC as -1."
                )
            )
            auroc = -1

        self.log(f"val/auroc_mira_only_{target_val_group}", auroc)

    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

        # Gather metrics to compute AUROC on the entire validation set
        preds = torch.cat(self.preds)  # list of tensors -> 1 tensor
        labels = torch.cat(self.labels)
        is_miras = torch.cat(self.is_miras)
        sample_indices = torch.tensor(self.sample_indices)
        preds = self.all_gather(preds).flatten()  # aggregate across gpus
        labels = self.all_gather(labels).flatten()
        is_miras = self.all_gather(is_miras).flatten()
        sample_indices = self.all_gather(sample_indices).flatten()

        # Filter preds and labels for MIRA samples only
        mira_preds = preds[is_miras == 1]
        mira_labels = labels[is_miras == 1]
        mira_sample_indices = sample_indices[is_miras == 1]

        # Log overall AUROC across all MIRA samples
        if (mira_labels == 1).sum() >= 10 and (mira_labels == 0).sum() >= 10:
            auroc_all = roc_auc_score(
                mira_labels.to(torch.float32).cpu().numpy(),
                mira_preds.to(torch.float32).cpu().numpy()
            )
        else:
            logging.warning(
                "Less than 10 positive or negative MIRA samples in validation set. "
                "Logging overall AUROC as -1."
            )
            auroc_all = -1
        self.log("val/auroc_mira_only", auroc_all)

        # Log AUROC for each validation group
        for target_val_group in self.val_group_to_sample_indices.keys():
            self.log_auroc(
                mira_preds,
                mira_labels,
                mira_sample_indices,
                target_val_group=target_val_group
            )

        # Clear the lists for the next validation epoch
        self.preds.clear()
        self.labels.clear()
        self.is_miras.clear()
        self.sample_indices.clear()

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            max_lr=self.config.lr_schedule.max_lr,
            warmup_no_steps=self.config.lr_schedule.warmup_no_steps,
            start_decay_after_n_steps=self.config.lr_schedule.start_decay_after_n_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        # Current seed must be different from previous seed so that we don't repeat train datasets
        if 'seed' in checkpoint and checkpoint['seed'] == self.trainer.datamodule.batch_seed:
            raise ValueError(f"Must provide a new seed to resume training. Do not reuse seed {checkpoint['seed']}")

        ema = checkpoint["ema"]
        if(not self.model.template_config.enabled):
            ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["seed"] = self.trainer.datamodule.batch_seed

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_from_jax(self, jax_path):
        model_basename = os.path.splitext(
                os.path.basename(
                    os.path.normpath(jax_path)
                )
        )[0]
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(
                self.model, jax_path, version=model_version
        )

def get_model_state_dict_from_ds_checkpoint(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, 'latest')
    if os.path.isfile(latest_path):
        with open(latest_path, 'r') as fd:
            tag = fd.read().strip()
    else:
        raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)
    _DS_CHECKPOINT_VERSION = 2  # based on manual parsing of checkpoint files
    state_file = zero_to_fp32.get_model_state_file(ds_checkpoint_dir, _DS_CHECKPOINT_VERSION)
    return torch.load(state_file)

def _write_crash_log(output_dir):
    if torch.distributed.is_initialized():
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        crash_file = os.path.join(output_dir, f"crash_rank{global_rank}.log")
        with open(crash_file, 'w') as f:
            f.write(traceback.format_exc())
        logging.error(f"Crash log written to {crash_file}")
    except Exception:
        pass


def main(args):
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    if(args.seed is not None):
        seed_everything(args.seed, workers=True)

    is_low_precision = args.precision in [
        "bf16-mixed", "16", "bf16", "16-true", "16-mixed", "bf16-mixed"]

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=is_low_precision,
    ) 
    if args.experiment_config_json: 
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)

    model_module = OpenFoldWrapper(config, args)

    if args.resume_from_ckpt:
        if args.resume_model_weights_only:
            # Load the checkpoint
            if os.path.isdir(args.resume_from_ckpt):
                # Convert deepspeed checkpoint to a standard state_dict
                if not os.path.exists("converted_checkpoint.pt"):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        args.resume_from_ckpt, "converted_checkpoint.pt"
                    )
                sd = torch.load("converted_checkpoint.pt")
            else:
                sd = torch.load(args.resume_from_ckpt)
            # Process the state dict
            if 'module' in sd:
                # Keys in sd may not have 'module.' prefix, so need
                # to check the keys first before deleting the prefix.
                new_sd = {}
                for key, val in sd["module"].items():
                    if key.startswith("module."):
                        key = key[len("module."):]
                    new_sd[key] = val
                import_openfold_weights_(model=model_module, state_dict=new_sd)
            elif 'state_dict' in sd:
                import_openfold_weights_(
                    model=model_module, state_dict=sd['state_dict'])
            else:
                # Loading from pre-trained model
                # Keys in sd can already have 'model.' prefix, so need
                # to check the keys first before adding the prefix.
                new_sd = {}
                for key, val in sd.items():
                    if not key.startswith('model.'):
                        key = 'model.' + key
                    new_sd[key] = val
                import_openfold_weights_(model=model_module, state_dict=new_sd)
            logging.info("Successfully loaded model weights...")

            # Set EMA weights
            if "ema" in sd:
                model_module.ema.load_state_dict(sd["ema"])
                logging.info("Successfully loaded EMA weights from checkpoint.")
            else:  # If no ema weights, use model weights instead
                model_module.ema.reinit_params(model_module.model)
                logging.info("No EMA weights found in checkpoint. "
                             "Initializing EMA weights from model weights.")

        else:  # Loads a checkpoint to start from a specific time step
            if os.path.isdir(args.resume_from_ckpt):
                sd = get_model_state_dict_from_ds_checkpoint(args.resume_from_ckpt)
            else:
                sd = torch.load(args.resume_from_ckpt)
            last_global_step = int(sd['global_step'])
            model_module.resume_last_lr_step(last_global_step)
            logging.info("Successfully loaded last lr step...")

            # Do not load model and EMA parameters since a full checkpoint load
            # is handled by Lightning's trainer.

    if args.resume_from_jax_params:
        model_module.load_from_jax(args.resume_from_jax_params)
        # re-initialize EMA parameters with imported weights
        model_module.ema.reinit_params(model_module.model)
        logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")
 
    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    if "multimer" in args.config_preset:
        data_module = OpenFoldMultimerDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )
    else:
        data_module = OpenFoldDataModule(
            config=config.data, 
            batch_seed=args.seed,
            **vars(args)
        )

    callbacks = []
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            every_n_epochs=1,
            auto_insert_metric_name=False,
            save_top_k=-1,
        )
        callbacks.append(mc)

    if (args.checkpoint_every_validation):
        checkpoint_callback = ModelCheckpointWandb(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            save_top_k=-1,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    is_rank_zero = args.mpi_plugin and (int(os.environ.get("PMI_RANK")) == 0)
    if(args.wandb):
        if args.mpi_plugin and is_rank_zero:
            wandb_init_dict = dict(
                name=args.experiment_name,
                project=args.wandb_project,
                id=args.wandb_id,
                dir=args.output_dir,
                resume="allow",
                anonymous=None,
                entity=args.wandb_entity
            )
            wandb.run = wandb.init(**wandb_init_dict)

        # WandbLogger won't write to output_dir if the directory doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)
        # Give access to WandbLogger to upload files to wandb
        checkpoint_callback.set_wandb_logger(wdb_logger)
        data_module.set_wandb_logger(wdb_logger)

    if args.kubeflow:
        cluster_environment = KubeflowEnvironment()
    elif args.mpi_plugin:
        cluster_environment = MPIEnvironment()
    else:
        cluster_environment = None
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedStrategy(
            config=args.deepspeed_config_path,
            cluster_environment=cluster_environment,
        )
        if(args.wandb and is_rank_zero):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False,
                               cluster_environment=cluster_environment)
    else:
        strategy = None
 
    if(args.wandb and is_rank_zero):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")

    trainer_kws = ['num_nodes', 'precision', 'max_epochs', 'log_every_n_steps',
                   'flush_logs_ever_n_steps', 'num_sanity_val_steps', 'reload_dataloaders_every_n_epochs',
                   'accumulate_grad_batches', 'val_check_interval']
    trainer_args = {k: v for k, v in vars(args).items() if k in trainer_kws}
    trainer_args.update({
        'default_root_dir': args.output_dir,
        'devices': args.gpus,
        'strategy': strategy,
        'callbacks': callbacks,
        'logger': loggers,
        # turn off default distributed sampler so we can use a custom one
        'use_distributed_sampler': False,
    })
    trainer = pl.Trainer(**trainer_args)

    if (args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    try:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    except BaseException:
        _write_crash_log(args.output_dir)
        raise


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--train_pos_mmcif_data_cache_path", type=str, default=None,
        help="Path to the json file which records all the information of positives used during training"
    )
    parser.add_argument(
        "--train_neg_mmcif_data_cache_path", type=str, default=None,
        help="Path to the json file which records all the information of negatives used during training"
    )
    parser.add_argument(
        "--train_crystal_mmcif_data_cache_path", type=str, default=None,
        help="Path to the json file which records all the information of crystal structures used during training"
    )
    parser.add_argument(
        "--use_single_seq_mode", type=str, default=False,
        help="Use single sequence embeddings instead of MSAs."
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--val_mmcif_data_cache_path", type=str, default=None,
        help="path to the json file which records all the information of mmcif structures used during validation"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--pmhc_templates_dir", type=str, default=None,
        help="""Directory containing pMHC PDB template files (e.g.
                7kgo.pmhc.pdb). Used to resolve template paths from
                alignment TSV files during pMHC feature generation."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch"""
    )
    parser.add_argument(
        "--checkpoint_every_validation", action="store_true", default=False,
        help="Whether to checkpoint at the end of every validation"
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_jax_params", type=str, default=None,
        help="""Path to an .npz JAX parameter file with which to initialize the model"""
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--use_custom_sampler", action="store_true", default=False,
        help=(
            "Use a custom sampler that selects specific sample composition "
            "for each batch."
        )
    )
    parser.add_argument(
        "--num_pos_crystal_per_train_batch", type=int, default=None,
        help=("Number of positive samples with crystal structures per training batch")
    )
    parser.add_argument(
        "--num_pos_predicted_per_train_batch", type=int, default=None,
        help=("Number of positive samples with predicted structures per training batch")
    )
    parser.add_argument(
        "--num_pos_no_structure_per_train_batch", type=int, default=None,
        help=("Number of positive samples with no structures per training batch")
    )
    parser.add_argument(
        "--num_neg_no_structure_per_train_batch", type=int, default=None,
        help=("Number of negative samples with no structures per training batch")
    )
    parser.add_argument(
        "--num_pos_crystal_per_val_batch", type=int, default=None,
        help=("Number of positive samples crystal structures per validation batch")
    )
    parser.add_argument(
        "--num_pos_predicted_per_val_batch", type=int, default=None,
        help=("Number of positive samples with predicted structurs per validation batch")
    )
    parser.add_argument(
        "--num_pos_no_structure_per_val_batch", type=int, default=None,
        help=("Number of positive samples with no structures per validation batch")
    )
    parser.add_argument(
        "--num_neg_no_structure_per_val_batch", type=int, default=None,
        help=("Number of negative samples with no structures per validation batch")
    )
    parser.add_argument(
        "--num_samples_per_validation", type=int, default=None,
        help=("Number of samples to process per validation")
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help='Number of GPUs to use per node. Also used for determining optimal strategy and effective batch size.'
    )
    parser.add_argument("--mpi_plugin", action="store_true", default=False,
                        help="Whether to use MPI for parallele processing")
    parser.add_argument("--kubeflow", action="store_true", default=False,
                        help="Whether kubeflow is used. Determines cluster_environment.")
    parser.add_argument("--in_debug_mode", action="store_true", default=False,
                        help="Stop when debug flag is reached.") # TODO remove after testing

    trainer_group = parser.add_argument_group(
        'Arguments to pass to PyTorch Lightning Trainer')
    trainer_group.add_argument(
        "--num_nodes", type=int, default=1,
    )
    trainer_group.add_argument(
        "--precision", type=str, default='bf16-mixed',
        help='Sets precision, lower precision improves runtime performance.',
    )
    trainer_group.add_argument(
        "--max_epochs", type=int, default=1,
    )
    trainer_group.add_argument(
        "--log_every_n_steps", type=int, default=25,
    )
    trainer_group.add_argument(
        "--flush_logs_every_n_steps", type=int, default=5,
    )
    trainer_group.add_argument(
        "--num_sanity_val_steps", type=int, default=0,
    )
    trainer_group.add_argument(
        "--val_check_interval", type=int, default=None,
        help="Number of training steps between validations"
    )
    trainer_group.add_argument(
        "--reload_dataloaders_every_n_epochs", type=int, default=None,
        help=(
            "Recreate train and validation DataLoaders every N epochs; 0 keeps "
            "loaders and workers alive. If not set (None), defaults to 0 with "
            "--use_custom_sampler, otherwise 1."
        ),
    )

    trainer_group.add_argument("--accumulate_grad_batches", type=int, default=1,
                               help="Accumulate gradients over k batches before next optimizer step.")

    args = parser.parse_args()

    # Keep dataloader workers alive with custom sampler to avoid re-forking
    if args.reload_dataloaders_every_n_epochs is None:
        args.reload_dataloaders_every_n_epochs = 0 if args.use_custom_sampler else 1

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if(args.resume_from_jax_params is not None and args.resume_from_ckpt is not None):
        raise ValueError("Choose between loading pretrained Jax-weights and a checkpoint-path")

    if args.val_check_interval and args.val_check_interval % args.accumulate_grad_batches != 0:
        raise ValueError("val_check_interval has to be a multiple of accumulate_grad_batches. "
                         "Otherwise validation loss will not be plotted correctly.")

    # check sample composition and allowed datasets if using custom sampler
    if args.use_custom_sampler:
        if args.distillation_data_dir is not None:
            raise ValueError("Custom sampler is not supported for distillation data")

        if (
            args.num_pos_crystal_per_train_batch is None
            or args.num_pos_predicted_per_train_batch is None
            or args.num_pos_no_structure_per_train_batch is None
            or args.num_neg_no_structure_per_train_batch is None
        ):
            raise ValueError("Must provide num_pos_crystal_per_train_batch, "
                             "num_pos_predicted_per_train_batch, "
                             "num_pos_no_structure_per_train_batch, "
                             "num_neg_no_structure_per_train_batch "
                             "if using the custom sampler")

        if args.val_data_dir is not None:
            if (
                args.num_pos_crystal_per_val_batch is None
                or args.num_pos_predicted_per_val_batch is None
                or args.num_pos_no_structure_per_val_batch is None
                or args.num_neg_no_structure_per_val_batch is None
            ):
                raise ValueError("Must provide num_pos_crystal_per_val_batch, "
                                 "num_pos_predicted_per_val_batch, "
                                 "num_pos_no_structure_per_val_batch, "
                                 "num_neg_no_structure_per_val_batch, "
                                 "if using the custom sampler")

    main(args)
