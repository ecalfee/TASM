# Copyright 2021 AlQuraishi Laboratory
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
# Modified by Adaptive Biotechnologies
import torch


def drmsd(structure_1, structure_2, mask=None):
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d ** 2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd ** 2
    if(mask is not None):
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if(mask is not None):
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)
    
    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])


def get_atomic_coordinates(coords, atom_mask, asym_id, target_asym_id):
    """Get xyz coordinates of all the atoms for the target asym_id
    Args:
        coords (tensor[batch_size, n_residue, 37, 3]): xyz coordinates of the
            atoms in each residue. Coordinates are given for all 37 possible atoms.
            Coordinates for atoms not present in the residue are set to (0,0,0).
        atom_mask (tensor[batch_size, n_residue, 37]): Indicates which of the
            37 possible atoms are present in each residue
        asym_id (tensor[batch_size, n_residue]): ID of each residue
            (1=alpha, 2=beta, 3=peptide, 4=mhca)
        target_asym_id (int): Which asym_id to target
            (1=alpha, 2=beta, 3=peptide, 4=mhca)
    """
    # Subset coords and atom_mask to the target asym_id
    # Assume batch size is always 1 and select the first sample in the batch
    residue_mask = (asym_id[0] == target_asym_id)
    coords_for_target_asym_id = coords[0, residue_mask, :]
    atom_mask_for_target_asym_id = atom_mask[0, residue_mask, :]

    # For each residue, get the xyz coordinates of all the atoms in the residue
    atom_coords_list = []
    for residue_i in range(coords_for_target_asym_id.shape[0]):
        atom_mask_for_this_residue = (
            atom_mask_for_target_asym_id[residue_i] == 1
        )  # used to mask out atoms that are not present in this residue
        atom_coords = coords_for_target_asym_id[
            residue_i, atom_mask_for_this_residue, :
        ]  # contains xyz coords for all the atoms in this residue
        atom_coords_list.append(atom_coords)

    atom_coords_tensor = torch.cat(atom_coords_list, dim=0)
    return atom_coords_tensor


def get_num_atomic_contacts(
    coords, atom_mask, asym_id, asym_id_group1, asym_id_group2, dist_thresh=4.0
):
    """Get the number of atomic contacts between two groups of atoms.
    Args:
        coords (tensor[batch_size, n_residue, 37, 3]): xyz coordinates of the
            atoms in each residue. Values given for all 37 possible atoms.
        atom_mask (tensor[batch_size, n_residue, 37]): Indicates which of the
            37 possible atoms are present in each residue
        asym_id (tensor[batch_size, n_residue]): ID of each residue
            (1=alpha, 2=beta, 3=peptide, 4=mhca)
        asym_id_group1 (list[int]): List of asym_ids for the first group of
            atoms. E.g. [3] for peptide
        asym_id_group2 (list[int]): List of asym_ids for the second group of
            atoms. E.g. [1, 2] for alpha and beta
        dist_thresh (float): Distance threshold for considering two atoms as
            in contact (4.0 Angstroms by default)
    """
    # get atomic coordinates for asym_ids in group 1
    temp1 = []
    for target_asym_id in asym_id_group1:
        atomic_coords = get_atomic_coordinates(
            coords, atom_mask, asym_id, target_asym_id
        ).double()
        temp1.append(atomic_coords)
    atomic_coords_group1 = torch.cat(temp1, dim=0)  # shape: (n_atoms_group1, 3)

    # get atomic coordinates for asym_ids in group 2
    temp2 = []
    for target_asym_id in asym_id_group2:
        atomic_coords = get_atomic_coordinates(
            coords, atom_mask, asym_id, target_asym_id
        ).double()
        temp2.append(atomic_coords)
    atomic_coords_group2 = torch.cat(temp2, dim=0)  # shape: (n_atoms_group2, 3)

    # pairwise distances between atoms in group1 and atoms in group 2
    # Returns L2 distance matrix of shape (n_atoms_group1, n_atoms_group2)
    dists = torch.cdist(atomic_coords_group1, atomic_coords_group2)

    num_contacts = (dists < dist_thresh).sum()

    return num_contacts
