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
#
# Interface pLDDT calculation adapted from TCRmodel2 (Apache 2.0 License):
#   https://github.com/piercelab/tcrmodel2
#   Yin R, Ribeiro-Filho HV, Lin V, Gowthaman R, Cheung M, Pierce BG. (2023)
#   TCRmodel2: high-resolution modeling of T cell receptor recognition using
#   deep learning. Nucleic Acids Res, 51(W1):W569-W576.
#   https://doi.org/10.1093/nar/gkad356
# Modified by Adaptive Biotechnologies
def calc_iplddt(pdb_file):

    chn1 = "CDE"  # pMHC (openfold output will make pMHC all chain 'C')
    chn2 = "AB"  # TCR
    dis_cut = 4
    lowest = -1.00

    try:
        with open(pdb_file, "r") as file:
            pdb_lines = file.readlines()
    except IOError:
        sys.exit(f"unable to open file: {pdb_file}")

    chn1int = {char: 1 for char in chn1}
    chn2int = {char: 1 for char in chn2}

    dis_cutoff = float(dis_cut) ** 2

    chn1_int_plddt = {}
    chn2_int_plddt = {}

    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        if line[12] == "H" or line[13] == "H":
            continue
        res_num = line[22:27].strip()
        chn_id = line[21].strip()
        res_id = line[17:20].strip()
        atm_id = line[12:16].strip()
        plddt1 = float(line[60:66].strip())

        if chn1int.get(chn_id) == 1:
            x1 = float(line[30:38].strip())
            y1 = float(line[38:46].strip())
            z1 = float(line[46:54].strip())

            for line2 in pdb_lines:
                if not line2.startswith("ATOM"):
                    continue
                if line2[12] == "H" or line2[13] == "H":
                    continue

                chn2 = line2[21].strip()
                res_num2 = line2[22:27].strip()
                res_id2 = line2[17:20].strip()
                atm_id = line2[12:16].strip()
                plddt2 = float(line2[60:66].strip())

                if chn2int.get(chn2) == 1:
                    x2 = float(line2[30:38].strip())
                    y2 = float(line2[38:46].strip())
                    z2 = float(line2[46:54].strip())
                    dist = ((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2)
                    if dist < dis_cutoff:
                        chn1_int_plddt[f"{res_num}\t{chn_id}\t{res_id}"] = plddt1
                        chn2_int_plddt[f"{res_num2}\t{chn2}\t{res_id2}"] = plddt2

    weighted_sum = sum(chn1_int_plddt.values()) + sum(chn2_int_plddt.values())
    counts = len(chn1_int_plddt) + len(chn2_int_plddt)

    final_score = lowest if counts == 0 else weighted_sum / counts

    # counts: number of interacting residues from TCR and pMHC (added by Adaptive)
    return final_score, counts

