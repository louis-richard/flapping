# Copyright 2020 Louis Richard
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

import numpy as np

from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc


def load_moments(tint, cfg, args):
    """
    Load FPI moments of the velocity distribution functions.

    If the option moments is set to "part" then use partial moments instead.
    """
    ic = np.arange(1, 5)

    suf = "fpi_{}_{}".format(cfg["data_rate"], cfg["level"])

    if cfg["moments"] == "partial":
        # index to split partial moments (from quasi-neutrality assumption)
        part_idx_i, part_idx_e = [cfg[f"part_idx_{s}"] for s in ["i", "e"]]

        # Load partial moments
        # number density
        part_n_i = [get_data("partNi_{}".format(suf), tint, i, args.verbose) for i in
                    ic[:-1]]
        part_n_e = [get_data("partNe_{}".format(suf), tint, i, args.verbose) for i in
                    ic[:-1]]

        # bulk velocity
        part_v_i = [get_data("partVi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_v_e = [get_data("partVe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # temperature tensor
        part_t_i = [get_data("partTi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_t_e = [get_data("partTe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # pressure tensor
        part_p_i = [get_data("partPi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_p_e = [get_data("partPe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # split partial moments
        # number density
        n_i = [part_n_i[i - 1][:, part_idx_i] for i in ic[:-1]]
        n_e = [part_n_e[i - 1][:, part_idx_e] for i in ic[:-1]]

        # bulk velocity
        v_i = [part_v_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        v_e = [part_v_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]

        # temperature tensor
        t_i = [part_t_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        t_e = [part_t_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]

        # pressure tensor
        p_i = [part_p_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        p_e = [part_p_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]
    elif cfg["moments"] == "full":
        # number density
        n_i = [get_data("Ni_{}".format(suf), tint, i, args.verbose) for i in ic[:-1]]
        n_e = [get_data("Ne_{}".format(suf), tint, i, args.verbose) for i in ic[:-1]]

        # bulk velocity
        v_i = [get_data("Vi_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        v_e = [get_data("Ve_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]

        # temperature tensor
        t_i = [get_data("Ti_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        t_e = [get_data("Te_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]

        # pressure tensor
        p_i = [get_data("Pi_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        p_e = [get_data("Pe_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
    else:
        raise ValueError("Invalid moments type")

    # Load spintone correction
    spintone_i = [get_data("STi_gse_{}".format(suf), tint, i, args.verbose) for i in
                  ic[:-1]]
    spintone_e = [get_data("STe_gse_{}".format(suf), tint, i, args.verbose) for i in
                  ic[:-1]]

    # remove spintone correction
    v_i = [v - spintone_i[i] for i, v in enumerate(v_i)]
    v_e = [v - spintone_e[i] for i, v in enumerate(v_e)]

    moments_i = [n_i, v_i, t_i, p_i]
    moments_e = [n_e, v_e, t_e, p_e]

    moments_i = [avg_4sc(moment) for moment in moments_i]
    moments_e = [avg_4sc(moment) for moment in moments_e]

    return moments_i, moments_e
