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

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyrfu.mms import get_data
from pyrfu.plot import plot_line
from pyrfu.pyrf import c_4_j, norm, new_xyz, c_4_grad

from spf import (lmn_cs, load_moments, remove_bz_offset, dec_temperature,
                 make_labels)


def grad_ni(r_mms, tint, cfg):
    r"""Compute density gradients


    Parameters
    ----------
    r_mms : list
        Spacecraft positions.

    tint : list
        Time interval.

    cfg : dict

    Returns
    -------
    grad_n_i : xarray.DataArray
        Density gradient time series.

    """

    ic = np.arange(1, 5)

    if cfg["fpi"]["moments"] == "partial":
        part_idx_i = cfg["fpi"]["part_idx_i"]
        part_n_i_mms = [get_data("partni_fpi_fast_l2", tint, i,
                                 data_path=cfg["data_path"]) for i in ic]
        n_i_mms = [part_n_i_mms[i - 1][:, part_idx_i] for i in ic]
    elif cfg["fpi"]["moments"] == "full":
        n_i_mms = [get_data("ni_fpi_fast_l2", tint, i,
                            data_path=cfg["data_path"]) for i in ic]
    else:
        raise ValueError("Invalid moments type")

    grad_n_i = 1e3 * c_4_grad(r_mms, n_i_mms)

    return grad_n_i


def main(args):
    """main function
    """

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tints"]["flapping"]

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute magnetic field at the center of mass of the tetrahedron
    j_xyz, _, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9

    # Load moments of the velocity distribution functions
    moments_i, moments_e = load_moments(tint, cfg["fpi"], args,
                                        cfg["data_path"])

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_e)

    grad_n_i = grad_ni(r_mms, tint, cfg)

    # Compute new coordinates system
    lmn = lmn_cs(b_xyz, moments_i[1])

    # Transform fields to MVA frame
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    grad_n_i_lmn = new_xyz(grad_n_i, lmn)

    # Transform moments
    moments_lmn_i = [moments_i[0], None, None, None]
    moments_lmn_e = [moments_e[0], None, None, None]

    moments_lmn_i[1:] = [new_xyz(mom, lmn) for mom in moments_i[1:]]
    moments_lmn_e[1:] = [new_xyz(mom, lmn) for mom in moments_e[1:]]

    # Unpack number density
    n_i, n_e = [moments_lmn_i[0], moments_lmn_e[0]]

    # Unpack bulk velocity
    v_lmn_i, v_lmn_e = [moments_lmn_i[1], moments_lmn_e[1]]

    # Plot
    fig, axs = plt.subplots(6, **cfg["figure"]["main"])
    fig.subplots_adjust(**cfg["figure"]["subplots"])

    # B GSE
    plot_line(axs[0], b_lmn)
    plot_line(axs[0], norm(b_lmn))
    axs[0].set_ylim([-14, 14])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$", "$|B|$"],
                  **cfg["figure"]["legend"])
    axs[0].grid(True, which="both")

    # ViM, VeM, B_L
    plot_line(axs[1], v_lmn_i[:, 1], color="tab:blue", label="Ions")
    plot_line(axs[1], v_lmn_e[:, 1], color="tab:red", label="Electrons")
    axs[1].set_ylim([-420, 420])
    axs[1].set_yticks([-300, 0, 300])
    axs[1].set_ylabel("$V_M$ [km s$^{-1}$]")
    axs[1].legend(**cfg["figure"]["legend"])
    axs[1].grid(True, which="both")

    # ViN, VeN, B_L
    plot_line(axs[2], v_lmn_i[:, 2], color="tab:blue", label="Ions")
    plot_line(axs[2], v_lmn_e[:, 2], color="tab:red", label="Electrons")
    axs[2].set_ylim([-420, 420])
    axs[2].set_yticks([-300, 0, 300])
    axs[2].set_ylabel("$V_N$ [km s$^{-1}$]")
    axs[2].legend(**cfg["figure"]["legend"])
    axs[2].grid(True, which="both")

    # Ni, Ne, B_L
    plot_line(axs[3], n_i, color="tab:blue")
    axs[3].set_ylim([0.06, 0.34])
    axs[3].set_ylabel("$n_i$ [cm$^{-3}$]")
    axs[3].grid(True, which="both")

    plot_line(axs[4], grad_n_i_lmn[:, 2],
              color="tab:blue", label="$(\\nabla n_i)_N$")
    axs[4].set_ylim([-.84, .84])
    axs[4].set_yticks([-.6, 0, .6])
    axs[4].legend(**cfg["figure"]["legend"])
    axs[4].set_ylabel("$\\nabla n_i$ [m$^{-4}$]")

    # JM, JN, B_L
    plot_line(axs[5], j_lmn[:, 1], color="tab:green", label="$J_M$")
    plot_line(axs[5], j_lmn[:, 2], color="tab:red", label="$J_N$")
    axs[5].set_ylim([-28, 28])
    axs[5].set_ylabel("$J$ [nA m$^{-2}$]")
    axs[5].legend(**cfg["figure"]["legend"])
    axs[5].grid(True, which="both")

    # Add magnetic field to all axes
    for ax in axs[1:]:
        axb = ax.twinx()
        plot_line(axb, b_lmn[:, 0], color="darkgrey")
        axb.set_ylim([-14, 14])
        axb.set_ylabel("$B_L$ [nT]")

    axs[-1].set_xlim(mdates.date2num(tint))

    fig.align_ylabels(axs)

    # Add panels labels
    labels_pos = [0.02, 0.85]
    _ = make_labels(axs, labels_pos)

    if args.figname:
        fig.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--figname",
                        help="Path and name of the figure to save with extension.",
                        type=str, default="")
    parser.add_argument("--config",
                        help="Path to (.yml) config file.",
                        type=str, required=True)

    main(parser.parse_args())
