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

"""Reproduces the Figure 1 in Richard et al. 2021.
@author: Louis Richard
"""

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astropy import constants
from pyrfu.mms import get_data, rotate_tensor
from pyrfu.pyrf import avg_4sc, cotrans
from pyrfu.plot import zoom, plot_line, plot_spectr, set_rlabel

from spf import load_moments, load_def_omni, make_labels, span_tint


def main(args):
    """main function
    """
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint_over = cfg["tints"]["overview"]
    tint_flap = cfg["tints"]["flapping"]

    mms_ids = np.arange(1, 5)

    # Load spacecraft position
    r_mms = [get_data("R_gsm", tint_over, i,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gsm_{}".format(suf_b), tint_over, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Earth radius
    r_earth = constants.R_earth.value * 1e-3

    # Spacecraft position and magnetic field at the center of mass of the
    # tetrahedron
    r_xyz = avg_4sc(r_mms) / r_earth
    b_xyz = avg_4sc(b_mms)

    # Load moments of the velocity distribution function
    moments_i, moments_e = load_moments(tint_over, cfg["fpi"], args,
                                        cfg["data_path"])

    # Load omni directional energy flux
    def_omni = load_def_omni(tint_over, cfg["fpi"], cfg["data_path"])

    # Unpack moments
    v_xyz_i, t_xyz_i = moments_i[1:3]
    _, t_xyz_e = moments_e[1:3]

    v_gsm_i = cotrans(v_xyz_i, "gse>gsm")

    # Unpack energy flux
    def_omni_i, def_omni_e = def_omni

    # Compute temperature tensor in field aligned coordinates
    t_xyzfac_i = rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")
    t_xyzfac_e = rotate_tensor(t_xyz_e, "fac", b_xyz, "pp")

    # Get parallel, perpendicular and total temperature
    t_para_i, t_perp_i = [t_xyzfac_i[:, 0, 0], t_xyzfac_i[:, 1, 1]]
    t_para_e, t_perp_e = [t_xyzfac_e[:, 0, 0], t_xyzfac_e[:, 1, 1]]

    # Figure options
    legend_options = dict(frameon=True, ncol=3)
    spectr_options = dict(yscale="log", cscale="log", cmap="Spectral_r")


    # Plot
    fig = plt.figure(**cfg["figure"]["main"])
    gsp1 = fig.add_gridspec(12, 1, **cfg["figure"]["gridspec"])

    gsp10 = gsp1[1:9].subgridspec(4, 1, hspace=0)
    gsp11 = gsp1[10:].subgridspec(1, 1, hspace=0)

    # Create axes in the grid spec
    axs10 = [fig.add_subplot(gsp10[i]) for i in range(4)]
    axs11 = [fig.add_subplot(gsp11[i]) for i in range(1)]

    # Magnetic field
    plot_line(axs10[0], b_xyz)
    axs10[0].legend(["$B_x$", "$B_y$", "$B_z$"],
                    loc="upper right", **legend_options)
    axs10[0].set_ylim([-23, 23])
    axs10[0].set_ylabel("$B$ [nT]")

    # Ions bulk velocity
    plot_line(axs10[1], v_gsm_i)
    axs10[1].legend(["$V_{ix}$", "$V_{iy}$", "$V_{iz}$"],
                    loc="upper right", **legend_options)
    axs10[1].set_ylabel("$V_i$ [km s$^{-1}$]")
    axs10[1].grid(True, which="both")

    # Ions energy spectrum
    axs10[2], caxs02 = plot_spectr(axs10[2], def_omni_i,
                                   clim=[1e4, 1e6], **spectr_options)
    plot_line(axs10[2], t_perp_i, color="k")
    plot_line(axs10[2], t_para_i, color="tab:blue")
    axs10[2].legend(["$T_{i,\\perp}$", "$T_{i,\\parallel}$"],
                    loc="lower right", **legend_options)
    axs10[2].set_ylabel("$E_i$ [eV]")
    caxs02.set_ylabel("DEF" + "\n" + "[keV/(cm$^2$ s sr keV)]")
    axs10[2].grid(False, which="both")

    # Electrons energy spectrum
    axs10[3], caxs11 = plot_spectr(axs10[3], def_omni_e,
                                   clim=[1e5, 3e7], **spectr_options)
    plot_line(axs10[3], t_perp_e, color="k")
    plot_line(axs10[3], t_para_e, color="tab:blue")
    axs10[3].legend(["$T_{e,\\perp}$", "$T_{e,\\parallel}$"],
                    loc="lower right", **legend_options)
    axs10[3].set_ylabel("$E_e$ [eV]")
    caxs11.set_ylabel("DEF" + "\n" + "[keV/(cm$^2$ s sr keV)]")
    axs10[3].grid(False, which="both")

    axs10[-1].get_shared_x_axes().join(*axs10)

    fig.align_ylabels(axs10)

    for ax in axs10[:-1]:
        ax.xaxis.set_ticklabels([])

    axs10[-1].set_xlim(mdates.datestr2num(tint_over))

    # Zoom
    plot_line(axs11[0], b_xyz)
    axs11[0].legend(["$B_x$", "$B_y$", "$B_z$"], loc="upper right",
                    **legend_options)
    axs11[0].set_ylim([-15, 15])
    axs11[0].set_ylabel("$B$ [nT]")
    axs11[0].grid(True, which="both")
    axs11[0].set_xlim(mdates.datestr2num(tint_flap))

    span_options = dict(linestyle="--", linewidth=0.8, facecolor="none",
                        edgecolor="k")
    span_tint(axs10, tint_flap, **span_options)

    axr = set_rlabel(axs10[0], r_xyz, spine=0, position="top")

    axr.set_xlabel("$X_{GSM}$ [$R_E$]\n$Y_{GSM}$ [$R_E$]\n$Z_{GSM}$ [$R_E$]",
                   fontsize=9)
    axr.xaxis.set_label_coords(-.13, 1.1)

    zoom(axs11[0], axs10[-1], ec="k")
    print(axs10[-1].get_xticklabels())
    # number subplots
    labels_pos = [0.02, 0.83]
    _ = make_labels(axs10 + axs11, labels_pos)


    if args.figname:
        fig.savefig(args.figname)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--figname",
                        help="Path of the figure to save with extension.",
                        type=str, default="")
    parser.add_argument("--config",
                        help="Path to (.yml) config file.",
                        type=str, required=True)

    main(parser.parse_args())
