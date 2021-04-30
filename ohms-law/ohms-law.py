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
from pyrfu.pyrf import avg_4sc, c_4_j, resample, new_xyz, norm
from pyrfu.plot import zoom, plot_line

from spf import (lmn_cs, load_moments, remove_bz_offset, calc_ol_terms,
                 span_tint, make_labels)


def main(args):
    """main function
    """
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint_flap = cfg["tints"]["flapping"]
    tint_zoom = cfg["tints"]["close-up"]

    mms_ids = np.arange(1, 5)

    # Load spacecraft position
    r_mms = [get_data("R_gse", tint_flap, i, data_path=cfg["data_path"]) for i
             in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint_flap, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Load electric field
    suf_e = "edp_{}_{}".format(cfg["edp"]["data_rate"], cfg["edp"]["level"])
    e_mms = [get_data("e_gse_{}".format(suf_e), tint_flap, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Load moments from FPI
    moments_i, moments_e = load_moments(tint_flap, cfg["fpi"], args,
                                        data_path=cfg["data_path"])

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute current density
    j_xyz, _, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9  # j A.m^{-2}->nA.m^{-2}

    # Compute electric field at the center of mass of the tetrahedron
    e_xyz = avg_4sc(e_mms)

    # Compute new coordinates system
    lmn = lmn_cs(b_xyz, moments_i[1])

    # Resample magnetic field and current density to electric field sampling
    b_xyz, j_xyz = [resample(field, e_xyz) for field in [b_xyz, j_xyz]]

    # Transform fields to MVA frame
    b_lmn, e_lmn, j_lmn = [new_xyz(x_, lmn) for x_ in [b_xyz, e_xyz, j_xyz]]

    # Resample ion/electron moments to electric field sampling
    moments_i = [resample(mom, e_xyz) for mom in moments_i]
    moments_e = [resample(mom, e_xyz) for mom in moments_e]

    # Transform moments
    moments_lmn_i = [moments_i[0], None, None, None]
    moments_lmn_e = [moments_e[0], None, None, None]

    moments_lmn_i[1:] = [new_xyz(mom, lmn) for mom in moments_i[1:]]
    moments_lmn_e[1:] = [new_xyz(mom, lmn) for mom in moments_e[1:]]

    # Compute terms of the Ohm's law
    vxb_lmn_i, vxb_lmn_e, jxb_lmn = calc_ol_terms(b_lmn, j_lmn,
                                                  moments_lmn_i, moments_lmn_e)

    # Compute ion scale total contribution
    s_lmn = -vxb_lmn_i + jxb_lmn

    # Plot
    fig = plt.figure(**cfg["figure"]["main"])
    gs0 = fig.add_gridspec(2, 1, **cfg["figure"]["gridspec"])

    gs00 = gs0[:1].subgridspec(4, 1, hspace=0)
    gs10 = gs0[1:].subgridspec(4, 1, hspace=0)

    axs0 = [fig.add_subplot(gs00[i]) for i in range(4)]
    axs1 = [fig.add_subplot(gs10[i]) for i in range(4)]

    for axs in [axs0, axs1]:
        plot_line(axs[0], b_lmn)
        axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])
        axs[0].set_ylabel("$B$" + "\n" + "[nT]")
        axs[0].grid(True, which="both")

        plot_line(axs[1], jxb_lmn[:, 1], label="$J\\times B / ne$")
        plot_line(axs[1], -vxb_lmn_i[:, 1], label="$-V_i\\times B$")
        plot_line(axs[1], s_lmn[:, 1], label="$J\\times B / ne -V_i\\times B$")
        plot_line(axs[1], e_lmn[:, 1], label="$E_{EDP}$")
        axs[1].set_ylim([-12, 12])
        axs[1].set_ylabel("$E_M$" + "\n" + "[mV m$^{-1}$]")
        axs[1].grid(True, which="both")

        plot_line(axs[2], jxb_lmn[:, 2], label="$J\\times B / ne$")
        plot_line(axs[2], -vxb_lmn_i[:, 2], label="$-V_i\\times B$")
        plot_line(axs[2], s_lmn[:, 2], label="$J\\times B / ne -V_i\\times B$")
        plot_line(axs[2], e_lmn[:, 2], label="$E_{EDP}$")
        axs[2].legend(frameon=True, ncol=2, bbox_to_anchor=(1, 1.3),
                      loc="upper right")
        axs[2].set_ylim([-12, 12])
        axs[2].set_ylabel("$E_N$" + "\n" + "[mV m$^{-1}$]")
        axs[2].grid(True, which="both")

        plot_line(axs[3], norm(e_xyz + vxb_lmn_i),
                  color="deeppink", label="$| E+V_{i}\\times B |$")
        plot_line(axs[3], norm(jxb_lmn),
                  color="tab:blue", label="$|J\\times B / ne|$")
        axs[3].legend(**cfg["figure"]["legend"])
        axs[3].set_ylim([0, 12])
        axs[3].set_ylabel("$|E|$" + "\n" + "[mV m$^{-1}$]")
        axs[3].grid(True, which="both")

        axs[-1].get_shared_x_axes().join(*axs)

        fig.align_ylabels(axs)

        for ax in axs[:-1]:
            ax.xaxis.set_ticklabels([])

    axs0[-1].set_xlim(mdates.date2num(tint_flap))
    axs1[-1].set_xlim(mdates.date2num(tint_zoom))

    # zoom
    zoom(axs1[0], axs0[-1], ec="k")
    span_options = dict(linestyle="--", linewidth=0.8, facecolor="none",
                        edgecolor="k")
    span_tint(axs0, tint_zoom, **span_options)
    # number subplots
    labels_pos = [0.02, 0.83]
    _ = make_labels(axs0 + axs1, labels_pos)

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
                        help="Path and name of the figure with extension.",
                        type=str, default="")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to (.yml) config file.")

    main(parser.parse_args())
