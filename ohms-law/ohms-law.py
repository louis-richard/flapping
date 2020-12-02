#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ohms-law.py

@author : Louis RICHARD
"""

import os
import yaml
import string
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dateutil.parser import parse
from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc, c_4_j, resample, mva, new_xyz, norm
from pyrfu.plot import zoom, plot_line

from spf import load_moments, remove_bz_offset, calc_ol_terms


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
    r_mms = [get_data("R_gse", tint_flap, i) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint_flap, i, args.verbose) for i in mms_ids]

    # Load electric field
    suf_e = "edp_{}_{}".format(cfg["edp"]["data_rate"], cfg["edp"]["level"])
    e_mms = [get_data("e_gse_{}".format(suf_e), tint_flap, i, args.verbose) for i in mms_ids]

    # Load moments from FPI
    moments_i, moments_e = load_moments(tint_flap, cfg["fpi"], args)

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute current density
    j_xyz, _, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9  # j A.m^{-2}->nA.m^{-2}

    # Compute electric field at the center of mass of the tetrahedron
    e_xyz = avg_4sc(e_mms)

    # Compute MVA frame
    _, _, lmn = mva(b_xyz)

    # Correct MVA frame
    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    # Resample magnetic field and current density to electric field sampling
    b_xyz, j_xyz = [resample(field, e_xyz) for field in [b_xyz, j_xyz]]

    # Resample ion/electron moments to electric field sampling
    moments_i = [resample(mom, e_xyz) for mom in moments_i]
    moments_e = [resample(mom, e_xyz) for mom in moments_e]

    # Transform fields to MVA frame
    b_lmn, e_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, e_xyz, j_xyz]]

    # Transform moments
    moments_lmn_i = [moments_i[0], None, None, None]
    moments_lmn_e = [moments_e[0], None, None, None]

    moments_lmn_i[1:] = [new_xyz(mom, lmn) for mom in moments_i[1:]]
    moments_lmn_e[1:] = [new_xyz(mom, lmn) for mom in moments_e[1:]]

    # Compute terms of the Ohm's law
    vxb_lmn_i, vxb_lmn_e, jxb_lmn = calc_ol_terms(b_lmn, j_lmn, moments_lmn_i, moments_lmn_e)

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

        plot_line(axs[1], jxb_lmn[:, 1])
        plot_line(axs[1], -vxb_lmn_i[:, 1])
        plot_line(axs[1], s_lmn[:, 1])
        plot_line(axs[1], e_lmn[:, 1])
        labels = ["$J\\times B$", "$-V_i\\times B$", "$J\\times B -V_i\\times B$", "$E_{EDP}$"]
        axs[1].legend(labels, **cfg["figure"]["legend"])
        axs[1].set_ylim([-12, 12])
        axs[1].set_ylabel("$E_M$" + "\n" + "[mV m$^{-1}$]")
        axs[1].grid(True, which="both")

        plot_line(axs[2], jxb_lmn[:, 2])
        plot_line(axs[2], -vxb_lmn_i[:, 2])
        plot_line(axs[2], s_lmn[:, 2])
        plot_line(axs[2], e_lmn[:, 2])
        labels = ["$J\\times B$", "$-V_i\\times B$", "$J\\times B -V_i\\times B$", "$E_{EDP}$"]
        axs[2].legend(labels, **cfg["figure"]["legend"])
        axs[2].set_ylim([-12, 12])
        axs[2].set_ylabel("$E_N$" + "\n" + "[mV m$^{-1}$]")
        axs[2].grid(True, which="both")

        plot_line(axs[3], norm(e_xyz + vxb_lmn_i), "deeppink")
        plot_line(axs[3], norm(jxb_lmn), "k")
        labels = ["$E+V_{i}\\times B$", "$\\frac{J\\times B}{ne}$"]
        axs[3].legend(labels, **cfg["figure"]["legend"])
        axs[3].set_ylim([0, 12])
        axs[3].set_ylabel("$|E|$" + "\n" + "[mV m$^{-1}$]")
        axs[3].grid(True, which="both")

        axs[-1].set_xlabel("2019-09-14 UTC")
        axs[-1].get_shared_x_axes().join(*axs)

        fig.align_ylabels(axs)

        for ax in axs[:-1]:
            ax.xaxis.set_ticklabels([])

    axs0[-1].set_xlim(tint_flap)
    axs1[-1].set_xlim(tint_zoom)

    # zoom
    zoom(axs1[0], axs0[-1], ec="k")

    span_options = dict(linestyle="--", linewidth=0.8, facecolor="lightgrey", edgecolor="k")
    for axis in axs0:
        axis.axvspan(parse(tint_zoom[0]), parse(tint_zoom[1]), **span_options)

    # number subplots
    posx_text, posy_text = [0.02, 0.83]

    axs = axs0 + axs1
    lbl = string.ascii_lowercase[:len(axs)]

    for label, axis in zip(lbl, axs):
        axis.text(posx_text, posy_text, "({})".format(label), transform=axis.transAxes)

    if args.figname:
        fig.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--figname",
                        help="Path and name of the figure to save with extension.",
                        type=str, default="")
    parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")

    main(parser.parse_args())
