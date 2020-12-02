#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ion_demagnetization.py

@author : Louis RICHARD
"""

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dateutil import parser as date_parser
from pyrfu.mms import get_data
from pyrfu.plot import plot_line
from pyrfu.pyrf import (c_4_j, plasma_calc, c_4_grad, norm, mva, new_xyz, medfilt, calc_dt)

from spf import (load_moments, remove_bz_offset, pressure_balance_b0,
                 dec_temperature, make_labels)


def main(args):
    """main function
    """

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tints"]["flapping"]

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint, i, args.verbose) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint, i, args.verbose) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute magnetic field at the center of mass of the tetrahedron
    j_xyz, _, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9                                                            # j A.m^{-2}->nA.m^{-2}

    # Load moments of the velocity distribution functions
    moments_i, moments_e = load_moments(tint, cfg["fpi"], args)

    # Estimate lobe field using pressure balance
    b0 = pressure_balance_b0(b_xyz, moments_i, moments_e)

    # Unpack number densities
    n_i, n_e = moments_i[0], moments_e[0]

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_e)

    plasma_params = plasma_calc(b0, t_i, t_e, n_i, n_e)

    # Compute curvature
    curvature_b = c_4_grad(r_mms, b_mms, "curv")

    # Compute radius of curvature
    r_c = 1 / norm(curvature_b)

    kappas = [np.sqrt(r_c / (1e-3 * rho)) for rho in [plasma_params.rho_p, plasma_params.rho_e]]

    # Compute MVA frame
    b_lmn, _, lmn = mva(b_xyz)

    # Correct minimum variance frame
    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    # Transform fields to MVA frame
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    # Transform moments
    moments_lmn_i = [moments_i[0], None, None, None]
    moments_lmn_e = [moments_e[0], None, None, None]

    moments_lmn_i[1:] = [new_xyz(mom, lmn) for mom in moments_i[1:]]
    moments_lmn_e[1:] = [new_xyz(mom, lmn) for mom in moments_e[1:]]

    # Unpack number density
    n_i, n_e = [moments_lmn_i[0], moments_lmn_e[0]]

    # Unpack bulk velocity
    v_lmn_i, v_lmn_e = [moments_lmn_i[1], moments_lmn_e[1]]

    kappa_i_filtered = medfilt(kappas[0] ** 2, int(np.floor(2 / calc_dt(v_lmn_i))))
    kappa_e_filtered = medfilt(kappas[1] ** 2, int(np.floor(2 / calc_dt(v_lmn_e))))

    figure_options = dict()
    gspecs_options = dict()
    legend_options = dict()

    # Plot
    fig, axs = plt.subplots(6, **cfg["figure"]["main"])
    fig.subplots_adjust(**cfg["figure"]["subplots"])

    # B GSE
    plot_line(axs[0], b_xyz)
    axs[0].set_ylim([-14, 14])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])

    # kappa_i, kappa_e
    plot_line(axs[1], kappa_i_filtered, "tab:blue")
    plot_line(axs[1], kappa_e_filtered, "tab:red")
    axs[1].axhline(10, color="k", linestyle="--")
    axs[1].axhline(25, color="k", linestyle="-.")
    axs[1].legend(["Ions", "Electrons"], **cfg["figure"]["legend"])
    axs[1].set_yscale("log")
    axs[1].set_ylabel("$\\kappa^2$")

    # ViM, VeM, B_L
    plot_line(axs[2], v_lmn_i[:, 1], "tab:blue")
    plot_line(axs[2], v_lmn_e[:, 1], "tab:red")
    axs[2].set_ylim([-420, 420])
    axs[2].set_ylabel("$V_M$ [km s$^{-1}$]")
    axs[2].legend(["Ions", "Electrons"], **cfg["figure"]["legend"])

    # ViN, VeN, B_L
    plot_line(axs[3], v_lmn_i[:, 2], "tab:blue")
    plot_line(axs[3], v_lmn_e[:, 2], "tab:red")
    axs[3].set_ylim([-420, 420])
    axs[3].set_ylabel("$V_N$ [km s$^{-1}$]")
    axs[3].legend(["Ions", "Electrons"], **cfg["figure"]["legend"])

    # Ni, Ne, B_L
    plot_line(axs[4], n_i, "tab:blue")
    plot_line(axs[4], n_e, "tab:red")
    axs[4].set_ylim([0.06, 0.34])
    axs[4].set_ylabel("$N$ [cm$^{-3}$]")
    axs[4].legend(["Ions", "Electrons"], **cfg["figure"]["legend"])

    # JM, JN, B_L
    plot_line(axs[5], j_lmn[:, 1], "tab:orange")
    plot_line(axs[5], j_lmn[:, 2], "tab:green")
    axs[5].set_ylim([-28, 28])
    axs[5].set_ylabel("$J$ [nA m$^{-2}$]")
    axs[5].legend(["$J_M$", "$J_N$"], **cfg["figure"]["legend"])

    # Add magnetic field to all axes
    for ax in axs[2:]:
        axb = ax.twinx()
        plot_line(axb, b_lmn[:, 0], "darkgrey")
        axb.set_ylim([-14, 14])
        axb.set_ylabel("$B_L$ [nT]")

    axs[-1].set_xlabel(date_parser.parse(tint[0]).strftime("%Y-%m-%d UTC"))
    axs[-1].set_xlim(tint)

    fig.align_ylabels(axs)

    # Add panels labels
    labels_pos = [0.02, 0.83]
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
