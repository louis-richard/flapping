#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dispersion-relation.py

@author : Louis RICHARD
"""

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy import constants
from pyrfu.mms import get_data
from pyrfu.plot import plot_spectr
from pyrfu.pyrf import (c_4_j, mva, new_xyz, norm, calc_disprel_tm, t_eval, c_4_grad)

from spf import (load_timing, remove_bz_offset, fit_harris_cs,
                 calc_vph_current, scaling_h_lambda, make_labels)


def main(args):
    """main function
    """

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tint"]

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint, i, args.verbose) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint, i, args.verbose) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute current density
    j_xyz, div_b, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)

    # j A.m^{-2}->nA.m^{-2}
    j_xyz *= 1e9

    # Compute MVA frame
    _, _, lmn = mva(b_xyz)

    # Correct Minimum Variance frame
    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    # Transform fields to MVA frame
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    # Compute dispersion relation using either Jn of the timing
    # Estimate phase velocity using normal current density
    disprel_jn = calc_vph_current(b_lmn, j_lmn)

    # Load timing data
    timing_lr = load_timing(args.timing)
    timing_ts = load_timing(args.table_wei2019)

    # Unpack slowness vector
    m_xyz, _ = [timing_lr.m, timing_lr.dm]

    # Crossing times
    crossing_times = m_xyz.time.data

    # Dispersion relation from timing
    disprel_lr = calc_disprel_tm(timing_lr.v, timing_lr.dv, 2 * timing_lr.tau, timing_lr.dtau)
    disprel_ts = calc_disprel_tm(timing_ts.v, timing_ts.dv, 2 * timing_ts.tau, timing_ts.dtau)

    # d_i = 527.0

    # Compute thickness of the CS using the lobe field corresponding to the Harris like CS
    # Compute lobe field using Harris current sheet fit
    harris_fit = fit_harris_cs(b_lmn, j_lmn)

    # Compute thickness as h = B0/curl(B) = B0/(m0*J) and the error
    mu0 = constants.mu0.value
    h_c = norm(harris_fit.B0) / (1e3 * mu0 * norm(j_lmn)) 	# continuous thickness
    h_d = t_eval(h_c, crossing_times) 						# discrete thickness

    # curl of the magnetic field
    curl_b = c_4_grad(r_mms, b_mms, "curl")

    # Errors
    dj_j = np.abs(div_b / norm(curl_b))	 					# relative error on curlometer technique
    dh_d = h_d * t_eval(dj_j / (1 + dj_j), crossing_times) 	# relative error on the thickness

    # Thickness from Wei 2019 table
    h_ts, dh_ts = [timing_ts.h, timing_ts.dh]

    # Outliers indices
    ols = cfg["outliers"]

    # Fit scaling wavelength CS thickness
    scaling_lr = scaling_h_lambda(h_d, dh_d, disprel_lr, ols["Richard2021"])
    scaling_ts = scaling_h_lambda(h_ts, dh_ts, disprel_ts, ols["Wei2019"])

    fig6 = plt.figure(**cfg["figure"]["main"])

    gs6 = fig6.add_gridspec(2, 2, **cfg["figure"]["gridspec"])

    gs60 = gs6[:, 0].subgridspec(2, 1)
    gs61 = gs6[:, 1].subgridspec(2, 1)

    axs0 = [fig6.add_subplot(gs60[i]) for i in range(2)]
    axs1 = [fig6.add_subplot(gs61[i]) for i in range(2)]

    axs0[0], caxs00 = plot_spectr(axs0[0], disprel_jn.hist, cscale="log", cmap="viridis")
    axs0[0].plot(disprel_jn.hires_dBdt, disprel_jn.pred_Jn, "k")
    axs0[0].fill_between(disprel_jn.hires_dBdt,
                         disprel_jn.bound_upper,
                         disprel_jn.bound_lower,
                         color="lightgrey")

    labels = ["$J_N=${}*d$_t B_L$".format(disprel_jn.fit_db_dt_jn.data), "CI 95%"]
    axs0[0].legend(labels, frameon=False, loc="upper right")
    axs0[0].set_xlabel("d$_t B_L$ [nT s$^{-1}$]")
    axs0[0].set_ylabel("$J_N$ [nA m$^{-2}$]")
    caxs00.set_ylabel("#")

    h_lambda = scaling_lr.h.data * scaling_lr.k.data / (2 * np.pi)
    _, _, _ = axs0[1].hist(h_lambda, **cfg["figure"]["hist"])
    axs0[1].axvline(scaling_lr.scaling.data / (2 * np.pi), color="k", linestyle="-")
    axs0[1].set_ylim([0, 17])
    left_bound = (scaling_lr.scaling.data - 1.96 * scaling_lr.sigma_scaling.data) / (2 * np.pi)
    right_bound = (scaling_lr.scaling.data + 1.96 * scaling_lr.sigma_scaling.data) / (2 * np.pi)
    width = right_bound - left_bound
    _, ymax = axs0[1].get_ylim()
    height = .5
    rect_drift = plt.Rectangle((0.8 / (2 * np.pi), ymax - .5), 1. / (2 * np.pi), .5,
                               color='tab:orange')
    rect_sigma = plt.Rectangle((left_bound, ymax - 2.1 * height), width, height, color='lightgrey')
    axs0[1].add_patch(rect_drift)
    axs0[1].add_patch(rect_sigma)

    axs0[1].set_xlabel("$h/\\lambda$")
    axs0[1].set_ylabel("#")
    axs1[0].errorbar(disprel_lr.k.data, disprel_lr.omega.data,
                     disprel_lr.omega_err.data, disprel_lr.k_err.data,
                     color="tab:blue", **cfg["figure"]["errorbar"])

    axs1[0].errorbar(disprel_ts.k.data, disprel_ts.omega.data,
                     disprel_ts.omega_err.data, disprel_ts.k_err.data,
                     color="tab:red", **cfg["figure"]["errorbar"])

    axs1[0].errorbar(disprel_lr.k.data[ols["Richard2021"]],
                     disprel_lr.omega.data[ols["Richard2021"]],
                     disprel_lr.omega_err.data[ols["Richard2021"]],
                     disprel_lr.k_err.data[ols["Richard2021"]],
                     color="k", **cfg["figure"]["errorbar"])

    axs1[0].plot(disprel_lr.hires_k.data, disprel_lr.pred_omega.data, 'k-')
    axs1[0].fill_between(disprel_lr.hires_k.data, disprel_lr.bound_lower.data,
                         disprel_lr.bound_upper.data, color="lightgrey")
    axs1[0].set_xlim([0, 2.6e-3])
    axs1[0].set_ylim([0, 1])
    axs1[0].set_xlabel("$k$ [km$^{-1}$]")
    axs1[0].set_ylabel("$\\omega$ [rad s$^{-1}$]")
    labels = ["$\\omega$ = {:4.2f}*$k$".format(disprel_lr.vph.data[0]), "CI 95%", "2019-09-14",
              "Table Wei2019"]
    axs1[0].legend(labels, frameon=True, loc="upper right")

    axs1[0].yaxis.set_label_position("right")
    axs1[0].yaxis.tick_right()

    axs1[1].errorbar(scaling_lr.k, scaling_lr.h,
                     scaling_lr.dh, scaling_lr.dk,
                     color="tab:blue", **cfg["figure"]["errorbar"])

    axs1[1].errorbar(scaling_ts.k, scaling_ts.h,
                     scaling_ts.dh, scaling_ts.dk,
                     color="tab:red", **cfg["figure"]["errorbar"])

    axs1[1].errorbar(scaling_lr.k_ols, scaling_lr.h_ols,
                     scaling_lr.dh_ols, scaling_lr.dk_ols,
                     color="k", **cfg["figure"]["errorbar"])

    axs1[1].plot(scaling_lr.hires_k, scaling_lr.pred_h, "k")
    axs1[1].fill_between(scaling_lr.hires_k,
                         scaling_lr.bound_lower,
                         scaling_lr.bound_upper,
                         color="lightgrey")

    axs1[1].set_xlabel("$k$ [km$^{-1}$]")
    axs1[1].set_ylabel("$h$ [km]")
    labels = ["$h=${:3.2f}$*\\lambda$".format(scaling_lr.scaling.data / (2 * np.pi)), "95% CI",
               "2019-09-14", "Table Wei2019"]
    axs1[1].legend(labels, frameon=True, loc="upper right")

    axs1[1].plot(scaling_lr.hires_k, 0.8 / scaling_lr.hires_k, "tab:orange")
    axs1[1].plot(scaling_lr.hires_k, 1.8 / scaling_lr.hires_k, "tab:orange")

    axs1[1].yaxis.set_label_position("right")
    axs1[1].yaxis.tick_right()
    axs1[1].set_xlim([0, 2.6e-3])
    axs1[1].set_ylim([0, 8500])

    # Add panels labels
    labels_pos = [0.05, 0.95]
    _ = make_labels(axs0 + axs1, labels_pos)

    if args.figname:
        fig6.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")

    parser.add_argument("--config",
                        help="Path to (.yml) config file.",
                        type=str, required=True)

    parser.add_argument("--table-wei2019",
                        help="Path to (.h5) Wei2019 file.",
                        type=str, required=True)

    parser.add_argument("--timing",
                        help="Path to (.h5) timing file.",
                        type=str, required=True)

    parser.add_argument("--figname",
                        help="Path and name of the figure to save with extension.",
                        type=str, default="")

    main(parser.parse_args())
