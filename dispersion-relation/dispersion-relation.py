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

"""Reproduces the Figure 6 in Richard et al. 2021.
@author: Louis Richard
"""

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import constants
from pyrfu.mms import get_data
from pyrfu.plot import plot_spectr
from pyrfu.pyrf import (c_4_j, new_xyz, norm, t_eval, c_4_grad, resample,
                        optimize_nbins_1d, plasma_calc)

from spf import (lmn_cs, load_timing, remove_bz_offset, fit_harris_cs,
                 calc_vph_current, scaling_h_lambda, make_labels,
                 load_moments, dec_temperature, calc_disprel_tm)


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def main(args):
    """main function
    """

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tint"]

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

    # Compute current density
    j_xyz, div_b, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)

    # j A.m^{-2}->nA.m^{-2}
    j_xyz *= 1e9

    # Load moments
    moments_i, moments_e = load_moments(tint, cfg["fpi"], args,
                                        cfg["data_path"])

    # Resample moments to magnetic field sampling
    moments_i = [resample(mom, b_xyz) for mom in moments_i]
    moments_e = [resample(mom, b_xyz) for mom in moments_e]


    lmn = lmn_cs(b_xyz, moments_i[1])

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
    m_lmn = new_xyz(m_xyz, lmn)

    n_lmn = new_xyz(timing_lr.n, lmn)
    delta = np.arccos(n_lmn[:, 1] / np.linalg.norm(n_lmn.data[:, 1:], axis=1))
    v_lmn = timing_lr.v.data[:, None] * n_lmn
    v_n = np.linalg.norm(v_lmn[:, 1:], axis=1)
    v_ph = v_n / np.cos(delta)
    v_ph[delta > np.pi / 2 - .01] = np.nan
    dv_ph = timing_lr.dv / np.cos(delta)
    dv_ph[delta > np.pi / 2 - .01] = np.nan

    # Crossing times
    crossing_times = m_xyz.time.data

    # Dispersion relation from timing
    disprel_lr = calc_disprel_tm(v_ph, dv_ph, 2 * timing_lr.tau,
                                 timing_lr.dtau, ci=95)

    disprel_ts = calc_disprel_tm(timing_ts.v, timing_ts.dv, 2 * timing_ts.tau,
                                 timing_ts.dtau)

    # d_i = 527.0
    # Compute thickness of the CS using the lobe field corresponding to the
    # Harris like CS
    # Compute lobe field using Harris current sheet fit
    harris_fit = fit_harris_cs(b_lmn, j_lmn)

    # Compute thickness as h = B0/curl(B) = B0/(m0*J) and the error
    mu0 = constants.mu_0
    # continuous thickness
    h_c = norm(harris_fit.B0) / (1e3 * mu0 * norm(j_lmn))
    # discrete thickness
    h_d = t_eval(h_c, crossing_times)

    # curl of the magnetic field
    curl_b = c_4_grad(r_mms, b_mms, "curl")

    # Errors
    # relative error on curlometer technique
    dj_j = np.abs(div_b / norm(curl_b))
    # relative error on the thickness
    dh_d = h_d * t_eval(dj_j / (1 + dj_j), crossing_times)

    # Thickness from Wei 2019 table
    h_ts, dh_ts = [timing_ts.h, timing_ts.dh]

    # Compute ion inertial length to normalize the thickness
    # unpack number densities
    n_i, n_e = moments_i[0], moments_e[0]

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_i)

    # Compute plasma parameters
    plasma_params = plasma_calc(harris_fit.B0, t_i, t_e, n_i, n_e)

    # Averaged ion inertial length
    d_i = np.mean(plasma_params.l_i).data
    r_p = np.mean(plasma_params.rho_p).data
    r_e = np.mean(plasma_params.rho_e).data

    # Outliers indices
    ols = cfg["outliers"]

    # Fit scaling wavelength CS thickness
    scaling_lr = scaling_h_lambda(h_d[~np.isnan(v_ph)], dh_d[~np.isnan(v_ph)],
                                  disprel_lr, ols["Richard2021"])
    scaling_ts = scaling_h_lambda(h_ts, dh_ts, disprel_ts, ols["Wei2019"])

    fig6 = plt.figure(**cfg["figure"]["main"])

    gs6 = fig6.add_gridspec(2, 2, **cfg["figure"]["gridspec"])

    gs60 = gs6[:, 0].subgridspec(2, 1)
    gs61 = gs6[:, 1].subgridspec(2, 1)

    axs0 = [fig6.add_subplot(gs60[i]) for i in range(2)]
    axs1 = [fig6.add_subplot(gs61[i]) for i in range(2)]

    axs0[0], caxs00 = plot_spectr(axs0[0], disprel_jn.hist,
                                  cscale="log", cmap="viridis")
    axs0[0].plot(disprel_jn.hires_dBdt, disprel_jn.pred_Jn, "k")
    axs0[0].fill_between(disprel_jn.hires_dBdt,
                         disprel_jn.bound_upper,
                         disprel_jn.bound_lower,
                         color="lightgrey")

    labels = ["$J_N=${:3.2f}*d$_t B_L$".format(disprel_jn.fit_db_dt_jn.data),
              "95% CI"]
    axs0[0].legend(labels, frameon=True, loc="upper right")
    axs0[0].set_xlabel("d$_t B_L$ [nT s$^{-1}$]")
    axs0[0].set_ylabel("$J_N$ [nA m$^{-2}$]")
    caxs00.set_ylabel("Counts")
    axs0[0].grid(True, which="both")
    axs0[0].set_title("$\\rho = {:3.2f}$".format(disprel_jn.rho.data))

    h_lambda = scaling_lr.h.data * scaling_lr.k.data / (2 * np.pi)
    dh_lambda = scaling_lr.h.data * scaling_lr.dk.data / (2 * np.pi)

    n_bins = optimize_nbins_1d(h_lambda)
    """
    y, edges_ = np.histogram(h_lambda, bins=n_bins)
    axs0[1].bar(0.5 * (edges_[1:] + edges_[:-1]), y,
                width=edges_[1:] - edges_[:-1], facecolor='tab:blue',
                edgecolor="k", yerr=np.sqrt(y))
    """
    #axs0[1].axvspan(0.8 / (2 * np.pi), 1.8 / (2 * np.pi), color="tab:orange")
    _, _, _ = axs0[1].hist(h_lambda, bins=n_bins, **cfg["figure"]["hist"])
    #axs0[1].errorbar(scaling_lr.scaling.data / (2 * np.pi), 16.5,
    #                 xerr=scaling_lr.sigma_scaling.data / (2 * np.pi),)
    axs0[1].axvline(scaling_lr.scaling.data / (2 * np.pi),
                    color="k", linestyle="-")
    axs0[1].set_ylim([0, 17])
    axs0[1].set_yticks(np.arange(0, 17, 2))

    left_bound = (scaling_lr.scaling.data
                  - 1.96 * scaling_lr.sigma_scaling.data) / (2 * np.pi)
    right_bound = (scaling_lr.scaling.data
                   + 1.96 * scaling_lr.sigma_scaling.data) / (2 * np.pi)
    width = right_bound - left_bound
    _, ymax = axs0[1].get_ylim()
    height = .5
    rect_drift = plt.Rectangle((0.8 / (2 * np.pi), ymax - .5),
                               1. / (2 * np.pi), .5,
                               color='tab:orange')
    rect_sigma = plt.Rectangle((left_bound, ymax - 2.1 * height),
                               width, height, color='lightgrey')
    axs0[1].add_patch(rect_drift)
    axs0[1].add_patch(rect_sigma)

    axs0[1].set_xlabel("$h/\\lambda$")
    axs0[1].set_ylabel("Counts")
    axs0[1].grid(True, which="both")

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
    labels = ["$\\omega$ = {:4.2f}*$k$".format(disprel_lr.vph.data[0]),
              "95% CI", "2019-09-14", "Table Wei2019"]
    axs1[0].legend(labels, frameon=True, loc="upper right")

    axs1[0].yaxis.set_label_position("right")
    axs1[0].yaxis.tick_right()
    axs1[0].grid(True, which="both")

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
    labels = [f"$h=${scaling_lr.scaling.data / (2 * np.pi):3.2f}$*\\lambda$",
              "95% CI",  "2019-09-14", "Table Wei2019"]
    axs1[1].legend(labels, frameon=True, loc="upper right")

    axs1[1].plot(scaling_lr.hires_k, 0.8 / scaling_lr.hires_k,
                 color="tab:orange")
    axs1[1].plot(scaling_lr.hires_k, 1.8 / scaling_lr.hires_k,
                 color="tab:orange")

    axs1[1].yaxis.set_label_position("right")
    axs1[1].yaxis.tick_right()
    axs1[1].set_xlim([0, 2.6e-3])
    axs1[1].set_ylim([0, 9000])
    axs1[1].grid(True, which="both")

    # Add panels labels
    labels_pos = [0.032, 0.95]
    _ = make_labels(axs0 + axs1, labels_pos)

    if args.figname:
        fig6.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()

    if args.verbose:
        b_lobe = harris_fit.b_lobe.data
        sigma_b_lobe = harris_fit.sigma_b_lobe.data

        h_h = harris_fit.scale.data / d_i
        sigma_h_h = harris_fit.sigma_scale.data / d_i

        a_j = disprel_jn.fit_db_dt_jn.data
        sigma_a_j = disprel_jn.sigma.data

        v_ph_j = 1e-3 / (constants.mu_0 * a_j)
        sigma_v_ph_j = v_ph_j * (sigma_a_j / a_j) / (1 + (sigma_a_j / a_j))

        v_ph = disprel_lr.vph.data[0]
        sigma_v_ph = disprel_lr.sigma_k_w.data[0]

        kh_ = scaling_lr.scaling.data
        sigma_kh = scaling_lr.sigma_scaling.data

        h_l = kh_ / (2 * np.pi)
        sigma_h_l = sigma_kh / (2 * np.pi)

        h_min = 1e3 * np.min(h_d.data)
        l_min = 1e3 * np.min(disprel_lr.lamb.data)

        l_lh = 2 * np.pi * np.sqrt(r_p * r_e)

        print(f"B_0 = {b_lobe:3.2f} \pm {sigma_b_lobe:3.2f} nT")
        print(f"h_H = {h_h:3.2f} \pm {sigma_h_h:3.2f} d_i")
        print(f"a = {a_j:3.2f} \pm {sigma_a_j:3.2f} s/H")
        print(f"v_ph = {v_ph_j:3.2f} \pm {sigma_v_ph_j:3.2f} km/s")
        print(f"v_ph = {v_ph:3.2f} \pm {sigma_v_ph:3.2f} km/s")
        print(f"kh = {kh_:3.2f} \pm {sigma_kh:3.2f}")
        print(f"h/\lambda = {h_l:3.2f} \pm {sigma_h_l:3.2f}")
        print(f"\lambda_min = {(l_min / d_i):3.2f} d_i")
        print(f"\lambda_min = {(l_min / r_p):3.2f} r_i")
        print(f"\lambda_LH = {(l_lh / d_i):3.2f}")
        print(f"h_min = {(h_min / d_i):3.2f} d_i")
        print(f"h_min = {(h_min / r_p):3.2f} r_i")


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
