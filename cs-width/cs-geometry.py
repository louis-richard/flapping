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
import string
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pyrfu.pyrf import (c_4_j, mva, new_xyz, resample, norm, t_eval, plasma_calc)
from pyrfu.mms import get_data
from pyrfu.plot import plot_spectr, zoom
from astropy import constants

from spf import (load_timing, load_moments, make_labels, remove_bz_offset,
                 fit_harris_cs, dec_temperature, st_derivative)


def main(args):
    """main function
    """
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", cfg["tint"], i, args.verbose) for i in mms_ids]

    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), cfg["tint"], i, args.verbose) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Load moments
    moments_i, moments_e = load_moments(cfg["tint"], cfg["fpi"], args)

    # Compute current density
    j_xyz, div_b, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9  														# j A.m^{-2}->nA.m^{-2}

    # Compute MVA frame
    _, _, lmn = mva(b_xyz)

    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    # transform magnetic field and current density to LMN coordinates system
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    # Resample moments to magnetic field sampling
    moments_i = [resample(mom, b_xyz) for mom in moments_i]
    moments_e = [resample(mom, b_xyz) for mom in moments_e]

    # Fit J perp vs B as Harris like CS
    harris_fit = fit_harris_cs(b_lmn, j_lmn)

    # Load data from timing
    timing_lr = load_timing(args.timing)

    # Transform slowness vector to LMN frame
    m_lmn = new_xyz(timing_lr.m, lmn)

    # Get crossing times
    crossing_times = m_lmn.time.data

    # Use Harris like CS lobe field to estimate the thickness
    # Compute thickness as h = B0/curl(B) = B0/(m0*J) and the error
    mu0 = constants.mu0.value
    h_c = norm(harris_fit.B0) / (1e3 * mu0 * norm(j_lmn)) 	# continuous thickness
    h_d = t_eval(h_c, crossing_times) 								# discrete thickness

    # Errors
    # relative error on curlometer technique
    dj_j = np.abs(div_b / (mu0 * norm(1e-9 * j_xyz)))
    # relative error on the thickness
    dh_d = h_d * t_eval(dj_j / (1 + dj_j), crossing_times)

    # Compute ion inertial length to normalize the thickness
    # unpack number densities
    n_i, n_e = moments_i[0], moments_e[0]

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_i)

    # Compute plasma parameters
    plasma_params = plasma_calc(harris_fit.B0, t_i, t_e, n_i, n_e)

    # Averaged ion inertial length
    d_i = 1e-3 * np.mean(plasma_params.l_i).data

    # Normalize thickness by ion inertial length
    h_d, dh_d = [h_d / d_i, dh_d / d_i]

    # Compute velocity and geometry of the CS using spatio-temporal derivative
    v_str_lmn, y_m, z_n = st_derivative(r_mms, b_mms, lmn, crossing_times)

    geometry = xr.Dataset({"y_m": y_m / d_i, "z_n": z_n / d_i})

    # Compute ym and zn at the crossings
    crossing_times = h_d.time.data

    yc_m, zc_n = [t_eval(x, crossing_times) for x in [geometry.y_m, geometry.z_n]]

    fig = plt.figure(**cfg["figure"]["main"])
    gs0 = fig.add_gridspec(4, 2, **cfg["figure"]["gridspec"])
    gs10 = gs0[:2, 0].subgridspec(1, 1)
    gs20 = gs0[:2, 1].subgridspec(2, 1, hspace=.4)
    gs30 = gs0[2:, :].subgridspec(1, 1)

    axs0 = [fig.add_subplot(gs10[i]) for i in range(1)]
    axs1 = [fig.add_subplot(gs20[i]) for i in range(2)]
    axs2 = [fig.add_subplot(gs30[i]) for i in range(1)]

    axs0[0], caxs0 = plot_spectr(axs0[0], harris_fit.hist, cscale="log", cmap="viridis")
    axs0[0].errorbar(harris_fit.bbins, harris_fit.medbin, harris_fit.medstd, marker="s", color="k")
    axs0[0].plot(harris_fit.hires_b, harris_fit.pred_j_perp, "k-", linewidth=2)
    axs0[0].set_ylim([0, 18])
    axs0[0].set_xlabel("$B_L$ [nT]")
    axs0[0].set_ylabel("$J_{MN}$ [nA m$^{-2}$]")
    caxs0.set_ylabel("#")
    labels0 = ["Harris fit $B_0$={:3.2f} nT".format(harris_fit.B0.data[0, 0]), "median"]
    axs0[0].legend(labels0, **cfg["figure"]["legend"])

    for ax in axs1:
        ax.errorbar(yc_m.data, zc_n.data, h_d.data / 2, **cfg["figure"]["errorbar"])
        ax.plot(geometry.y_m.data, geometry.z_n.data, "tab:blue")
        ax.set_ylim([-6, 6])
        ax.set_xlabel("$M/d_i$")
        ax.set_ylabel("$N/d_i$")

    axs1[0].legend(["CS", "h"], **cfg["figure"]["legend"])
    axs1[1].set_xticks(yc_m.data)
    axs1[0].set_xlim([0, np.max(geometry.y_m.data)])
    axs1[1].set_xlim([70, 105])

    zoom(axs1[1], axs1[0], ec="k")

    # Label the panels
    axs0[0].text(0.02, 0.95, "({})".format("a"), transform=axs0[0].transAxes)
    axs1 = make_labels(axs1, [0.02, 0.86], 1)

    axs2[0].text(0.01, 0.95, "({})".format(string.ascii_lowercase[len(axs1) + 1]),
                 transform=axs2[0].transAxes)
    axs2[0].axis("off")

    if args.figname:
        fig.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--figname", help="Path and name of the figure to save with extension.",
                        type=str, default="")
    parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")
    parser.add_argument("--timing", type=str, required=True, help="Path to (.h5) timing file.")

    main(parser.parse_args())
