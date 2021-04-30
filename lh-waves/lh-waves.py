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

"""Reproduces the Figure 7 in Richard et al. 2021.
@author: Louis Richard
"""

import yaml
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cycler import cycler
from scipy import constants
from pyrfu.mms import get_data, rotate_tensor
from pyrfu.plot import plot_line, plot_spectr
from pyrfu.pyrf import (avg_4sc, mva, time_clip, new_xyz, resample, trace,
                        norm, plasma_calc, convert_fac, filt, wavelet,
                        c_4_j)

from spf import load_moments, remove_bz_offset, compress_cwt, make_labels

color = ["tab:blue", "tab:green", "tab:red", "k"]
default_cycler = cycler(color=color)
plt.rc('axes', prop_cycle=default_cycler)


def main(args):
    """main function
    """
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint_flap = cfg["tints"]["flapping"]
    tint_zoom = cfg["tints"]["close-up"]

    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint_flap, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint_flap, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]
    
    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute magnetic field at the center of mass of the tetrahedron
    j_xyz, _, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9  # j A.m^{-2}->nA.m^{-2}

    # Compute magnetic field at the center of mass of the tetrahedron
    b_xyz = avg_4sc(b_mms)

    # Compute MVA frame
    b_lmn, _, lmn = mva(b_xyz)

    # Correct minimum variance frame
    # lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T
    l = lmn[:, 0]
    m = np.mean(j_xyz.data, axis=0) / np.linalg.norm(np.mean(j_xyz, axis=0))
    n = np.cross(l, m) / np.linalg.norm(np.cross(l, m))
    m = np.cross(n, l)
    lmn = np.transpose(np.vstack([l, m, n]))

    b_xyz = time_clip(b_mms[args.mms_id-1], tint_zoom)
    b_lmn = new_xyz(b_xyz, lmn)

    # Load electric field and magnetic field fluctuation
    # Load electric field
    suf_e = "edp_{}_{}".format(cfg["edp"]["data_rate"], cfg["edp"]["level"])
    e_xyz = get_data("e_gse_{}".format(suf_e), tint_zoom, args.mms_id,
                     args.verbose, data_path=cfg["data_path"])

    # Load electric field
    suf_db = "scm_{}_{}".format(cfg["scm"]["data_rate"], cfg["scm"]["level"])
    b_scm = get_data("b_gse_{}".format(suf_db), tint_zoom, args.mms_id,
                     args.verbose, data_path=cfg["data_path"])

    # Load density and pressure tensor
    moments_i, moments_e = load_moments(tint_flap, cfg["fpi"], args,
                                        data_path=cfg["data_path"])

    # Unpack moments
    _, _, _, p_xyz_i = moments_i
    n_e, _, _, p_xyz_e = moments_e

    # Total pressure tensor
    p_xyz = p_xyz_i + resample(p_xyz_e, p_xyz_i)

    # Convert to field-aligned coordinates
    p_xyzfac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Permittivity
    mu0 = constants.mu_0

    # Thermal pressure (scalar)
    pth = 1e-9 * trace(p_xyzfac) / 3.

    # Magnetic pressure
    p_mag = 1e-18 * norm(b_xyz) ** 2 / (2 * mu0)

    # Plasma beta
    beta = resample(pth, b_xyz) / p_mag

    # Plasma parameters
    plasma_params = plasma_calc(b_xyz, n_e, n_e, n_e, n_e)

    # Convert electric field and magnetic field fluctuation to field aligned
    # coordinates
    e_xyzfac = convert_fac(e_xyz, b_xyz, [1, 0, 0])

    # bandpass filter electric field and magnetic field fluctuation
    f_min = 4
    e_xyzfac_hf = filt(e_xyzfac, f_min, 0, 3)

    cwt_options = dict(nf=100, f=[0.5, 1e3], plot=False)

    # Compute continuous wavelet transform of the electric field
    e_xyzfac_cwt = wavelet(e_xyzfac, **cwt_options)

    # Construct compressed spectrogram of E
    e_cwt_times, e_cwt_x, e_cwt_y, e_cwt_z = compress_cwt(e_xyzfac_cwt, nc=100)

    e_cwt = xr.DataArray(e_cwt_x + e_cwt_y + e_cwt_z,
                         coords=[e_cwt_times, e_xyzfac_cwt.frequency],
                         dims=["time", "frequency"])

    # Compute continuous wavelet transform of the magnetic field fluctuations
    b_scm_cwt = wavelet(b_scm, **cwt_options)

    # Construct compressed spectrogram of E
    b_cwt_times, b_cwt_x, b_cwt_y, b_cwt_z = compress_cwt(b_scm_cwt, nc=100)

    b_cwt = xr.DataArray(b_cwt_x + b_cwt_y + b_cwt_z,
                         coords=[b_cwt_times, b_scm_cwt.frequency],
                         dims=["time", "frequency"])

    # Plot
    fig, axs = plt.subplots(4, sharex="all", figsize=(6.5, 9))
    fig.subplots_adjust(bottom=.05, top=.95, left=.15, right=.85, hspace=0.)

    kwargs_legend = dict(ncol=3, loc="upper right", frameon=True)
    kwargs_spectr = dict(cscale="log", yscale="log", cmap="Spectral_r",
                         clim=[1e-7, 1e0])

    plot_line(axs[0], b_lmn)
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].grid(True, which="both")

    plot_line(axs[1], e_xyzfac_hf)
    labels = ["$E_{\\perp 1}$", "$E_{\\perp 2}$", "$E_{\\parallel}$"]
    axs[1].legend(labels, **cfg["figure"]["legend"])
    axs[1].set_ylabel("$E$ [mV m$^{-1}$]")
    axs[1].set_ylim([-9, 9])
    axs[1].set_yticks([-6, 0, 6])
    axs1b = axs[1].twinx()
    plot_line(axs1b, beta, color="k")
    axs1b.set_yscale("log")
    axs1b.set_ylabel("$\\beta$")
    axs1b.set_ylim([10 ** (.5) * 1e0, 10 ** (.5) * 1e3])
    axs[1].grid(True, which="both")

    axs[2], caxs2 = plot_spectr(axs[2], e_cwt, **cfg["figure"]["spectrum"])
    plot_line(axs[2], plasma_params.f_lh, color="k", label="$f_{lh}$")
    plot_line(axs[2], plasma_params.f_ce, color="w", label="$f_{ce}$")
    plot_line(axs[2], plasma_params.f_pp, color="red", label="$f_{pi}$")
    axs[2].set_ylim([0.5, 1e3])
    axs[2].set_ylabel("$f$ [Hz]")
    caxs2.set_ylabel("$E^2$ " + "\n" + "[mV$^2$ m$^{-2}$ Hz$^{-1}$]")
    axs[2].legend(**cfg["figure"]["legend"])

    axs[3], caxs3 = plot_spectr(axs[3], b_cwt, **cfg["figure"]["spectrum"])
    plot_line(axs[3], plasma_params.f_lh, color="k", label="$f_{lh}$")
    plot_line(axs[3], plasma_params.f_ce, color="w", label="$f_{ce}$")
    plot_line(axs[3], plasma_params.f_pp, color="red", label="$f_{pi}$")
    axs[3].set_ylim([0.5, 1e3])
    axs[3].set_ylabel("$f$ [Hz]")
    caxs3.set_ylabel("$B^2$ " + "\n" + "[nT$^2$ Hz$^{-1}$]")
    axs[3].legend(**cfg["figure"]["legend"])

    fig.align_ylabels(axs)

    axs[1].text(.85, .15, "$f > ${:2.1f} Hz".format(f_min),
                transform=axs[1].transAxes)

    # Time interval of the flapping
    axs[-1].set_xlim(mdates.date2num(tint_zoom))

    # Add panels labels
    labels_pos = [0.02, 0.90]
    _ = make_labels(axs, labels_pos)

    if args.figname:
        fig.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mms-id",
                        help="Spacecraft index",
                        choices=[1, 2, 3, 4],
                        type=int, required=True)

    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")

    parser.add_argument("--figname",
                        help="Path and name of the figure to save with extension.",
                        type=str, default="")

    parser.add_argument("--config",
                        help="Path to (.yml) config file.",
                        type=str, required=True)

    # Time interval
    main(parser.parse_args())
