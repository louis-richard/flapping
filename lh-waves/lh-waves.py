#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lh-waves.py

@author : Louis RICHARD
"""

import yaml
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from astropy import constants
from dateutil import parser as date_parser
from pyrfu.mms import get_data, rotate_tensor
from pyrfu.plot import plot_line, plot_spectr
from pyrfu.pyrf import (avg_4sc, mva, time_clip, new_xyz, resample, trace,
                        norm, plasma_calc, convert_fac, filt, wavelet)

from spf import load_moments, remove_bz_offset, compress_cwt, make_labels


def main(args):
    """main function
    """
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint_flap = cfg["tints"]["flapping"]
    tint_zoom = cfg["tints"]["close-up"]

    mms_ids = np.arange(1, 5)
    
    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint_flap, i, args.verbose) for i in mms_ids]
    
    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Compute magnetic field at the center of mass of the tetrahedron
    b_xyz = avg_4sc(b_mms)

    # Compute MVA frame
    b_lmn, _, lmn = mva(b_xyz)

    # Correct Minimum Variance Frame
    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    b_xyz = time_clip(b_mms[args.mms_id-1], tint_zoom)
    b_lmn = new_xyz(b_xyz, lmn)

    # Load electric field and magnetic field fluctuation
    # Load electric field
    suf_e = "edp_{}_{}".format(cfg["edp"]["data_rate"], cfg["edp"]["level"])
    e_xyz = get_data("e_gse_{}".format(suf_e), tint_zoom, args.mms_id, args.verbose)

    # Load electric field
    suf_db = "scm_{}_{}".format(cfg["scm"]["data_rate"], cfg["scm"]["level"])
    b_scm = get_data("b_gse_{}".format(suf_db), tint_zoom, args.mms_id, args.verbose)

    # Load density and pressure tensor
    moments_i, moments_e = load_moments(tint_flap, cfg["fpi"], args)

    # Unpack moments
    _, _, _, p_xyz_i = moments_i
    n_e, _, _, p_xyz_e = moments_e

    # Total pressure tensor
    p_xyz = p_xyz_i + resample(p_xyz_e, p_xyz_i)

    # Convert to field-aligned coordinates
    p_xyzfac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Permittivity
    mu0 = constants.mu0.value

    # Thermal pressure (scalar)
    pth = 1e-9 * trace(p_xyzfac) / 3.

    # Magnetic pressure
    p_mag = 1e-18 * norm(b_xyz) ** 2 / (2 * mu0)

    # Plasma beta
    beta = resample(pth, b_xyz) / p_mag

    # Plasma parameters
    plasma_params = plasma_calc(b_xyz, n_e, n_e, n_e, n_e)

    # Convert electric field and magnetic field fluctuation to field aligned coordinates
    e_xyzfac = convert_fac(e_xyz, b_xyz, [1, 0, 0])

    # bandpass filter electric field and magnetic field fluctuation
    f_min = 4
    e_xyzfac_hf = filt(e_xyzfac, f_min, 0, 3)

    cwt_options = dict(nf=100, f=[0.5, 1e3], plot=False)

    # Compute continuous wavelet transform of the electric field
    e_xyzfac_cwt = wavelet(e_xyzfac, **cwt_options)

    # Construct compressed spectrogram of E
    e_cwt_times, e_cwt_x, e_cwt_y, e_cwt_z = compress_cwt(e_xyzfac_cwt, nc=100)

    options = dict(coords=[e_cwt_times, e_xyzfac_cwt.frequency], dims=["time", "frequency"])
    e_cwt = xr.DataArray(e_cwt_x + e_cwt_y + e_cwt_z, **options)

    # Compute continuous wavelet transform of the magnetic field fluctuations
    b_scm_cwt = wavelet(b_scm, **cwt_options)

    # Construct compressed spectrogram of E
    b_cwt_times, b_cwt_x, b_cwt_y, b_cwt_z = compress_cwt(b_scm_cwt, nc=100)

    options = dict(coords=[b_cwt_times, b_scm_cwt.frequency], dims=["time", "frequency"])
    b_cwt = xr.DataArray(b_cwt_x + b_cwt_y + b_cwt_z, **options)

    # Plot
    fig, axs = plt.subplots(4, sharex="all", figsize=(6.5, 9))
    fig.subplots_adjust(bottom=.05, top=.95, left=.15, right=.85, hspace=0.)

    kwargs_legend = dict(ncol=3, loc="upper right", frameon=True)
    kwargs_spectr = dict(cscale="log", yscale="log", cmap="Spectral_r", clim=[1e-7, 1e0])

    plot_line(axs[0], b_lmn[:, 0], "k")
    plot_line(axs[0], b_lmn[:, 1], "tab:blue")
    plot_line(axs[0], b_lmn[:, 2], "tab:red")
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].grid(True, which="both")

    plot_line(axs[1], e_xyzfac_hf[:, 0], "k")
    plot_line(axs[1], e_xyzfac_hf[:, 1], "tab:blue")
    plot_line(axs[1], e_xyzfac_hf[:, 2], "tab:red")
    labels = ["$E_{\\perp 1}$", "$E_{\\perp 2}$", "$E_{\\parallel}$"]
    axs[1].legend(labels, **cfg["figure"]["legend"])
    axs[1].set_ylabel("$E$ [mV m$^{-1}$]")
    axs[1].set_ylim([-9, 6])
    axs1b = axs[1].twinx()
    plot_line(axs1b, beta, "tab:green")
    axs1b.set_yscale("log")
    axs1b.set_ylabel("$\\beta$", color="tab:green")
    axs[1].grid(True, which="both")

    axs[2], caxs2 = plot_spectr(axs[2], e_cwt, **cfg["figure"]["spectrum"])
    plot_line(axs[2], plasma_params.f_lh, "k")
    plot_line(axs[2], plasma_params.f_ce, "w")
    plot_line(axs[2], plasma_params.f_pp, "red")
    axs[2].set_ylim([0.5, 1e3])
    axs[2].set_ylabel("$f$ [Hz]")
    caxs2.set_ylabel("$E^2$ " + "\n" + "[mV$^2$ m$^{-2}$ Hz$^{-1}$]")
    axs[2].legend(["$f_{lh}$", "$f_{ce}$", "$f_{pi}$"], **cfg["figure"]["legend"])

    axs[3], caxs3 = plot_spectr(axs[3], b_cwt, **cfg["figure"]["spectrum"])
    plot_line(axs[3], plasma_params.f_lh, "k")
    plot_line(axs[3], plasma_params.f_ce, "w")
    plot_line(axs[3], plasma_params.f_pp, "red")
    axs[3].set_ylim([0.5, 1e3])
    axs[3].set_ylabel("$f$ [Hz]")
    caxs3.set_ylabel("$B^2$ " + "\n" + "[nT$^2$ Hz$^{-1}$]")
    axs[3].legend(["$f_{lh}$", "$f_{ce}$", "$f_{pi}$"], **cfg["figure"]["legend"])

    fig.align_ylabels(axs)

    axs[1].text(.85, .15, "$f > ${:2.1f} Hz".format(f_min), transform=axs[1].transAxes)

    # Time interval of the flapping
    axs[-1].set_xlabel(date_parser.parse(tint_zoom[0]).strftime("%Y-%m-%d UTC"))
    axs[-1].set_xlim(tint_zoom)

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
