#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
load_def_omni.py

@author : Louis RICHARD
"""
import numpy as np
import xarray as xr
import h5py as h5

from astropy.time import Time
from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc, ts_scalar, ts_vec_xyz


def load_def_omni(tint, cfg):
    """Loads Density Energy Flux spectrum
    """
    ic = np.arange(1, 5)

    suf = "fpi_{}_{}".format(cfg["data_rate"], cfg["level"])

    # Ion/electron omni directional energy flux
    def_omni_mms_i = [get_data("DEFi_{}".format(suf), tint, i) for i in ic[:-1]]
    def_omni_mms_e = [get_data("DEFe_{}".format(suf), tint, i) for i in ic[:-1]]

    def_omni_i, def_omni_e = [avg_4sc(def_omni) for def_omni in [def_omni_mms_i, def_omni_mms_e]]

    return def_omni_i, def_omni_e


def load_moments(tint, cfg, args):
    """
    Load FPI moments of the velocity distribution functions.

    If the option moments is set to "part" then use partial moments instead.
    """
    ic = np.arange(1, 5)

    suf = "fpi_{}_{}".format(cfg["data_rate"], cfg["level"])

    if cfg["moments"] == "partial":
        # index to split partial moments (from quasi-neutrality assumption)
        part_idx_i, part_idx_e = [cfg[f"part_idx_{s}"] for s in ["i", "e"]]

        # Load partial moments
        # number density
        part_n_i = [get_data("partNi_{}".format(suf), tint, i, args.verbose) for i in
                    ic[:-1]]
        part_n_e = [get_data("partNe_{}".format(suf), tint, i, args.verbose) for i in
                    ic[:-1]]

        # bulk velocity
        part_v_i = [get_data("partVi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_v_e = [get_data("partVe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # temperature tensor
        part_t_i = [get_data("partTi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_t_e = [get_data("partTe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # pressure tensor
        part_p_i = [get_data("partPi_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]
        part_p_e = [get_data("partPe_gse_{}".format(suf), tint, i, args.verbose) for i
                    in ic[:-1]]

        # split partial moments
        # number density
        n_i = [part_n_i[i - 1][:, part_idx_i] for i in ic[:-1]]
        n_e = [part_n_e[i - 1][:, part_idx_e] for i in ic[:-1]]

        # bulk velocity
        v_i = [part_v_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        v_e = [part_v_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]

        # temperature tensor
        t_i = [part_t_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        t_e = [part_t_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]

        # pressure tensor
        p_i = [part_p_i[i - 1][:, part_idx_i, ...] for i in ic[:-1]]
        p_e = [part_p_e[i - 1][:, part_idx_e, ...] for i in ic[:-1]]
    elif cfg["moments"] == "full":
        # number density
        n_i = [get_data("Ni_{}".format(suf), tint, i, args.verbose) for i in ic[:-1]]
        n_e = [get_data("Ne_{}".format(suf), tint, i, args.verbose) for i in ic[:-1]]

        # bulk velocity
        v_i = [get_data("Vi_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        v_e = [get_data("Ve_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]

        # temperature tensor
        t_i = [get_data("Ti_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        t_e = [get_data("Te_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]

        # pressure tensor
        p_i = [get_data("Pi_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
        p_e = [get_data("Pe_gse_{}".format(suf), tint, i, args.verbose) for i in
               ic[:-1]]
    else:
        raise ValueError("Invalid moments type")

    # Load spintone correction
    spintone_i = [get_data("STi_gse_{}".format(suf), tint, i, args.verbose) for i in
                  ic[:-1]]
    spintone_e = [get_data("STe_gse_{}".format(suf), tint, i, args.verbose) for i in
                  ic[:-1]]

    # remove spintone correction
    v_i = [v - spintone_i[i] for i, v in enumerate(v_i)]
    v_e = [v - spintone_e[i] for i, v in enumerate(v_e)]

    moments_i = [n_i, v_i, t_i, p_i]
    moments_e = [n_e, v_e, t_e, p_e]

    moments_i = [avg_4sc(moment) for moment in moments_i]
    moments_e = [avg_4sc(moment) for moment in moments_e]

    return moments_i, moments_e


def load_timing(path="./timing/20190914_timing.h5"):
    """
    Loads results of timing computed in Matlab
    """

    with h5.File(path, "r") as f:
        tc = Time(f["tc"][0, ...], format="unix").datetime64

        # Period of the oscillations
        tau, dtau = [f["T"][0, ...], f["dT"][0, ...]]

        # Velocity of the structure
        v, dv = [f["v"][0, ...], f["dv"][0, ...]]

        try:
            # Normal to the current sheet
            n, dn = [f["n"][...], f["dn"][...]]

            # Thickness estimation
            h, dh = [np.zeros(tau.shape[0]), np.zeros(tau.shape[0])]

        except KeyError:
            n, dn = [np.zeros((tau.shape[0], 3)), np.zeros((tau.shape[0], 3))]

            # Thickness estimation
            h, dh = [f["h"][0, ...], f["dh"][0, ...]]

    # To time series
    # Period (semi-period ??), error on estimation of the period
    tau, dtau = [ts_scalar(tc, var) for var in [tau, dtau]]

    # Velocity of the structure at the crossing, error on estimation of the velocity
    v, dv = [ts_scalar(tc, var) for var in [v, dv]]

    # Normal direction to the CS at the crossing (direction of Vtm), error on the estimation of the
    # normal direction
    n, dn = [ts_vec_xyz(tc, var) for var in [n, dn]]

    # Current sheet thickness, error on the estimation of the current sheet thickness
    h, dh = [ts_scalar(tc, var) for var in [h, dh]]

    # Slowness vector
    m_xyz = v * n

    # Propagation of the error on the slowness vector
    dm_xyz = (v + dv) * (n + dn) - m_xyz

    out_dict = {"tau": tau, "dtau": dtau, "v": v, "dv": dv, "n": n, "dN": dn, "m": m_xyz,
                "dm": dm_xyz, "h": h, "dh": dh}

    out = xr.Dataset(out_dict)

    return out
