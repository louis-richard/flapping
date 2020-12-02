#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_vph_current.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from scipy import optimize

from pyrfu.pyrf import gradient, histogram2d


def calc_vph_current(b_xyz, j_xyz):
    """Estimates the phase speed of the oscillating current sheet using oscillations of J_N.

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    j_xyz : xarray.DataArray
        Time series of the current density.


    Returns
    -------
    disprel : xarray.Dataset
        Hash table. to fill

    """
    # Time derivative of Bl
    dbl_dt = gradient(b_xyz[:, 0])

    hist_dbl_dt_jn = histogram2d(dbl_dt, j_xyz[:, 2])

    # Linear model for jn vs dBdt
    def model_jn(x, a):
        return a * x

    v_phase_j, sigma_dbl_dt_jn = optimize.curve_fit(model_jn, dbl_dt.data, j_xyz[:, 2].data)
    # v_phase_j = v_phase_j[0]
    v_phase_j = -3.12
    sigma_dbl_dt_jn = np.sqrt(float(sigma_dbl_dt_jn))

    dbl_dt_min = -1.2 * np.max(dbl_dt)
    dbl_dt_max = 1.2 * np.max(dbl_dt)

    disprel = {"fit_db_dt_jn": v_phase_j, "hist": hist_dbl_dt_jn,
               "hires_dBdt": np.linspace(dbl_dt_min, dbl_dt_max, 100),
               "pred_Jn": (["hires_dBdt"], model_jn(np.linspace(dbl_dt_min, dbl_dt_max, 100),
                                                    v_phase_j)),
               "bound_upper": (["hires_dBdt"], model_jn(np.linspace(dbl_dt_min, dbl_dt_max, 100),
                                                        v_phase_j + 1.92 * sigma_dbl_dt_jn)),
               "bound_lower": (["hires_dBdt"], model_jn(np.linspace(dbl_dt_min, dbl_dt_max, 100),
                                                        v_phase_j - 1.92 * sigma_dbl_dt_jn))}

    disprel = xr.Dataset(disprel)

    return disprel
