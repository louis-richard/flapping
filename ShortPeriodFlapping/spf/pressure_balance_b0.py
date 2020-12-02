#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pressure_balance_b0.py

@author : Louis RICHARD
"""

import numpy as np

from astropy import constants
from pyrfu.mms import rotate_tensor
from pyrfu.pyrf import resample, trace, norm


def pressure_balance_b0(b_xyz=None, moments_i=None, moments_e=None):
    """Compute lobe magnetic field using pressure balance condition

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    moments_i : list of xarray.DataArray
        Time series of the moments of the ion VDF.

    moments_e : list of xarray.DataArray
        Time series of the moments of the electron VDF.

    Returns
    -------
    b0 : xarray.DataArray
        Time series of the lobe field.

    """
    # Unpack pressure tensors
    p_xyz_i, p_xyz_e = [moments_i[-1], moments_e[-1]]

    # Compute total pressure tensor
    p_xyz = p_xyz_i + resample(p_xyz_e, p_xyz_i)

    # Transform to field aligned coordinates
    p_xyzfac = rotate_tensor(p_xyz, "fac", b_xyz, "pp")

    # Estimate magnetic field in the lobe using pressure balance
    mu0 = constants.mu0.value  # magnetic permittivity

    p_th, p_b = [1e-9 * trace(p_xyzfac), 1e-18 * norm(b_xyz) ** 2 / (2 * mu0)]

    # Compute plasma parameter
    beta = resample(p_th, p_b) / p_b

    # Compute lobe field using pressure balance
    b0 = b_xyz * np.sqrt(1 + beta)

    return b0
