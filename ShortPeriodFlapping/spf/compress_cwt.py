#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compress_cwt.py

@author : Louis RICHARD
"""

import numpy as np


def compress_cwt(cwt=None, nc=100):
    """Compress the wavelet transform averaging of nc time steps.

    Parameters
    ----------
    cwt : xarray.DataArray
        Wavelet transform to compress.

    nc : int
        Number of time steps for averaging.


    Returns
    -------
    cwt_times : xarray.DataArray
        Sampling times.

    cwt_x : numpy.ndarray
        Compressed wavelet transform of the first component of the field.

    cwt_y : numpy.ndarray
        Compressed wavelet transform of the second component of the field.

    cwt_z : numpy.ndarray
        Compressed wavelet transform of the third component of the field.

    """
    assert cwt is not None

    # Number of frequencies
    nf = cwt.x.shape[1]

    idxs = np.arange(int(nc / 2), len(cwt.time) - int(nc / 2), step=nc).astype(int)

    cwt_times = cwt.time[idxs]

    cwt_x, cwt_y, cwt_z = [np.zeros((len(idxs), nf)) for _ in range(3)]

    for i, idx in enumerate(idxs):
        cwt_x[i, :] = np.squeeze(
            np.nanmean(cwt.x[idx - int(nc / 2) + 1:idx + int(nc / 2), :], axis=0))
        cwt_y[i, :] = np.squeeze(
            np.nanmean(cwt.y[idx - int(nc / 2) + 1:idx + int(nc / 2), :], axis=0))
        cwt_z[i, :] = np.squeeze(
            np.nanmean(cwt.z[idx - int(nc / 2) + 1:idx + int(nc / 2), :], axis=0))

    return cwt_times, cwt_x, cwt_y, cwt_z
