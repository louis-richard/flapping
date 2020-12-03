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

import numpy as np
import xarray as xr

from scipy import optimize
from pyrfu.pyrf import ts_vec_xyz, histogram2d, median_bins


def fit_harris_cs(b_xyz, j_xyz):
    """Compute the Harris fit of the current density with respect to the magnetic field. USefull
    to get the lobe field and the Harris scale.

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    j_xyz : xarray.DataArray
        Time series of the current density.

    Returns
    -------
    harris : xarray.Dataset
        Hash table. to fill

    """
    j_perp = np.sqrt(j_xyz[:, 1] ** 2 + j_xyz[:, 2] ** 2)

    # Fit J vs B using Harris model
    def mod(x, a):
        return a * (1 - (x / 15) ** 2)

    opt, g = optimize.curve_fit(mod, b_xyz[:, 0], j_perp)

    b_lobe = 15
    harris_scale = opt[0]
    sigma_scale = np.sqrt(float(g))

    b0 = np.zeros((len(b_xyz), 3))
    b0[:, 0] = b_lobe * np.ones(len(b_xyz))
    b0 = ts_vec_xyz(b_xyz.time.data, b0)

    hist_b_j_mn = histogram2d(b_xyz[:, 0], j_perp, bins=100)  	# 2D histogram
    med_b_j_mn = median_bins(b_xyz[:, 0], j_perp, bins=10)  	# Median

    harris = {"B0": b0, "hist": hist_b_j_mn, "bbins": med_b_j_mn.bins.data,
              "medbin": (["bbins"], med_b_j_mn.data.data),
              "medstd": (["bbins"], med_b_j_mn.sigma.data),
              "hires_b": np.linspace(np.min(b_xyz[:, 0]), np.max(b_xyz[:, 0]), 100, endpoint=True)}

    harris["pred_j_perp"] = (["hires_b"], mod(harris["hires_b"], harris_scale))
    harris["bound_upper"] = (["hires_b"], mod(harris["hires_b"], harris_scale + 1.96 * sigma_scale))
    harris["bound_lower"] = (["hires_b"], mod(harris["hires_b"], harris_scale - 1.96 * sigma_scale))

    harris = xr.Dataset(harris)

    return harris
