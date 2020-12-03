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
