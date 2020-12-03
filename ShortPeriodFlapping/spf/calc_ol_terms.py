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

from astropy import constants
from pyrfu.pyrf import cross


def calc_ol_terms(b_xyz, j_xyz, moments_i, moments_e):
    """Compute terms of the Ohm's law.

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    j_xyz : xarray.DataArray
        Time series of the current density.

    moments_i : list of xarray.DataArray
        Time series of the moments of the ion VDF.

    moments_e : list of xarray.DataArray
        Time series of the moments of the electron VDF.

    Returns
    -------
    vxb_xyz_i : xarray.DataArray
        Time series of the ion convection electric field.

    vxb_xyz_e : xarray.DataArray
        Time series of the electron convection electric field.

    jxb_xyz : xarray.DataArray
        Time series of the Hall electric field.

    """

    # Unpack number density
    _, n_e = [moments_i[0], moments_e[0]]

    # Unpack bulk velocity
    v_xyz_i, v_xyz_e = [moments_i[1], moments_e[1]]

    # charge
    e = constants.e.value

    # Compute Ohm's law terms
    # ion/electron convection
    vxb_xyz_i, vxb_xyz_e = [1e-3 * cross(v_xyz, b_xyz) for v_xyz in [v_xyz_i, v_xyz_e]]

    # Hall term
    jxb_xyz = 1e-15 * cross(j_xyz, b_xyz) / (1e6 * n_e * e)

    return vxb_xyz_i, vxb_xyz_e, jxb_xyz
