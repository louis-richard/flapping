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

from pyrfu.pyrf import mva


def lmn_cs(b_xyz, v_xyz):
    r"""Computes the flaring free coordinates system.

    Parameters
    ----------
    b_xyz : xarray.DataArray
        Time series of the magentic field at the center of mass of the
        tetrahedron.

    v_xyz : xarray.DataArray
        Time series of the ion bulk velocity at the center of mass of the
        tetrahedron.

    Returns
    -------
    lmn_ : ndarray
        LMN coordinates systems transformation matrix.

    """

    _, _, lmn = mva(b_xyz)

    # L is kept because \lambda_{max} >> \lambda_{int}
    l = lmn[:, 0]

    # M is taken to be the average ion bulk flow direction.
    m = np.median(v_xyz.data, axis=0)
    m /= np.linalg.norm(m, axis=0)

    # N completes the coordinates system
    n = np.cross(l, m) / np.linalg.norm(np.cross(l, m))

    # Adjust M to get a orthogonal coordinates system. M is the least
    # accurate between L and M
    m = np.cross(n, l)

    # Build transformation matrix
    lmn_ = np.transpose(np.vstack([l, m, n]))

    return lmn_
