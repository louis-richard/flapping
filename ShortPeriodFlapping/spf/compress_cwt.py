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
import numba

@numba.jit(nopython=True, parallel=True)
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
