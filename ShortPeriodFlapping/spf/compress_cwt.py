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

import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def _compress_cwt_1d(cwt, nc: int = 100):
    nf = cwt.shape[1]
    idxs = np.arange(start=int(nc / 2), stop=len(cwt) - int(nc / 2),
                     step=nc, dtype=np.int64)
    cwt_c = np.zeros((len(idxs), nf))

    for i, idx in enumerate(idxs):
        for j in range(nf):
            x_data = cwt[idx - int(nc / 2):idx + int(nc / 2), j]
            cwt_c[i, j] = np.nanmean(x_data)

    return cwt_c


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
    cwt_t : xarray.DataArray
        Sampling times.

    cwt_x : numpy.ndarray
        Compressed wavelet transform of the first component of the field.

    cwt_y : numpy.ndarray
        Compressed wavelet transform of the second component of the field.

    cwt_z : numpy.ndarray
        Compressed wavelet transform of the third component of the field.

    """

    indices = np.arange(int(nc / 2), len(cwt.time.data) - int(nc / 2),
                        step=nc, dtype=np.int64)

    cwt_t = cwt.time.data[indices]
    cwt_x = _compress_cwt_1d(cwt.x.data, nc=nc)
    cwt_y = _compress_cwt_1d(cwt.y.data, nc=nc)
    cwt_z = _compress_cwt_1d(cwt.z.data, nc=nc)

    return cwt_t, cwt_x, cwt_y, cwt_z
