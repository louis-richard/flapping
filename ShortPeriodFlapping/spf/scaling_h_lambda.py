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


def scaling_h_lambda(h, dh, disprel, indices):
    """Compute scaling between the thickness and the wavelength of the current sheet
    """

    not_indices = np.delete(np.arange(len(h)), indices)

    # stack this events and other studies
    ks = disprel.k[not_indices]
    dks = disprel.k_err[not_indices]
    hs = h[not_indices]
    dhs = dh[not_indices]
    weights = 1 / disprel.k_err[not_indices]

    # Model scaling CS thickness vs wavelength
    def model_lamb_h(x, a):
        return a / x

    [mu_k_h, cov_k_h] = optimize.curve_fit(model_lamb_h, ks, hs, sigma=weights)
    mu_k_h = mu_k_h[0]
    sigma_k_h = np.sqrt(cov_k_h[0][0])

    scaling = {"k": ks, "dk": (["k"], dks), "h": (["k"], hs), "dh": (["k"], dhs),
               "hires_k": np.linspace(1e-4, 1.2 * np.max(ks)), "scaling": mu_k_h,
               "sigma_scaling": sigma_k_h, "k_ols":
                   disprel.k[indices].data,
               "h_ols": (["k_ols"], h[indices]), "dh_ols": (["k_ols"], dh[indices]),
               "dk_ols": (["k_ols"], disprel.k_err[indices].data)}
    scaling["pred_h"] = (["hires_k"], model_lamb_h(scaling["hires_k"], mu_k_h))
    scaling["bound_lower"] = (
    ["hires_k"], model_lamb_h(scaling["hires_k"], mu_k_h - 1.96 * sigma_k_h))
    scaling["bound_upper"] = (
    ["hires_k"], model_lamb_h(scaling["hires_k"], mu_k_h + 1.96 * sigma_k_h))

    out = xr.Dataset(scaling)

    return out
