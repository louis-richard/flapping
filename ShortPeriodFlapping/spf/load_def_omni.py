# Copyright 2020 Louis Richard
#
# Licensed under the MIT Open Source License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc


def load_def_omni(tint, cfg):
    """Loads Density Energy Flux spectrum

    Parameters
    ----------
    tint : list of str
        Time interval

    cfg : dict
        Hash table from configuration file.

    Returns
    -------


    """
    ic = np.arange(1, 5)

    suf = "fpi_{}_{}".format(cfg["data_rate"], cfg["level"])

    # Ion/electron omni directional energy flux
    def_omni_mms_i = [get_data("DEFi_{}".format(suf), tint, i) for i in ic[:-1]]
    def_omni_mms_e = [get_data("DEFe_{}".format(suf), tint, i) for i in ic[:-1]]

    def_omni_i, def_omni_e = [avg_4sc(def_omni) for def_omni in [def_omni_mms_i, def_omni_mms_e]]

    return def_omni_i, def_omni_e
