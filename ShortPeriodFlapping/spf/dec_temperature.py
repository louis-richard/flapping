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

from pyrfu.mms import rotate_tensor
from pyrfu.pyrf import trace


def dec_temperature(b_xyz, moments):
    """
    Decomposes temperature tensor from GSE to para/perp/tot
    """

    t_xyz = moments[2]

    t_xyzfac = rotate_tensor(t_xyz, "fac", b_xyz, "pp")

    t_para, t_perp, t_tot = [t_xyzfac[:, 0, 0], t_xyzfac[:, 1, 1], trace(t_xyzfac) / 3]

    return t_para, t_perp, t_tot
