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


def remove_bz_offset(b_mms):
    """
    Remove offset on Bz. The offset is computed using the time interval ["",""]
    """

    offset = np.array([0., 0.06997924, 0.11059547, -0.05232682])

    for i, b_xyz in enumerate(b_mms):
        b_xyz[:, 2] -= offset[i]

    return b_mms
