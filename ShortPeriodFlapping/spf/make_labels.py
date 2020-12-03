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

import string


def make_labels(axs, pos, pad=0):
    lbl = string.ascii_lowercase[pad:len(axs) + pad]

    for label, axis in zip(lbl, axs):
        axis.text(pos[0], pos[1], "({})".format(label), transform=axis.transAxes)

    return axs