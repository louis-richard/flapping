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

import yaml
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pyrfu.mms import get_data
from pyrfu.plot import plot_line
from pyrfu.pyrf import (c_4_j, mva, new_xyz, resample, medfilt)

from spf import (load_timing, load_moments, remove_bz_offset,
                 dec_temperature, st_derivative, make_labels)


def main(args):
    """main function
    """
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tints"]["flapping"]

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint, i) for i in mms_ids]
    
    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint, i, args.verbose) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Load moments
    moments_i, moments_e = load_moments(tint, cfg["fpi"], args)

    # Compute current density
    j_xyz, div_b, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9  														# j A.m^{-2}->nA.m^{-2}

    # Compute MVA frame
    _, _, lmn = mva(b_xyz)

    lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T

    # transform magnetic field and current density to LMN coordinates system
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    # Resample moments to magnetic field sampling
    moments_i = [resample(mom, b_xyz) for mom in moments_i]
    moments_e = [resample(mom, b_xyz) for mom in moments_e]

    # Load data from timing
    timing_lr = load_timing(args.timing)

    # Transform slowness vector to LMN frame
    m_lmn, dm_lmn = [new_xyz(field, lmn) for field in [timing_lr.m, timing_lr.dm]]

    slowness = xr.Dataset({"m": m_lmn, "dm": dm_lmn})

    # Get crossing times
    crossing_times = m_lmn.time.data

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_i)

    # Compute velocity and geometry of the CS using spatio-temporal derivative
    v_str_lmn, y_m, z_n = st_derivative(r_mms, b_mms, lmn, crossing_times)

    v_xyz_i = moments_i[1]
    v_xyz_e = moments_e[1]

    v_lmn_i, v_lmn_e = [new_xyz(v_xyz, lmn) for v_xyz in [v_xyz_i, v_xyz_e]]

    # filter velocity of the CS
    v_str_lmn_filtered = medfilt(v_str_lmn, 100)  # change 257 to physical value

    # Plot
    fig, axs = plt.subplots(3, **cfg["figure"]["main"])
    fig.subplots_adjust(**cfg["figure"]["subplots"])

    plot_line(axs[0], b_lmn)
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].grid(True, which="both")

    plot_line(axs[1], v_lmn_i[:, 1], "tab:blue")
    plot_line(axs[1], v_str_lmn_filtered[:, 1], "k")
    axs[1].errorbar(slowness.time.data, slowness.m.data[:, 1], slowness.dm.data[:, 1],
                    color="tab:green")
    axs[1].legend(["Ions", "STD", "Timing"], **cfg["figure"]["legend"])
    axs[1].set_ylim([-650, 650])
    axs[1].set_ylabel("$V_M$ [km s$^{-1}$]")
    axs[1].grid(True, which="both")

    plot_line(axs[2], v_lmn_e[:, 2], "tab:red")
    plot_line(axs[2], v_str_lmn_filtered[:, 2], "k")
    axs[2].errorbar(slowness.time.data, slowness.m.data[:, 2], slowness.dm.data[:, 2],
                    color="tab:green")
    axs[2].legend(["Electrons", "STD", "Timing"], **cfg["figure"]["legend"])
    axs[2].set_ylim([-650, 650])
    axs[2].set_ylabel("$V_N$ [km s$^{-1}$]")
    axs[2].grid(True, which="both")

    axs[-1].set_xlabel("2019-09-14 UTC")
    axs[-1].set_xlim(tint)
    fig.align_ylabels(axs)

    labels_pos = [0.02, 0.92]
    _ = make_labels(axs, labels_pos)

    if args.figname:
        fig.savefig(args.figname, **cfg["figure"]["save"])
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity",
                        action="store_true")

    parser.add_argument("--config",
                        help="Path to (.yml) config file.",
                        type=str, required=True)

    parser.add_argument("--timing",
                        help="Path to (.h5) timing file.",
                        type=str, required=True)

    parser.add_argument("--figname",
                        help="Path and name of the figure to save with extension.",
                        type=str, default="")

    main(parser.parse_args())
