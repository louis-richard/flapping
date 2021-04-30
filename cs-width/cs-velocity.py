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
import matplotlib.dates as mdates

from cycler import cycler
from pyrfu.mms import get_data
from pyrfu.plot import plot_line
from pyrfu.pyrf import (c_4_j, mva, new_xyz, resample, medfilt,
                        ts_scalar, c_4_grad)

from spf import (load_timing, load_moments, remove_bz_offset,
                 dec_temperature, st_derivative, make_labels)

color = ["tab:blue", "tab:green", "tab:red", "k"]
default_cycler = cycler(color=color)
plt.rc('axes', prop_cycle=default_cycler)


def _downsample(b_xyz, dt):
    delta_t = (b_xyz.time.data[-1] - b_xyz.time.data[0]).view("i8") * 1e-9
    n_t = int(delta_t / dt)
    timeline = b_xyz.time.data[0].copy()
    timeline += (np.linspace(0, delta_t, n_t) * 1e9).astype(int)
    timeline = ts_scalar(timeline, np.zeros(len(timeline)))
    return resample(b_xyz, timeline)


def main(args):
    """main function
    """
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tints"]["flapping"]

    # Spacecraft indices
    mms_ids = np.arange(1, 5)

    # Load spacecraft position and background magnetic field
    r_mms = [get_data("R_gse", tint, i,
                      data_path=cfg["data_path"]) for i in mms_ids]
    
    # Load background magnetic field
    suf_b = "fgm_{}_{}".format(cfg["fgm"]["data_rate"], cfg["fgm"]["level"])
    b_mms = [get_data("B_gse_{}".format(suf_b), tint, i, args.verbose,
                      data_path=cfg["data_path"]) for i in mms_ids]

    # Remove offset on z component of the background magnetic field
    b_mms = remove_bz_offset(b_mms)

    # Load moments
    moments_i, moments_e = load_moments(tint, cfg["fpi"], args,
                                        cfg["data_path"])

    # Compute current density
    j_xyz, div_b, b_xyz, _, _, _ = c_4_j(r_mms, b_mms)
    j_xyz *= 1e9

    # Compute MVA frame
    _, _, lmn = mva(b_xyz)

    # Resample moments to magnetic field sampling
    moments_i = [resample(mom, b_xyz) for mom in moments_i]
    moments_e = [resample(mom, b_xyz) for mom in moments_e]

    # lmn = np.vstack([lmn[:, 0], -lmn[:, 2], lmn[:, 1]]).T
    l = lmn[:, 0]
    m = np.mean(moments_i[1].data, axis=0)
    m /= np.linalg.norm(m, axis=0)
    n = np.cross(l, m) / np.linalg.norm(np.cross(l, m))
    m = np.cross(n, l)
    lmn = np.transpose(np.vstack([l, m, n]))

    # transform magnetic field and current density to LMN coordinates system
    b_lmn, j_lmn = [new_xyz(field, lmn) for field in [b_xyz, j_xyz]]

    # Load data from timing
    timing_lr = load_timing(args.timing)

    # Transform slowness vector to LMN frame
    m_lmn, dm_lmn = [new_xyz(vec, lmn) for vec in [timing_lr.m, timing_lr.dm]]

    # Transform normal from timing to LMN coordinates system
    n_lmn = new_xyz(timing_lr.n, lmn)

    slowness = xr.Dataset({"m": m_lmn, "dm": dm_lmn})

    # Get crossing times
    crossing_times = m_lmn.time.data

    # Unpack ion/electron temperature
    _, _, t_i = dec_temperature(b_xyz, moments_i)
    _, _, t_e = dec_temperature(b_xyz, moments_i)

    v_xyz_i = moments_i[1]
    v_xyz_e = moments_e[1]

    v_lmn_i, v_lmn_e = [new_xyz(v_xyz, lmn) for v_xyz in [v_xyz_i, v_xyz_e]]

    # Compute velocity and geometry of the CS using spatio-temporal derivative
    b_mms_ds = [_downsample(b_xyz, 2.5) for b_xyz in b_mms]
    #v_str_lmn, y_m, z_n = st_derivative(r_mms, b_mms, lmn, crossing_times)
    v_str_lmn, y_m, z_n = st_derivative(r_mms, b_mms_ds, lmn, crossing_times)


    # filter velocity of the CS
    # change 257 to physical value
    #v_str_lmn_filtered = medfilt(v_str_lmn, 100)
    grad_b = c_4_grad(r_mms, b_mms_ds)
    grad_b_dete = ts_scalar(grad_b.time.data,
                            np.abs(np.linalg.det(grad_b.data)))
    res_dete = grad_b_dete.data - medfilt(grad_b_dete, 5).data
    idx_ = np.abs(res_dete) > np.std(res_dete)

    v_str_lmn_filtered = v_str_lmn.copy()
    v_str_lmn_filtered.data[idx_, 1] = np.nan
    v_str_lmn_filtered.data[idx_, 2] = np.nan

    # Plot
    fig, axs = plt.subplots(4, **cfg["figure"]["main"])
    fig.subplots_adjust(**cfg["figure"]["subplots"])

    plot_line(axs[0], b_lmn)
    axs[0].legend(["$B_L$", "$B_M$", "$B_N$"], **cfg["figure"]["legend"])
    axs[0].set_ylabel("$B$ [nT]")
    axs[0].grid(True, which="both")

    axs[1].quiver(timing_lr.tau.time.data, np.zeros(len(n_lmn)),
                  n_lmn[:, 1], n_lmn[:, 2],
                  color="tab:green", angles="uv")
    axs[1].set_ylabel("$n_{timing}^{MN}$")
    axs[1].grid(True, which="both")

    plot_line(axs[2], v_lmn_i[:, 1], color="tab:blue", label="Ions")
    plot_line(axs[2], v_str_lmn_filtered[:, 1], color="k", label="STD")
    axs[2].errorbar(slowness.time.data, slowness.m.data[:, 1],
                    slowness.dm.data[:, 1],
                    color="tab:green", label="Timing")
    axs[2].legend(**cfg["figure"]["legend"])
    axs[2].set_ylim([-650, 650])
    axs[2].set_ylabel("$V_M$ [km s$^{-1}$]")
    axs[2].grid(True, which="both")

    plot_line(axs[3], v_lmn_e[:, 2], color="tab:red", label="Electrons")
    plot_line(axs[3], v_str_lmn_filtered[:, 2], color="k", label="STD")
    axs[3].errorbar(slowness.time.data, slowness.m.data[:, 2],
                    slowness.dm.data[:, 2],
                    color="tab:green", label="Tining")
    axs[3].legend(**cfg["figure"]["legend"])
    axs[3].set_ylim([-650, 650])
    axs[3].set_ylabel("$V_N$ [km s$^{-1}$]")
    axs[3].grid(True, which="both")
    axs[-1].set_xlim(mdates.date2num(tint))
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
