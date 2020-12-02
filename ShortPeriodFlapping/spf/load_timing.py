
import h5py as h5
import numpy as np
import xarray as xr

from astropy.time import Time
from pyrfu.pyrf import ts_scalar, ts_vec_xyz


def load_timing(path="./timing/20190914_timing.h5"):
    """
    Loads results of timing computed in Matlab
    """

    with h5.File(path, "r") as f:
        tc = Time(f["tc"][0, ...], format="unix").datetime64

        # Period of the oscillations
        tau, dtau = [f["T"][0, ...], f["dT"][0, ...]]

        # Velocity of the structure
        v, dv = [f["v"][0, ...], f["dv"][0, ...]]

        try:
            # Normal to the current sheet
            n, dn = [f["n"][...], f["dn"][...]]

            # Thickness estimation
            h, dh = [np.zeros(tau.shape[0]), np.zeros(tau.shape[0])]

        except KeyError:
            n, dn = [np.zeros((tau.shape[0], 3)), np.zeros((tau.shape[0], 3))]

            # Thickness estimation
            h, dh = [f["h"][0, ...], f["dh"][0, ...]]

    # To time series
    # Period (semi-period ??), error on estimation of the period
    tau, dtau = [ts_scalar(tc, var) for var in [tau, dtau]]

    # Velocity of the structure at the crossing, error on estimation of the velocity
    v, dv = [ts_scalar(tc, var) for var in [v, dv]]

    # Normal direction to the CS at the crossing (direction of Vtm), error on the estimation of the
    # normal direction
    n, dn = [ts_vec_xyz(tc, var) for var in [n, dn]]

    # Current sheet thickness, error on the estimation of the current sheet thickness
    h, dh = [ts_scalar(tc, var) for var in [h, dh]]

    # Slowness vector
    m_xyz = v * n

    # Propagation of the error on the slowness vector
    dm_xyz = (v + dv) * (n + dn) - m_xyz

    out_dict = {"tau": tau, "dtau": dtau, "v": v, "dv": dv, "n": n, "dN": dn, "m": m_xyz,
                "dm": dm_xyz, "h": h, "dh": dh}

    out = xr.Dataset(out_dict)

    return out