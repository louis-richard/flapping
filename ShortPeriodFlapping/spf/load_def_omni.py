#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
load_def_omni.py

@author : Louis RICHARD
"""
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
