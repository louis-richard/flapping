#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dec_temperature.py

@author : Louis RICHARD
"""

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
