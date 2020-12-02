#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remove_bz_offset.py

@author : Louis RICHARD
"""

import numpy as np


def remove_bz_offset(b_mms):
    """
    Remove offset on Bz. The offset is computed using the time interval ["",""]
    """

    offset = np.array([0., 0.06997924, 0.11059547, -0.05232682])

    for i, b_xyz in enumerate(b_mms):
        b_xyz[:, 2] -= offset[i]

    return b_mms
