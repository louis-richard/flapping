#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
st_derivative.py

@author : Louis RICHARD
"""

import numpy as np

from pyrfu.pyrf import c_4_grad, gradient, ts_vec_xyz, calc_dt, resample, t_eval, avg_4sc, ts_scalar


def st_derivative(r, b, mva, crossing_times):
    """
    Computes velocity of the structure using spatio-temporal derivative method
    """

    b_xyz = avg_4sc(b)

    # Gradient of the magnetic field
    grad_b = c_4_grad(r, b)
    db_dt = gradient(b_xyz)

    # Transform gradient to LMN frame
    l_grad_b = np.matmul(grad_b.data, mva[:, 0])
    m_grad_b = np.matmul(grad_b.data, mva[:, 1])
    n_grad_b = np.matmul(grad_b.data, mva[:, 2])

    # Compute velocity of the structure using MDD
    v_str = np.zeros(db_dt.shape)
    v_str[:, 0] = np.sum(db_dt * l_grad_b, axis=1) / np.linalg.norm(l_grad_b, axis=1) ** 2
    v_str[:, 1] = np.sum(db_dt * m_grad_b, axis=1) / np.linalg.norm(m_grad_b, axis=1) ** 2
    v_str[:, 2] = np.sum(db_dt * n_grad_b, axis=1) / np.linalg.norm(n_grad_b, axis=1) ** 2

    dt = calc_dt(b_xyz)
    y_m = np.abs(np.cumsum(-v_str[:, 1]) * dt)
    z_n = np.cumsum(-v_str[:, 2]) * dt

    v_str = ts_vec_xyz(b_xyz.time.data, -v_str)
    y_m = ts_scalar(b_xyz.time.data, y_m)
    z_n = ts_scalar(b_xyz.time.data, z_n)

    z_off = resample(t_eval(z_n, crossing_times), y_m)
    z_n -= z_off

    return v_str, y_m, z_n
