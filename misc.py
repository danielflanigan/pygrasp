from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from pygrasp.flat_map import GridMap
from pygrasp.convert import Convert


def plot_radial_cut(grid_map, component, angle, center=(0, 0)):
    x_deg, y_deg, cut = grid_map.cut(grid_map.dB(component), angle, center=center, single_sided=True)
    r_deg = np.sqrt(x_deg**2 + y_deg**2)
    plt.ioff()
    fig = plt.figure()
    plt.plot(r_deg, cut)
    plt.xlabel('[degrees]')
    plt.ylabel('[dB]')
    plt.ion()
    plt.show()
    return fig

# Add support for double-sided cuts?
def plot_radial_cut_uv(grid_map, component, angle, center=(0, 0)):
    x_uv, y_uv, cut = grid_map.cut(grid_map.dB(component), angle, center=center, single_sided=True)
    x_deg = np.degrees(np.sign(x_uv) * Convert.u_v_to_theta_phi(x_uv, 0)[0])
    y_deg = np.degrees(np.sign(y_uv) * Convert.u_v_to_theta_phi(0, y_uv)[0])
    r_deg = np.sqrt(x_deg**2 + y_deg**2)
    plt.ioff()
    fig = plt.figure()
    plt.plot(r_deg, cut)
    plt.xlabel('[degrees]')
    plt.ylabel('[dB]')
    plt.ion()
    plt.show()
    return fig

def mirror_x(half):
    full = GridMap()
    full.shape = half.shape
    full.x = np.concatenate((-half.x[::-1], half.x[1:]))
    full.y = half.y
    full.map = np.empty((full.shape[0], full.shape[1],
                         full.x.size, full.y.size),
                        dtype=np.complex)
    full.map[:, :, :half.x.size, :] = half.map[:, :, ::-1, :]
    full.map[:, :, half.x.size:, :] = half.map[:, :, 1:, :]
    return full

# This is useful enough that it should be somewhere else.
def u_v_to_degrees(gm):
    gm.x = np.degrees(Convert.u_v_to_theta_phi(gm.x, 0)[0] * np.sign(gm.x))
    gm.y = np.degrees(Convert.u_v_to_theta_phi(gm.y, 0)[0] * np.sign(gm.y))
    return gm
    
def normalize(gm):
    gm.map = gm.map / np.max(np.abs(gm.map))
    return gm

def combine_and_normalize(co, cx):
    if any(co.x != cx.x) or any(co.y != cx.y):
        raise ValueError("Mismatch in x or y values.")
    if co.map.shape != cx.map.shape:
        raise ValueError("Mismatch in map shapes.")
    both = GridMap()
    both.x = co.x
    both.y = cx.y
    both.map = np.empty((co.map.shape), dtype=np.complex)
    maximum = np.max(np.abs(co.map[0]))
    both.map[0, 0] = co.map[0, 0] / maximum
    both.map[1, 0] = cx.map[0, 0] / maximum
    return both

# For y-polarization. Generalize.
def plot_E_and_H_for_coupling_and_far(coupling, far, frequency):
    fig = plt.figure(figsize=(7, 7), dpi=160)
    x_c = coupling.x * 60
    y_c = coupling.y * 60
    ic_x0 = np.floor(x_c.size / 2)
    ic_y0 = np.floor(y_c.size / 2)
    coupling_E = coupling.dB(0)[ic_x0, :]
    plt.plot(y_c, coupling_E, 'r', label="coupling E")
    coupling_H = coupling.dB(0)[:, ic_y0]
    plt.plot(x_c, coupling_H, 'm', label="coupling H")
    x_f = far.x * 60
    y_f = far.y * 60
    if_x0 = np.floor(x_c.size / 2)
    if_y0 = np.floor(y_c.size / 2)
    far_E = far.dB(1)[if_x0, :]
    plt.plot(y_f, far_E, 'b', label="far field E")
    far_H = far.dB(1)[:, if_y0]
    plt.plot(x_f, far_H, 'g', label="far field H")
    plt.title("{} E-plane and H-plane cuts; FWHM values are\nCoupling E: {:.1f}'  Coupling H: {:.1f}'  Far E: {:.1f}'  Far H: {:.1f}'" \
                  .format(frequency, FWHM(y_c, coupling_E), FWHM(x_c, coupling_H), FWHM(y_f, far_E), FWHM(x_f, far_H)))
    plt.xlabel('[arcminutes]')
    plt.ylabel('[dB]')
    plt.xlim(-30, 30)
    plt.ylim(-60, 2)
    plt.legend(loc=8)
    plt.grid()
    return fig

def plot_coupling_radial_cuts(coupling, title, hor_label, vert_label, cx_label):
    fig = plt.figure(figsize=(7, 7), dpi=160)
    r_hor, co_hor = coupling.radial_cut(0, 0)
    r_vert, co_vert = coupling.radial_cut(0, np.pi/2)
    r_cx, cx = coupling.radial_cut(1, np.pi/4)
    plt.plot(r_hor * 60, co_hor, label=hor_label)
    plt.plot(r_vert * 60, co_vert, label=vert_label)
    plt.plot(r_cx * 60, cx, label=cx_label)
    plt.xlim(0, 30)
    plt.ylim(-60, 2)
    plt.xlabel('[arcminutes]')
    plt.ylabel('[dB]')
    plt.title(title)
    plt.legend(loc=1)
    plt.grid()
    return fig
