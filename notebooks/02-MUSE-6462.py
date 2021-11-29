# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Extract C II 6462 line from MUSE cube

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
from numpy.polynomial import Chebyshev as T
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

# We have two versions of the cont-subtracted MUSE cubes:
# * Rebinned 16x16 cubes are from the Raman project
# * Full-resolution cubes are too big for Dropbox, so stored elsewhere

RAW_DATA_PATH = Path.home() / "Dropbox" / "dib-scatter-hii" / "data" / "orion-muse"
RAW_BIG_DATA_PATH = Path.home() / "Work" / "Muse-Hii-Data" / "M42"

# ## First try with the smaller cube

hdu = fits.open(
    RAW_DATA_PATH / "muse-hr-data-wavsec23-rebin16x16-cont-sub.fits"
)["DATA"]

w = WCS(hdu)
w

# ### Find the wavelength pixel that corresponds to the rest wavelength of the C II line
#
# AtLL gives 2 components: 6461.95 and 6462.13, with $g_k A_{ki}$-weighted wavelength of 6462.05:

wav0_6462 = 6462.05 * u.Angstrom
k0 = int(w.spectral.world_to_pixel(wav0_6462))
k0

# Take a +/- 50 pixel window around rest wavelength pixel:

NWIN = 100
window_slice = slice(k0 - (NWIN//2), k0 + (NWIN//2) + 1)
window_cube = hdu.data[window_slice, ...]

# Make an array of wavelengths for the entire cube, and then select the wavs of the window

VOMC = 20.0

nwavs, ny, nx = hdu.data.shape
wavs = w.spectral.array_index_to_world(np.arange(nwavs)).Angstrom
wavs *= (1.0 - VOMC / 3e5)
window_wavs = wavs[window_slice]
window_wavs

# Look at the average profile

window_norm = window_cube / np.median(window_cube, axis=0)
window_median = np.median(window_norm, axis=(1, 2))
window_mad = np.median(
    np.abs(window_norm - window_median[:, None, None]), 
    axis=(1, 2)
)

# Also calculate the sky profile, since I am now convinced that most of the other lines are sky.

xskyslice, yskyslice = slice(0, 20), slice(55, 76)
window_sky = np.median(window_norm[:, yskyslice, xskyslice], axis=(1, 2))

# Get some potential line IDs from Fang 2011

# + tags=[]
line_ids = {
    6425.9: "O II",
    6441.295: "[Ni II]",
    6445.81: "Ne II",
#    6451.97: "V II",
    6456.38: "Fe II",
#    6461.95: "C II",
    6462.05: "C II", # mean doublet wavelength
#    6462.13: "C II",
    6466.07: "Fe II",
#    6468.8: "C II",
#    6467.288: "[Ni II]",
#    6469.213: "[Co II]",
    6471.42: "O II",
#    6471.91: "C II",
    6478.72: "N II",
#    6485.2983: "[Fe II]",
#    6483.97: "O II",
#    6485.06: "O II",
#    6491.91: "O II",
    6481.706: " ",
    6482.699: " ",
    6483.753: "N I",
    6484.808: " ",
#    6491.222: "N I",
#    6499.518: " ",
    6501.41: "O II",
#    6506.302: "N I",
#    6507.024: "",
#    6521.110: "N I",
}


# +
def wav2k(wav):
    return w.spectral.world_to_array_index(
         np.asarray(wav) * u.Angstrom
    )

def k2wav(k):
    if len(k):
        return w.spectral.array_index_to_world(
            np.atleast_1d(k)
        ).Angstrom
    else:
        return np.array([])


# -

wav2k(list(line_ids.keys()))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(window_wavs, window_median)
ax.fill_between(
    window_wavs, 
    window_median - window_mad,
    window_median + window_mad,
    color="k",
    alpha=0.1,
    linewidth=0,
    zorder=-1,
)
ax.plot(window_wavs, window_sky)
toggle = 1.0
for wav, label in line_ids.items():
    ax.annotate(
        label, 
        (wav, 2.5), 
        xytext=(0, 30 + 10 * toggle), 
        textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="-"),
    )
    toggle *= -1
    

# Now try subtracting the average sky spectrum:

sky = np.median(
    window_cube[:, yskyslice, xskyslice], 
    axis=(1, 2),
    keepdims=True,
)
window_norm = (window_cube - sky) / np.median(window_cube, axis=0)
window_median = np.median(window_norm, axis=(1, 2))
window_mad = np.median(
    np.abs(window_norm - window_median[:, None, None]), 
    axis=(1, 2)
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(window_wavs, window_median)
ax.fill_between(
    window_wavs, 
    window_median - window_mad,
    window_median + window_mad,
    color="k",
    alpha=0.1,
    linewidth=0,
    zorder=-1,
)
ax.plot(window_wavs, 0.5 * sky[:, 0, 0] / np.max(sky))
toggle = 1.0
for wav, label in line_ids.items():
    ax.annotate(
        label, 
        (wav, 1.9), 
        xytext=(0, 30 + 10 * toggle), 
        textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="-"),
    )
    toggle *= -1
    

# Yes, that looks a lot better!  So most of the line IDs are not detected at all. It looks very clean around the C II line, with the possible exception of Fe II.  Then, to the red we have a possible detection of the N I multiplet at 6485 and O II at 6502

# Now plot against array index 

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wav2k(window_wavs), window_median, marker=".")
ax.set(
    xlim=[770, 810],
)
ax.grid(axis="x")

# 770-787 blue continuum. 
#
# 796-810 red continuum 
#
# 788-795 covers the line
#
# Try it out:

myav = np.mean
fullsky = myav(
    hdu.data[:, yskyslice, xskyslice], 
    axis=(1, 2),
    keepdims=True,
)
fullcube_nosky = hdu.data - fullsky
cont_map_blue = myav(fullcube_nosky[770:788, ...], axis=0)
cont_map_red = myav(fullcube_nosky[796:810, ...], axis=0)
cont_map = myav(
    np.stack([cont_map_red, cont_map_blue]),
    axis=0,
)
cii_map = (fullcube_nosky[788:796, ...] - cont_map).sum(axis=0)

fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(cii_map, vmin=-3e4, vmax=6e5, origin="lower", cmap="gray_r")
ax.set_title("C II 6462");

fig, ax = plt.subplots(figsize=(12,10))
smooth = 5
ax.imshow(
    median_filter(cii_map, size=smooth), 
    vmin=-3e4, vmax=6e5, origin="lower", cmap="gray_r")
ax.set_title("C II 6462");

fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(cont_map, vmin=-2e3, vmax=1e5, origin="lower", cmap="gray_r")
ax.set_title("Continuum that is mainly Raman wing");

fig, ax = plt.subplots(figsize=(12,10))
smooth = 4
ax.imshow(
    median_filter(cont_map, size=smooth), 
    vmin=-2e3, vmax=1e5, origin="lower", cmap="gray_r")
ax.set_title("Continuum that is mainly Raman wing");

# ## Compare with H alpha

wav0_6563 = 6562.79 * u.Angstrom
k0 = int(w.spectral.world_to_pixel(wav0_6563))
k0

NWIN = 100
window_slice = slice(k0 - (NWIN//2), k0 + (NWIN//2) + 1)
window_cube = hdu.data[window_slice, ...]
window_wavs = wavs[window_slice]
window_wavs

sky = np.median(
    window_cube[:, yskyslice, xskyslice], 
    axis=(1, 2),
    keepdims=True,
)
window_norm = (window_cube - sky) / np.median(window_cube, axis=0)
window_median = np.median(window_norm, axis=(1, 2))
window_mad = np.median(
    np.abs(window_norm - window_median[:, None, None]), 
    axis=(1, 2)
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(window_wavs, window_median)
ax.fill_between(
    window_wavs, 
    window_median - window_mad,
    window_median + window_mad,
    color="k",
    alpha=0.1,
    linewidth=0,
    zorder=-1,
)
ax.plot(window_wavs, window_mad)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wav2k(window_wavs), window_median, marker=".")
ax.set(
    xlim=[k0 - 20, k0 + 20],
    ylim=[0, 5],
)
ax.grid(axis="x")

cont_map_blue = myav(fullcube_nosky[898:901, ...], axis=0)
cont_map_red = myav(fullcube_nosky[922:925, ...], axis=0)
cont_map_ha = myav(
    np.stack([cont_map_red, cont_map_blue]),
    axis=0,
)
ha_map = (fullcube_nosky[906:916, ...] - cont_map_ha).sum(axis=0)

fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(ha_map, origin="lower", cmap="gray_r")
ax.set_title("Hα 6563");

fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(cont_map_ha, vmin=0, vmax=1e6, origin="lower", cmap="gray_r")
ax.set_title("Hα 6563 continuum");

fig, ax = plt.subplots(figsize=(12,10))
mask = ha_map < 0.01 * np.median(ha_map)
smooth = 4
ratio = (
    median_filter(cont_map_ha, size=smooth) 
    / median_filter(ha_map, size=smooth)
)
ratio[mask] = np.nan
ax.imshow(
    ratio, 
    vmin=0, 
    vmax=2.5e-4, 
    origin="lower", 
    cmap="gray",
)
ax.set_title("Near continuum / Hα 6563");

fig, ax = plt.subplots(figsize=(12,10))
mask = ha_map < 1 * np.median(ha_map)
ratio = cii_map / ha_map
ratio[mask] = np.nan
ax.imshow(
    ratio, 
    vmin=3.2e-5, 
    vmax=1.6e-4, 
    origin="lower", 
    cmap="gray",
)
ax.set_title("C II 6462 / Hα 6563");

fig, ax = plt.subplots(
    figsize=(12,9),
    subplot_kw=dict(projection=w.celestial),
)
mask = ha_map < 0.2 * np.median(ha_map)
smooth = 8
ratio = (
    median_filter(cii_map, size=smooth) 
    / median_filter(ha_map, size=smooth)
)
ratio[mask] = np.nan
im = ax.imshow(
    ratio, 
    vmin=3.2e-5, 
    vmax=1.6e-4, 
    origin="lower", 
    cmap="magma",
)
fig.colorbar(im, ax=ax)
ax.set_title(f"C II 6462 / Hα 6563   median filtered ({smooth} pixels)");

# So the typical value is 1e-4.  We can estimate the C++/H+ abundance by looking at the effective recombination rates.

# ## Now try and extract 6578

# Find the shift in wav pixels between the two [N II] lines.

NWIN = 160
window_slice = slice(k0 - (NWIN//2), k0 + (NWIN//2) + 1)
window_cube = hdu.data[window_slice, ...]
window_wavs = wavs[window_slice]
sky = np.median(
    window_cube[:, yskyslice, xskyslice], 
    axis=(1, 2),
    keepdims=True,
)
window_norm = (window_cube - sky) / np.median(window_cube, axis=0)
window_median = np.median(window_norm, axis=(1, 2))

wav0_6548 = 6548.05 * u.Angstrom
wav0_6583 = 6583.45 * u.Angstrom
k0_6548 = w.spectral.world_to_pixel(wav0_6548)
k0_6583 = w.spectral.world_to_pixel(wav0_6583)
k0_6548, k0_6583

kshift = k0_6583 - k0_6548
kshift

kwindow = wav2k(window_wavs)
kfull = np.arange(nwavs)

p = T.fit(kwindow[:40], window_median[:40], deg=2)
p2 = T.fit(kwindow[-40:], window_median[-40:], deg=2)

window_shift = np.interp(
    kwindow, 
    kwindow + kshift,
    3 * (window_median - 0.85 * p(kwindow)),
)

from scipy.ndimage import convolve
delta = 0.105
kernel = np.array([delta, 1.0 - 2 * delta, delta])
window_median_smooth = convolve(window_median, kernel)

# + tags=[]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(kwindow, window_median, marker=".")
ax.plot(kwindow, 0.9 * p(kwindow), color="k")
ax.plot(kwindow, 0.9 * p2(kwindow), color="k")
ax.plot(kwindow, window_shift, marker=".")
ax.plot(
    kwindow, 
    window_median_smooth - window_shift - 0.9 * p2(kwindow), 
    marker="o", color="r", lw=2)

ax.axhline(0.0, color="k", linestyle="dashed")
ax.set(
    xlim=[k0 - 40, k0 + 40],
    ylim=[-1, 10],
)
ax.grid(axis="x")
# -

wav0_6578 = 6578.05 * u.Angstrom
k0_6578 = w.spectral.world_to_pixel(wav0_6578)
k0_6578


