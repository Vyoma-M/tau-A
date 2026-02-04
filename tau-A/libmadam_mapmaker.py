"""
This is a wrapper script to use libmadam functionalities
(https://github.com/hpc4cmb/libmadam) to make maps from
Planck NPIPE destriped Time-Ordered Data (TOD).
"""

import npipe_utils as utils
import healpy as hp
import numpy as np
from astropy.io import fits
from mpi4py import MPI
import libmadam_wrapper as madam

comm = MPI.COMM_WORLD
nside = 2048  # Healpix NSIDE
nnz = 3  # IQU
freq = 143  # Planck frequency band in GHz
file_root = "madam_maps_"  # Name of folder into which maps are saved

# Path to NPIPE TOD of the Crab nebula
path = utils.get_data_path(subfolder="data")
dets = utils.get_dets(freq)
ndet = len(dets)

# Define variables to import data from TOD FITS files
timestamps = []
pixels = []
iweights = []
qweights = []
uweights = []
signal = []
periods = []
offset = 0

# Get TOD from FITS files
for det in dets:
    with fits.open(path + "small_dataset_M1_{}.fits".format(det)) as hdul:
        data = hdul[1].data
        cols = hdul[1].columns
        times = data["time"].ravel()
        timestamps.append(times)
        nsample = times.size
        theta = data["theta"].ravel()
        phi = data["phi"].ravel()
        pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
        iweights.append(np.ones_like(data["qweight"].ravel()))
        qweights.append(data["qweight"].ravel())
        uweights.append(data["uweight"].ravel())
        signal.append(data["signal"].ravel())
        # Separate breaks in time stamps larger than an hour into a separate period
        # for destriping
        dtime = np.diff(times)
        ind = np.argwhere(dtime > 3600).ravel()
        for i in ind:
            periods.append(offset + i + 1)
        offset += nsample

timestamps = np.hstack(timestamps).astype(madam.TIMESTAMP_TYPE)
pixels = np.hstack(pixels).astype(madam.PIXEL_TYPE)
pixweights = (
    np.vstack(
        [
            np.hstack(iweights),
            np.hstack(qweights),
            np.hstack(uweights),
        ]
    )
    .astype(madam.WEIGHT_TYPE)
    .flatten("F")
)
signal = np.hstack(signal).astype(madam.SIGNAL_TYPE)
periods = np.array(periods)

nsample = timestamps.size
fsample = 1 / np.median(np.diff(timestamps))

# Treat the entire dataset as if coming from a single detector

dets = [str(freq)]
detweights = np.ones(ndet)

# Defining power spectrum density of 1/f noise
# for the purposes of using the destriper.
# But destriping is overridden to perform only binning.
npsd = np.ones(ndet, dtype=np.int64)
npsdtot = np.sum(npsd)
psdstarts = np.zeros(npsdtot)
npsdbin = 10
psdfreqs = np.arange(npsdbin) * fsample / npsdbin
npsdval = npsdbin * npsdtot
psdvals = np.zeros(npsdval)

# NPIPE data is destriped, so the madam instance will only bin the
# TOD at the defined NSIDE.
pars = {}
pars["kfirst"] = False  # Remember that NPIPE data is already destriped
pars["base_first"] = 1000000  # Full periods
pars["kfilter"] = False
pars["fsample"] = fsample
pars["nside_map"] = nside
pars["nside_cross"] = nside // 2
pars["nside_submap"] = 16
pars["write_map"] = True
pars["write_binmap"] = True
pars["write_matrix"] = False
pars["write_wcov"] = False
pars["write_mask"] = False
pars["write_hits"] = True
pars["write_leakmatrix"] = False
pars["file_root"] = file_root + str(freq)
pars["path_output"] = "./"
pars["allreduce"] = True
pars["info"] = 3

# MADAM destriper
madam.destripe(
    comm,
    pars,
    dets,
    detweights,
    timestamps,
    pixels,
    pixweights,
    signal,
    periods,
    npsd,
    psdstarts,
    psdfreqs,
    psdvals,
)
