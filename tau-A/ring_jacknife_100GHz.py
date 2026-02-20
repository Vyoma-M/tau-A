import numpy as np
from astropy.io import fits
import healpy as hp
import npipe_utils as utils
import scipy
import os
from astropy.io import ascii
from astropy.table import Table

nside = 2048
nnz = 3
f = 100  # GHz
path = utils.get_data_path(subfolder="data/")
split = "A"
pixels = []
iweights = []
qweights = []
uweights = []
signal = []
rings = []

# Nominal frequencies of Planck bands
freq = np.array([30, 44, 70, 100, 143, 217, 353])  # GHz
fwhm_arcmin = np.array([32.3, 27.1, 13.3, 9.66, 7.27, 5.01, 4.86])

# Unit conversion obtained from UC-CC tables published in Planck 2015 catalogue
conversion = np.array(
    (23.5099, 55.7349, 129.1869, 244.0960, 371.7327, 483.6874, 287.4517)
)
fa = 1.4  # Aperture correction factor
ind = np.where(freq == f)[0][0]
pa = np.zeros((100))
sumi = np.zeros((pa.shape[0], nnz))

dets = utils.get_dets_split(f, split)
for det in dets:
    with fits.open(
        os.path.join(path, "small_dataset_M1_{}.fits".format(det))
    ) as hdul:
        data = hdul[1].data
        cols = hdul[1].columns
        theta = data["theta"].ravel()
        phi = data["phi"].ravel()
        pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
        iweights.append(np.ones_like(data["qweight"].ravel()))
        qweights.append(data["qweight"].ravel())
        uweights.append(data["uweight"].ravel())
        signal.append(data["signal"].ravel())
        rings.append(data["ring"].ravel())

pixweights = np.vstack(
    [
        np.hstack(iweights),
        np.hstack(qweights),
        np.hstack(uweights),
    ]
)
signal = np.hstack(signal)
pixels = np.hstack(pixels)
rings = np.hstack(rings)
pix_ids = np.array(sorted(list(set(pixels))))
ring_ids = np.array(sorted(list(set(rings))))
# Select rings for jackknife sampling
pix_ids_2 = sorted(list(set(pixels[np.logical_and(rings > 1343, rings < 1370)])))
pix_ids_2.extend(sorted(list(set(pixels[np.logical_and(rings > 6621, rings < 6651)]))))
pix_ids_2.extend(
    sorted(list(set(pixels[np.logical_and(rings > 12323, rings < 12350)])))
)
pix_ids_2.extend(
    sorted(list(set(pixels[np.logical_and(rings > 17612, rings < 17646)])))
)
pix_ids_2.extend(
    sorted(list(set(pixels[np.logical_and(rings > 23115, rings < 23141)])))
)
pix_ids_2.extend(
    sorted(list(set(pixels[np.logical_and(rings > 23409, rings < 23439)])))
)


for i in np.arange(len(pa)):
    n = np.random.randint(0, len(pix_ids_2))
    pix_ids_3 = pix_ids_2.copy()
    element = pix_ids_3[n]
    pix_ids_3.remove(element)
    pix_ids_3 = np.array(pix_ids_3)
    bmap = np.zeros((12 * nside**2, nnz))
    # Make binned maps from TOD from the selected rings
    for pix in pix_ids_3:
        PTP = pixweights[:, pixels == pix] @ pixweights[:, pixels == pix].T
        mp = (
            scipy.linalg.pinv(PTP)
            @ pixweights[:, pixels == pix]
            @ signal[pixels == pix]
        )
        bmap[pix] = mp
    for j in np.arange(nnz):
        map1 = hp.gnomview(
            bmap[:, j] * conversion[ind],
            rot=[184.55746, -05.78436],
            xsize=80,
            nest=True,
            flip="astro",
            return_projected_map=True,
            no_plot=True,
        )
        # Estimate flux density of the Crab nebula through aperture photometry
        # over an aperture of 1.5'.
        sum_circle_i, numc = utils.sum_pixels(
            map1,
            eff_beam=fwhm_arcmin[ind],
            outer_extent=1.5,
            inttype="circle",
            inner_extent=None,
            pix_size=1.5,
        )
        # Estimate background flux by computing flux density in an annulus 1 deg. away 
        # from centre of the tau-A field.
        f_bg = utils.int_annuli(
            map1,
            eff_beam=fwhm_arcmin[ind],
            num_pix=map1.shape[0],
            inner=1.5**2,
            outer=3.0,
            pixel_size=1.5,
        )
        # Calculate background flux density scaled to source aperture
        flux_bg = np.median(f_bg[f_bg != 0]) * utils.jysr_area(numc, 1.5) * fa
        sumi[i, j] = sum_circle_i * utils.jysr_area(1, 1.5) * fa - flux_bg

pa = np.degrees(0.5 * np.arctan2(-sumi[:, 2], sumi[:, 1]))
tab = Table()
tab["pa"] = pa
tab["pd"] = np.sqrt(sumi[:, 1] ** 2 + sumi[:, 2] ** 2)
tab["i"] = sumi[:, 0]
tab["q"] = sumi[:, 1]
tab["u"] = sumi[:, 2]
ascii.write(tab, path + "100GHz_ring_" + split + ".txt", overwrite=True)
