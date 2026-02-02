import numpy as np
from astropy.io import fits
from astropy.io import ascii
import scipy.integrate as integrate
from astropy.constants import k_B, c, h
from astropy.table import Table
import npipe_utils as utils

c = c.si.value
h = h.si.value
k_B = k_B.si.value
T_CMB = 2.7255  # CMB temp. in K
instrument = "HFI"

def bandpass_weight(freq, freq_nominal, transmission, alpha):
    h_x = freq ** (alpha)
    weight = integrate.simpson(h_x * transmission, freq) / (
        integrate.simpson(transmission * (freq_nominal / freq), freq)
    )
    return weight


def bandpass_iras_to_alpha(freq, freq_nominal, transmission, alpha):
    weight = (
        integrate.simpson(transmission * (freq_nominal / freq), freq)
    ) / integrate.simpson(transmission * (freq / freq_nominal) ** alpha, freq)
    return weight


def Kcmb_to_Mjysr(freq, freq_nominal, transmission):
    bn = []
    for i in np.arange(len(freq)):
        x = h * freq[i] / k_B / T_CMB
        b = (
            2
            * k_B**3.0
            * T_CMB**2.0
            / (h * c) ** 2.0
            * x**4.0
            * np.exp(x)
            / (np.exp(x) - 1) ** 2.0
            * 1e20
        )
        bn.append(b)
    weight = integrate.simpson(np.array(bn) * transmission, freq) / (
        integrate.simpson(transmission * (freq_nominal / freq), freq)
    )
    return weight

# Get bandpass transmission and frequency info from Planck RIMO files.
path = utils.get_data_path(subfolder="PR2-3")
hdu = fits.open(path + instrument + "_RIMO_R4.00.fits")
if instrument=="HFI":
    detind = np.arange(3, 47)
    freq_nominal = np.array((100.0, 143.0, 217.0, 353.0))
else:
    detind = np.arange(3, 24)
    freq_nominal = np.array((30, 44, 70))
detnames = []
est_weights = np.zeros(detind.shape[0])
cc = np.zeros(detind.shape[0])
uc = np.zeros(detind.shape[0])

# index of the power-law defining Crab nebula's SED
# -0.28 is obtained from fitting a power-law to the flux
# densities estimated via  aperture photometry on maps from
# Planck PR3.
alpha = -0.28 # -0.35

for i in np.arange(detind.shape[0]):
    detname = hdu[detind[i]].name.split("_")[1]
    det = float(detname.split("-")[0])
    ind = np.where(det == freq_nominal)
    detnames.append(detname)
    f = hdu[detind[i]].data["WAVENUMBER"]
    if instrument=="HFI":
        f *= 1e-7 * c  # converting wave number to GHz
    trans = hdu[detind[i]].data["TRANSMISSION"]
    trans_err = hdu[detind[i]].data["UNCERTAINTY"]
    cc[i] = bandpass_iras_to_alpha(f[1:] * 1e9, det * 1e9, trans[1:], alpha)
    uc[i] = Kcmb_to_Mjysr(f[1:] * 1e9, det * 1e9, trans[1:])

hdu.close()
tab = Table()
tab["Detector-name"] = detnames
tab["Unit-Conversion"] = np.round(uc, 4)
tab["Colour-Correction"] = np.round(cc, 5)
tab["UCxCC"] = np.round(uc * cc, 5)
ascii.write(
    tab, path + instrument + "_UC_CC_RIMO-4_alpha-{}.txt".format(alpha), overwrite=True
)
