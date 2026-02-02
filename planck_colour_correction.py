import numpy as np
from astropy.io import fits
from astropy.io import ascii
import scipy.integrate as integrate
from astropy.constants import k_B, c, h
from astropy.table import Table
import npipe_utils as utils

c = c.si.value  # speed of light in m/s
h = h.si.value  # Planck constant in J.s
k_B = k_B.si.value  # Boltzmann constant in J/K
T_CMB = 2.7255  # CMB temp. in K

class PlanckColourCorrection:
    """
    A class to compute colour correction and unit conversion coefficients for Planck detectors
    based on the bandpass transmission data from RIMO files.
    Attributes:
        instrument (str): 'LFI' or 'HFI' indicating the Planck instrument.
        alpha (float): Spectral index for colour correction.
        planck_freq (dict): Nominal frequencies for LFI and HFI instruments.
        freq_nominal (list): List of nominal frequencies for the selected instrument.
    Methods:
        bandpass_weight(freq, freq_nominal, transmission, alpha): Computes bandpass weight.
        bandpass_iras_to_alpha(freq, freq_nominal, transmission, alpha): Computes colour correction.
        Kcmb_to_Mjysr(freq, freq_nominal, transmission): Computes unit conversion factor.
        compute_uc_cc(instrument='HFI', detind=None, alpha=-0.28): Computes UC and CC for all detectors and saves to file.
    """
    def __init__(self, instrument="HFI", alpha=-0.28):
        self.instrument = instrument
        self.alpha = alpha
        self.planck_freq = {"LFI": [30, 44, 70], "HFI": [100, 143, 217, 353, 545, 857]}
        self.freq_nominal = self.planck_freq[instrument]

    def bandpass_weight(self, freq, freq_nominal, transmission, alpha):
        h_x = freq ** (alpha)
        weight = integrate.simpson(h_x * transmission, freq) / (
            integrate.simpson(transmission * (freq_nominal / freq), freq)
        )
        return weight

    def bandpass_iras_to_alpha(self, freq, freq_nominal, transmission, alpha):
        weight = (
            integrate.simpson(transmission * (freq_nominal / freq), freq)
        ) / integrate.simpson(transmission * (freq / freq_nominal) ** alpha, freq)
        return weight

    def Kcmb_to_Mjysr(self, freq, freq_nominal, transmission):
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
    
    def compute_uc_cc(self, instrument="HFI", detind=None, alpha=-0.28):
        detnames = []
        # Get bandpass transmission and frequency info from Planck RIMO files.
        path = utils.get_data_path(subfolder="PR2-3")
        hdu = fits.open(path + instrument + "_RIMO_R4.00.fits")
        if instrument=="HFI":
            detind = np.arange(3, 47)
        else:
            detind = np.arange(3, 24)
        cc = np.zeros(detind.shape[0])
        uc = np.zeros(detind.shape[0])

        # index of the power-law defining Crab nebula's SED
        # -0.28 is obtained from fitting a power-law to the flux
        # densities estimated via  aperture photometry on maps from
        # Planck PR3.
        for i in np.arange(detind.shape[0]):
            detname = hdu[detind[i]].name.split("_")[1]
            det = float(detname.split("-")[0])
            detnames.append(detname)
            f = hdu[detind[i]].data["WAVENUMBER"]
            if instrument=="HFI":
                f *= 1e-7 * c  # converting wave number to GHz
            trans = hdu[detind[i]].data["TRANSMISSION"]
            cc[i] = self.bandpass_iras_to_alpha(f[1:] * 1e9, det * 1e9, trans[1:], alpha)
            uc[i] = self.Kcmb_to_Mjysr(f[1:] * 1e9, det * 1e9, trans[1:])

        hdu.close()
        tab = Table()
        tab["Detector-name"] = detnames
        tab["Unit-Conversion"] = np.round(uc, 4)
        tab["Colour-Correction"] = np.round(cc, 5)
        tab["UCxCC"] = np.round(uc * cc, 5)
        ascii.write(
            tab, path + instrument + "_UC_CC_RIMO-4_alpha-{}.txt".format(alpha), overwrite=True
        )
        return detnames, uc, cc
