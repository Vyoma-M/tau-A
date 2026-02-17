""" Module for loading Planck Time-Ordered Data (TOD) for tau-A analysis.
Provides the `Get_TOD` class which can be configured with various parameters.
One can then call methods like `tod()` or `tod_withcc()` to load the TOD data as a 
`TOD` object containing signal, pixel indices (or x-y coordinates for TOD projected
onto a grid), and weights. The MapMaker class can then use this TOD object to perform mapmaking.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict

import npipe_utils as utils
import numpy as np
from astropy.io import fits, ascii
import astropy.units as u
import healpy as hp
import logging


@dataclass
class GetTODConfig:
    """
    Configuration class for `Get_TOD` loader.
    Parameters:
    ------------
    instrument: str
        Planck instrument to load data for. Must be one of "HFI" or "LFI". Default is "HFI".
    data_path: Optional[str]
        Path to folder containing destriped Planck TOD files. If not provided, will use default path from `utils.get_data_path()`.
    alpha: float
        Spectral index for colour correction. Default is -0.28, appropriate for tau-A.
    freq: int
        Frequency channel to load data for. Default is 100 GHz.
    coord: Optional[Sequence[float]]
        Coordinates of the target field. If not provided, will default to tau-A coordinates based on `coord_system`.
    coord_system: str
        Coordinate system for `coord`. Must be one of "galactic" or "equatorial". Default is "galactic".
    withcc: bool
        Whether to apply colour correction to the TOD. Default is False.
    f_bg: Optional[np.ndarray]
        Background flux values for background subtraction, in K_CMB units. If `bg_subtraction` is True and this is not provided, default values will be used based on the instrument and frequencies.
    bg_subtraction: bool
        Whether to perform background subtraction using `f_bg`. Default is False.
    nside: Optional[int]
        Nside parameter for HEALPix pixelization.
    planck_freq: Dict[str, Sequence[int]]
        Dictionary mapping instruments to their available frequency channels. Default is {"LFI": [30, 40, 70], "HFI": [100, 143, 217, 353, 545, 857]}.
    
    Returns
    -----------
    GetTODConfig
        An instance of the `GetTODConfig` dataclass with the specified configuration.
    """
    instrument: str = "HFI"
    data_path: Optional[str] = None
    alpha: float = -0.28
    freq: int = 100
    coord: Optional[Sequence[float]] = None
    coord_system: str = "galactic"
    withcc: bool = False
    f_bg: Optional[np.ndarray] = None
    bg_subtraction: bool = False
    nside: Optional[int] = None
    planck_freq: Dict[str, Sequence[int]] = field(
        default_factory=lambda: {
            "LFI": [30, 40, 70],
            "HFI": [100, 143, 217, 353, 545, 857],
        }
    )

    def __post_init__(self):
        if self.coord is None:
            if self.coord_system == "galactic":
                self.coord = [184.5574, -5.7843]
            else:
                self.coord = [83.63304, 22.01449]
        if self.bg_subtraction and self.f_bg is None:
            self.f_bg = self._default_bg()

    def _default_bg(self) -> np.ndarray:
        if self.instrument == "LFI":
            return np.array([1.70065975e-03, 1.54363888e-03, 2.03856413e-04])
        return np.array([9.50687754e-5, 1.89320788e-4, 6.29073416e-4, 6.16040430e-3])


@dataclass
class TOD:
    signal: np.ndarray
    pixels: np.ndarray
    weights: np.ndarray
    theta: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    nside: Optional[int] = None


class Get_TOD:
    def __init__(self, config: Optional[GetTODConfig] = None, **kwargs):
        if config is None:
            config = GetTODConfig(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.data_path = config.data_path or utils.get_data_path(subfolder="data/")
        self.alpha = config.alpha
        self.instrument = config.instrument
        self.coord = config.coord
        self.coord_system = config.coord_system
        self.withcc = config.withcc
        self.f_bg = config.f_bg
        self.bg_subtraction = config.bg_subtraction
        self.freq = config.freq
        self.planck_freq = config.planck_freq

        logging.getLogger(__name__).info("Data path set to: %s", self.data_path)

        # Validate data path
        self._validate_data_path()

    def _validate_data_path(self):
        if os.path.isdir(self.data_path):
            print(
                f"Using {self.data_path} as the folder containing \
destriped Planck TOD files."
            )
        elif not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"Data path {self.data_path} does not exist. Please \
provide a valid path to the folder containing destriped Planck TOD files."
            )
        elif os.path.isfile(self.data_path):
            raise IsADirectoryError(
                f"Expected a folder containing destriped Planck TOD files, but found a\
                    file path at: {self.data_path}\n"
                f"Please provide the full path to the data folder."
            )

    def _read_detector_arrays(self, det):
        """Read TOD for a single detector FITS file and return flattened arrays."""
        path = os.path.join(self.data_path, "M1", f"small_dataset_M1_{det}.fits")
        with fits.open(path) as hdul:
            data = hdul[1].data
            theta = data["theta"].ravel()
            phi = data["phi"].ravel()
            iweight = np.ones_like(data["qweight"].ravel())
            qweight = data["qweight"].ravel()
            uweight = data["uweight"].ravel()
            signal = data["signal"].ravel()
        return theta, phi, iweight, qweight, uweight, signal

    def _get_tod(self, freq, nside):
        pixels = []
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []
        dets = utils.get_dets(freq)
        for det in dets:
            theta, phi, iwt, qwt, uwt, sig = self._read_detector_arrays(det)
            thetas.append(theta)
            phis.append(phi)
            pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
            iweights.append(iwt)
            qweights.append(qwt)
            uweights.append(uwt)
            signal.append(sig)
        signal = np.hstack(signal)
        pixels = np.hstack(pixels)
        pixweights = np.vstack(
            [
                np.hstack(iweights),
                np.hstack(qweights),
                np.hstack(uweights),
            ]
        )
        return signal, pixels, pixweights

    def _get_tod_withcc(self, freq, nside, alpha, instrument, bg_subtraction):
        pixels = []
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []
        dets = utils.get_dets(freq)
        for det in dets:
            # read detector arrays
            theta, phi, iwt, qwt, uwt, sig = self._read_detector_arrays(det)
            # read colour-correction table (try instrument-prefixed filename, fallback to unprefixed)
            cc_fname = os.path.join(
                self.data_path, "PR2-3", f"{instrument}_UC_CC_RIMO-4_alpha{alpha}.txt"
            )
            if not os.path.exists(cc_fname):
                cc_fname = os.path.join(
                    self.data_path, "PR2-3", f"UC_CC_RIMO-4_alpha{alpha}.txt"
                )
            data_uccc = ascii.read(cc_fname)
            uc_cc = np.array((data_uccc["UCxCC"]))
            detnames = np.array((data_uccc["Detector-name"]))
            # find detector index in cc table
            ind = np.where(detnames == det)[0]
            if ind.size == 0:
                # if not found, skip this detector
                continue
            thetas.append(theta)
            phis.append(phi)
            pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
            iweights.append(iwt)
            qweights.append(qwt)
            uweights.append(uwt)
            # background subtraction
            if bg_subtraction and (self.f_bg is not None):
                if instrument == "HFI":
                    try:
                        f_ind = list(self.planck_freq[instrument]).index(freq)
                    except ValueError:
                        f_ind = 0
                    sig = sig - self.f_bg[f_ind]
                elif instrument == "LFI":
                    sig = sig - self.f_bg
            signal.append(sig * uc_cc[ind])
        signal = np.hstack(signal)
        pixels = np.hstack(pixels)
        pixweights = np.vstack(
            [
                np.hstack(iweights),
                np.hstack(qweights),
                np.hstack(uweights),
            ]
        )
        return signal, pixels, pixweights

    def _get_tod_ongrid(self, coord, freq, npix, pixsize, coord_system="galactic"):
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []
        dets = utils.get_dets(freq)
        for det in dets:
            theta, phi, iwt, qwt, uwt, sig = self._read_detector_arrays(det)
            thetas.append(theta)
            phis.append(phi)
            iweights.append(iwt)
            qweights.append(qwt)
            uweights.append(uwt)
            signal.append(sig)

        pixweights = np.vstack(
            [
                np.hstack(iweights),
                np.hstack(qweights),
                np.hstack(uweights),
            ]
        )
        signal = np.hstack(signal)
        theta = np.hstack(thetas)
        phi = np.hstack(phis)
        pix_size = pixsize / 60
        theta_e, phi_e = np.rad2deg(np.radians(90) - theta), np.rad2deg(phi)
        if coord_system == "galactic":
            w2 = utils.gen_wcs(
                side=npix, pixel_size=pix_size, l=coord[0] * u.deg, b=coord[1] * u.deg
            )
            x, y, xbinning, ybinning = utils.pixelize_skycoords(
                phi_e, theta_e, w2, side=npix
            )
        if coord_system == "equatorial":
            ra, dec = utils.gal_to_icrs_coord(np.deg2rad(phi_e), np.deg2rad(theta_e))
            w2 = utils.gen_wcs(
                side=npix, pixel_size=pix_size, l=coord[0] * u.deg, b=coord[1] * u.deg
            )
            x, y, xbinning, ybinning = utils.pixelize_skycoords(ra, dec, w2, side=npix)
        return x, y, xbinning, ybinning, signal, pixweights

    # Public wrappers
    def tod(self, freq: Optional[int] = None, nside: Optional[int] = None) -> TOD:
        """Load TOD for given frequency and nside and return a `TOD` object.

        Defaults to configuration values when arguments are not provided.
        """
        freq = freq or self.freq
        nside = nside or self.config.nside
        if nside is None:
            raise ValueError(
                "nside must be provided either in config or as an argument"
            )
        signal, pixels, pixweights = self._get_tod(freq, nside)
        return TOD(signal=signal, pixels=pixels, weights=pixweights, nside=nside)

    def tod_withcc(
        self,
        freq: Optional[int] = None,
        nside: Optional[int] = None,
        alpha: Optional[float] = None,
        instrument: Optional[str] = None,
        bg_subtraction: Optional[bool] = None,
    ) -> TOD:
        """Load TOD with colour correction applied and return a `TOD` object.

        Arguments override configuration when provided.
        """
        freq = freq or self.freq
        nside = nside or self.config.nside
        if nside is None:
            raise ValueError(
                "nside must be provided either in config or as an argument"
            )
        alpha = alpha if alpha is not None else self.alpha
        instrument = instrument or self.instrument
        bg_subtraction = (
            bg_subtraction if bg_subtraction is not None else self.bg_subtraction
        )
        signal, pixels, pixweights = self._get_tod_withcc(
            freq, nside, alpha, instrument, bg_subtraction
        )
        return TOD(signal=signal, pixels=pixels, weights=pixweights, nside=nside)

    def tod_ongrid(
        self,
        coord: Optional[Sequence[float]] = None,
        freq: Optional[int] = None,
        npix: int = 80,
        pixsize: float = 1.5,
        coord_system: Optional[str] = None,
    ):
        """Project TOD onto a square grid around `coord`.
        Only use this for small fields where flat projection
        is applicable.

        Returns: x, y, xbinning, ybinning, signal, pixweights
        """
        coord = coord or self.coord
        freq = freq or self.freq
        coord_system = coord_system or self.coord_system
        return self._get_tod_ongrid(
            coord, freq, npix, pixsize, coord_system=coord_system
        )
