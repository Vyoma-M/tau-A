import os
import npipe_utils as utils
import numpy as np
from astropy.io import fits
import astropy.units as u
import healpy as hp


class Get_TOD:
    def __init__(
        self,
        instrument="HFI",
        data_path=None,
        alpha=-0.28,
        coord=None,
        coord_system="galactic",
        withcc=False,
        f_bg=None,
        bg_subtraction=False,
    ):
        self.data_path = data_path
        self.alpha = alpha
        self.instrument = instrument
        self.coord = coord
        self.coord_system = coord_system
        self.withcc = withcc
        self.f_bg = f_bg
        self.bg_subtraction = bg_subtraction
        self.planck_freq = {"LFI": [30, 40, 70], "HFI": [100, 143, 217, 353, 545, 857]}
        if self.data_path is None:
            self.data_path = utils.get_data_path(subfolder="data/")
        if coord is None and coord_system == "galactic":
            coord = [184.5574, -5.7843]  # galactic coord of tau-A
        elif coord is None and coord_system == "equatorial":
            coord = [83.63304, 22.01449]  # equatorial coord of tau-A
        if self.bg_subtraction and self.f_bg is None:
            if instrument == "LFI":
                f_bg = np.array(
                    [1.70065975e-03, 1.54363888e-03, 2.03856413e-04]
                )  # in K_CMB units
            elif instrument == "HFI":
                f_bg = np.array(
                    [9.50687754e-5, 1.89320788e-4, 6.29073416e-4, 6.16040430e-3]
                )  # in K_CMB units
            print(
                "Background flux for background subtraction not provided. Using default values of"
                "{} in K_CMB units for frequencies {}".format(
                    f_bg, self.planck_freq[instrument]
                )
            )
        if withcc:
            print(
                "Colour correction for SED with index {} will be applied".format(
                    self.alpha
                )
            )

        # Validate data path
        self._validate_data_path()

    def _validate_data_path(self):
        if os.path.isdir(self.data_path):
            print(
                f"Using {self.data_path} as the folder containing\
                 destriped Planck TOD files."
            )
        elif os.path.isfile(self.data_path):
            raise IsADirectoryError(
                f"Expected a folder containing destriped Planck TOD files, but found a\
                    file path at: {self.data_path}\n"
                f"Please provide the full path to the data folder."
            )

    def _get_tod(self, dets, nside):
        pixels = []
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []

        for det in dets:
            with fits.open(
                self.data_path + "M1/small_dataset_M1_{}.fits".format(det)
            ) as hdul:
                data = hdul[1].data
                theta = data["theta"].ravel()
                thetas.append(theta)
                phi = data["phi"].ravel()
                phis.append(phi)
                pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
                iweights.append(np.ones_like(data["qweight"].ravel()))
                qweights.append(data["qweight"].ravel())
                uweights.append(data["uweight"].ravel())
                signal.append(data["signal"].ravel())
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

    def _get_tod_withcc(self, dets, nside, alpha, instrument, bg_subtraction):
        pixels = []
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []
        fr = self.planck_freq[self.instrument]
        for det in dets:
            with fits.open(
                self.data_path + "M1/small_dataset_M1_{}.fits".format(det)
            ) as hdul:
                if alpha == -0.32:
                    data_uccc = ascii.read(self.data_path + "/PR2-3/UC_CC_RIMO-4.txt")
                elif alpha == -0.35:
                    data_uccc = ascii.read(
                        self.data_path + "/PR2-3/UC_CC_RIMO-4_alpha-0.35.txt"
                    )
                elif alpha == -0.28:
                    data_uccc = ascii.read(
                        self.data_path + "/PR2-3/UC_CC_RIMO-4_alpha-0.28.txt"
                    )
                uc_cc = np.array((data_uccc["UCxCC"]))
                detnames = np.array((data_uccc["Detector-name"]))
                if instrument == "HFI":
                    det_split = det.split("-")[0]
                    f_ind = np.where(int(det_split) == fr)
                ind = np.where(detnames == det)[0]
                data = hdul[1].data
                theta = data["theta"].ravel()
                thetas.append(theta)
                phi = data["phi"].ravel()
                phis.append(phi)
                pixels.append(hp.ang2pix(nside, theta, phi, nest=True, lonlat=False))
                iweights.append(np.ones_like(data["qweight"].ravel()))
                qweights.append(data["qweight"].ravel())
                uweights.append(data["uweight"].ravel())
                sig = np.array(data["signal"].ravel())
                if bg_subtraction:
                    if instrument == "HFI":
                        sig -= self.f_bg[f_ind]
                    elif instrument == "LFI":
                        sig -= self.f_bg
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

    def _get_tod_ongrid(self, coord, dets, npix, pixsize, coord_system="galactic"):
        iweights = []
        qweights = []
        uweights = []
        signal = []
        thetas = []
        phis = []
        for det in dets:
            with fits.open(
                self.data_path + "M1/small_dataset_M1_{}.fits".format(det)
            ) as hdul:
                data = hdul[1].data
                theta = data["theta"].ravel()
                phi = data["phi"].ravel()
                thetas.append(theta)
                phis.append(phi)
                iweights.append(np.ones_like(data["qweight"].ravel()))
                qweights.append(data["qweight"].ravel())
                uweights.append(data["uweight"].ravel())
                signal.append(data["signal"].ravel())

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
