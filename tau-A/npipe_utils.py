import numpy as np
import healpy as hp
import os
from pathlib import Path
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits, ascii

# Functions for unit conversions
def arcmin_to_deg(x):
    return x / 60.0


def deg_to_arcmin(x):
    return x * 60.0


def arcsec_to_deg(x):
    return x / 3600.0


def deg_to_arcsec(x):
    return x * 3600.0


def arcsec_to_arcmin(x):
    return x / 60.0


def arcmin_to_arcsec(x):
    return x * 60.0


# Functions for file and data handling
def get_path(name, path):
    """A function to get path to a file with a name
    located in any of the subfolders in the path specified by user.

    Parameters
    ----------
    name: str
            Name of the file
    path: str
            Bash variable assigning the path to directory containing file.
    Returns
    ---------
    Path to file with filename <name>.
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def get_data_path(subfolder: str = None):
    env_data_path = os.getenv("NPIPE_DATA_DIR")
    if env_data_path:
        base_path = Path(env_data_path).expanduser().resolve()
    else:
        base_path = Path(__file__).resolve().parent.parent
    if subfolder:
        return base_path / subfolder
    return base_path


# Functions for coordinate transformations
def polangle_transform(rot, coords, psi, psi_deg=True):
    """
    Transform position angle of polarisation between different coordinate systems.
    Parameters
    ------------
    rot: str.
    Specify coordinate system transformation. Eg: ['G', 'C'] if transforming from Galactic to Equatorial coord systems.
    coords: array
    Coordinates (or direction) of reference in lonlat format in degrees
    psi: Float
    Polarisation angle in degrees

    Returns
    ------------
    Position angle of polarisation in desired coordinate system in degrees
    """
    r = hp.rotator.Rotator(coord=rot, deg=True)
    dpsi = hp.rotator.Rotator.angle_ref(r, coords, lonlat=True)
    if psi_deg:
        dpsi = np.degrees(dpsi)
    return dpsi + psi


def galactic_to_equatorial(theta, phi):
    """
    Convert galactic coordinates to equatorial coordinates.
    Parameters
    ------------
    rot: str.
    Specify coordinate system transformation. Eg: ['G', 'C'] if transforming from Galactic to Equatiorial coord systems.
    coords: array
    Coordinates (or direction) of reference in lonlat format in radians


    Returns
    ------------
    RA and DEC in Equatorial coordinates in degrees
    """
    r = hp.rotator.Rotator(coord=["G", "C"])
    dec, ra = r(theta, phi)
    return np.degrees(ra), 90 - np.degrees(dec)


def gal_to_icrs_coord(lon, lat):
    """
    Convert galactic coordinates to ICRS coordinates.
    Parameters
    ------------
    lon: array
        Galactic longitude in radians
    lat: array
        Galactic latitude in radians

    Returns
    ------------
    ra: array
        Right Ascension in radians
    dec: array
        Declination in radians
    """
    lon_c = []  # ra
    lat_c = []  # dec
    for i in np.arange(len(lon)):
        sc = SkyCoord(l=lon[i], b=lat[i], unit="rad", frame="galactic")
        lon_c.append(sc.transform_to("icrs").ra.rad)
        lat_c.append(sc.transform_to("icrs").dec.rad)
    return np.array(lon_c), np.array(lat_c)


def calc_centpix(side):
    """
    Calculate centre pixel for wcs generation.
    Parameters
    ------------
    side: int
        Number of pixels along one dimension.
    Returns
    ------------
    Centre pixel value.
    """
    return 0.5 * (side + 1)


def gen_wcs(side, pixel_size, l, b):
    w = WCS(naxis=2)
    centpix = calc_centpix(side)
    w.wcs.crpix = [centpix, centpix]
    w.wcs.cdelt = [-pixel_size, pixel_size]
    c = SkyCoord(l, b, frame="galactic", unit="degree")
    w.wcs.crval = [c.l.deg, c.b.deg]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.ctype = ["l---TAN", "b---TAN"]
    w.array_shape = side, side
    return w


def pixelize_skycoords(ra, dec, wcs, side, origin=0, offset=0.5):
    coords = np.vstack((ra, dec)).transpose()
    pix = wcs.wcs_world2pix(coords, origin)
    x = pix[:, 1] + offset
    y = pix[:, 0] + offset
    nx, ny = side, side
    xbinning = np.linspace(origin, nx + origin, nx + 1)
    ybinning = np.linspace(origin, ny + origin, ny + 1)
    return x, y, xbinning, ybinning


# Functions for TOD extraction from Planck NPIPE small datasets
def get_tod(dets, nside):
    pixels = []
    iweights = []
    qweights = []
    uweights = []
    signal = []
    thetas = []
    phis = []

    for det in dets:
        with fits.open(
            "/home/vmura/npipe/data/M1/small_dataset_M1_{}.fits".format(det)
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


def get_tod_withcc(dets, nside, alpha, bg_subtraction=False):
    pixels = []
    iweights = []
    qweights = []
    uweights = []
    signal = []
    thetas = []
    phis = []
    fr = np.array((100, 143, 217, 353))
    f_bg = np.array(
        [9.50687754e-5, 1.89320788e-4, 6.29073416e-4, 6.16040430e-3]
    )  # np.array([0.00013969, 0.00026815, 0.00088054, 0.00864909]) # in K_CMB units

    for det in dets:
        with fits.open(
            "/home/vmura/npipe/data/M1/small_dataset_M1_{}.fits".format(det)
        ) as hdul:
            if alpha == -0.32:
                data_uccc = ascii.read("/home/vmura/npipe/data/PR2-3/UC_CC_RIMO-4.txt")
            if alpha == -0.35:
                data_uccc = ascii.read(
                    "/home/vmura/npipe/data/PR2-3/UC_CC_RIMO-4_alpha-0.35.txt"
                )
            if alpha == -0.28:
                data_uccc = ascii.read(
                    "/home/vmura/npipe/data/PR2-3/UC_CC_RIMO-4_alpha-0.28.txt"
                )
            if alpha == "cc_komatsu":
                data_uccc = ascii.read(
                    "/home/vmura/npipe/data/PR2-3/cc_komatsu_HFI.txt"
                )
            uc_cc = np.array((data_uccc["UCxCC"]))
            detnames = np.array((data_uccc["Detector-name"]))
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
                sig -= f_bg[f_ind]
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


def get_tod_withcc_lfi(dets, nside, alpha, bg_subtraction=False):
    pixels = []
    iweights = []
    qweights = []
    uweights = []
    signal = []
    thetas = []
    phis = []
    f_bg = np.array([2.03856413e-04])  # in K_CMB units 1.70065975e-03, 1.54363888e-03,

    for det in dets:
        with fits.open(
            "/home/vmura/npipe/data/M1/small_dataset_M1_{}.fits".format(det)
        ) as hdul:
            # if alpha==-0.32:
            #     data_uccc = ascii.read('/home/vmura/npipe/data/PR2-3/UC_CC_RIMO-4.txt')
            # if alpha==-0.35:
            #     data_uccc = ascii.read('/home/vmura/npipe/data/PR2-3/UC_CC_RIMO-4_alpha-0.35.txt')
            if alpha == -0.28:
                data_uccc = ascii.read(
                    "/home/vmura/npipe/data/PR2-3/LFI_UC_CC_RIMO-4_alpha-0.28.txt"
                )
            uc_cc = np.array((data_uccc["UCxCC"]))
            detnames = np.array((data_uccc["Detector-name"]))
            # det_split = detnames.split('_')[1]
            # f_ind = np.where(det_split==fr)
            print(detnames)
            print(det)
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
                sig -= f_bg
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


def get_tod_ongrid(dets, npix, pixsize, coord="galactic"):
    iweights = []
    qweights = []
    uweights = []
    signal = []
    thetas = []
    phis = []
    for det in dets:
        with fits.open(
            "/home/vmura/npipe/data/M1/small_dataset_M1_{}.fits".format(det)
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
    if coord == "galactic":
        w2 = gen_wcs(
            side=npix, pixel_size=pix_size, l=184.5574 * u.deg, b=-5.7843 * u.deg
        )
        x, y, xbinning, ybinning = pixelize_skycoords(phi_e, theta_e, w2, side=npix)
    if coord == "equatorial":
        ra, dec = gal_to_icrs_coord(np.deg2rad(phi_e), np.deg2rad(theta_e))
        w2 = gen_wcs(
            side=npix, pixel_size=pix_size, l=83.63304 * u.deg, b=22.01449 * u.deg
        )
        x, y, xbinning, ybinning = pixelize_skycoords(ra, dec, w2, side=npix)
    return x, y, xbinning, ybinning, signal, pixweights


def get_dets(f):
    if f == 30:
        dets = ["LFI27M", "LFI27S", "LFI28M", "LFI28S"]
    if f == 44:
        dets = ["LFI24M", "LFI24S", "LFI25M", "LFI25S", "LFI26M", "LFI26S"]
    if f == 70:
        dets = [
            "LFI18S",
            "LFI19M",
            "LFI19S",
            "LFI20M",
            "LFI20S",
            "LFI21M",
            "LFI21S",
            "LFI22M",
            "LFI22S",
            "LFI23M",
            "LFI23S",
        ]
    if f == 100:
        dets = [
            "100-1a",
            "100-2a",
            "100-3a",
            "100-4a",
            "100-1b",
            "100-2b",
            "100-3b",
            "100-4b",
        ]
    if f == 143:
        dets = [
            "143-1a",
            "143-2a",
            "143-3a",
            "143-4a",
            "143-1b",
            "143-2b",
            "143-3b",
            "143-4b",
            "143-5",
            "143-6",
            "143-7",
        ]
    if f == 217:
        dets = [
            "217-1",
            "217-2",
            "217-3",
            "217-4",
            "217-5a",
            "217-5b",
            "217-6a",
            "217-6b",
            "217-7a",
            "217-7b",
            "217-8a",
            "217-8b",
        ]
    if f == 353:
        dets = [
            "353-1",
            "353-2",
            "353-3a",
            "353-3b",
            "353-4a",
            "353-4b",
            "353-5a",
            "353-5b",
            "353-6a",
            "353-6b",
            "353-7",
            "353-8",
        ]
    if f == 545:
        dets = ["545-1", "545-2", "545-4"]
    if f == 857:
        dets = ["857-1", "857-2", "857-3", "857-4"]
    return dets


def get_dets_split(f, split):
    if f == 70:
        if split == "A":
            dets = ["LFI18S", "LFI20M", "LFI20S", "LFI23M", "LFI23S"]  # "LFI18M",
        if split == "B":
            dets = ["LFI19M", "LFI19S", "LFI21M", "LFI21S", "LFI22M", "LFI22S"]
    elif f == 100:
        if split == "A":
            dets = ["100-1a", "100-1b", "100-4a", "100-4b"]
        if split == "B":
            dets = ["100-2a", "100-2b", "100-3a", "100-3b"]
    elif f == 143:
        if split == "A":
            dets = ["143-1a", "143-1b", "143-3a", "143-3b", "143-5", "143-7"]
        if split == "B":
            dets = ["143-2a", "143-2b", "143-4a", "143-4b", "143-6"]
    elif f == 217:
        if split == "A":
            dets = ["217-1", "217-3", "217-5a", "217-5b", "217-7a", "217-7b"]
        if split == "B":
            dets = ["217-2", "217-4", "217-6a", "217-6b", "217-8a", "217-8b"]
    elif f == 353:
        if split == "A":
            dets = ["353-1", "353-3a", "353-3b", "353-5a", "353-5b", "353-7"]
        if split == "B":
            dets = ["353-2", "353-4a", "353-4b", "353-6a", "353-6b", "353-8"]
    return dets


def create_fits(file_name, data):
    """A function to A function to write data into a fits file.

    Parameters
    ----------
    file_name: str
        Name of file for data to be written to.
    data
        Data to be written to.
    """
    hdu = fits.PrimaryHDU()
    hdu.data = np.array(data, dtype=np.float32)
    hdu.writeto(file_name, overwrite=True)
    return None


def healpy_getmap(path, side=None, coord=None, field=None, nest=False):
    """A function to read and extract Healpix maps. If side is None, Map is mollweide projected.
    If side and coords are specified, map is gnomonic projected with coord as centre of map.

    Parameters
    ----------
    path: str
        Path to Healpix map
    side: int
        number of pixels along one dimension.
        Map of dimension side x side will be extracted
    coord: list
        Image extracted around centre with coordinates in the form [lon, lat]
    field: int
        Field to extract from the Healpix maps. Indices to use: I=0, Q=1, U=2, IQU=3
    nest: bool
        True if Nested ordering. Default: False

    """
    if coord is None:
        if field <= 2:
            imap = hp.read_map(path, field=field, verbose=False)
            imap[imap == hp.UNSEEN] = 0
            map1 = hp.mollview(imap, nest, return_projected_map=True)
            return map1
        if field > 2:
            maps = []
            for i in np.arange(3):
                imap = hp.read_map(path, field=i, verbose=False)
                imap[imap == hp.UNSEEN] = 0
                map1 = hp.mollview(imap, nest, return_projected_map=True)
                maps.append(map1)
            return np.array(maps)
    else:
        if field <= 2:
            imap = hp.read_map(path, field=field, verbose=False)
            imap[imap == hp.UNSEEN] = 0
            map1 = hp.gnomview(
                imap,
                nest,
                rot=coord,
                xsize=side,
                return_projected_map=True,
                flip="astro",
                no_plot=True,
            )
            return map1
        if field > 2:
            maps = []
            for i in np.arange(3):
                imap = hp.read_map(path, field=i, verbose=False, dtype=np.float32)
                imap[imap == hp.UNSEEN] = 0
                map1 = hp.gnomview(
                    imap,
                    nest,
                    rot=coord,
                    xsize=side,
                    return_projected_map=True,
                    flip="astro",
                    no_plot=True,
                )
                maps.append(map1)
            return np.array(maps)
