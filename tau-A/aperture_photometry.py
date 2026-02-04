"""Aperture photometry functions for estimating flux densities from 2D maps."""
import numpy as np

def jysr_area(numpix, pix_size):
    """
    Calculate the aperture in Jy/sr given number of pixels and pixel size
    to estimate flux density within aperture in appropriate units.
    Parameters
    ------------
    numpix: int
        Number of pixels.
    pix_size: float
        Pixel size in arcmin.
    Returns
    -----------
        Area in Jy/sr.
    """
    return numpix * pix_size**2 / 3600 / 180**2 * np.pi**2 * 1e6


def int_area(data, eff_beam, num_pix, extent=None, pixel_size=None):
    """
    Return map with values outside the circular aperture set to zero.
    
    Parameters
    ------------
    data: array
        2D array of map data.
    eff_beam: float
        Effective beam FWHM in arcmin.
    num_pix: int
        Number of pixels along one side of the map.
    extent: float
        Extent of the circular aperture in arcmin.
    pixel_size: float, optional
        Pixel size in arcmin. Default: 1.5 arcmin.
    Returns
    -----------
        2D array with values outside the circular area set to zero.
    """
    if pixel_size is None:
        pixel_size = 1.5  # arcmin
    center = np.array([num_pix // 2, num_pix // 2])
    x, y = np.indices((num_pix, num_pix))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r * pixel_size / 60
    theta_tau = 7.0
    R = extent * np.sqrt(eff_beam**2 + theta_tau**2) / 60 / 2.0  # converting to degrees
    maparea = np.copy(data)
    for i in np.arange(num_pix):
        for j in np.arange(num_pix):
            if r[i, j] > R:
                maparea[i, j] = 0.0
    return maparea


def int_annuli(data, eff_beam, num_pix, inner=None, outer=None, pixel_size=None):
    """
    Return map with values outside the annular area set to zero.
    Parameters
    ------------
    data: array
        2D array of map data.
    eff_beam: float
        Effective beam FWHM in arcmin.
    num_pix: int
        Number of pixels along one side of the map.
    inner: float
        Inner extent of the annulus from the center in arcmin.
    outer: float
        Outer extent of the annulus from the center in arcmin.
    pixel_size: float, optional
        Pixel size in arcmin. Default: 1.5 arcmin.
    Returns
    -----------
        2D array with values outside the annular area set to zero.
    """
    if pixel_size is None:
        pixel_size = 1.5  # arcmin
    center = np.array([num_pix // 2, num_pix // 2])
    x, y = np.indices((num_pix, num_pix))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r * pixel_size / 60
    theta_tau = 7.0
    ri = inner * np.sqrt(eff_beam**2 + theta_tau**2) / 60 / 2.0
    R = outer * np.sqrt(eff_beam**2 + theta_tau**2) / 60 / 2.0  # converting to degrees
    maparea = np.copy(data)
    for i in np.arange(num_pix):
        for j in np.arange(num_pix):
            if r[i, j] > R:
                maparea[i, j] = 0.0
            elif r[i, j] < ri:
                maparea[i, j] = 0.0
    return maparea


# Extract different areas for each and estimate the sum of pixels.
def sum_pixels(
    data, eff_beam, outer_extent, inttype="circle", inner_extent=None, pix_size=None
):
    """
    Sum the pixel values within a circular or annular aperture.
    Parameters
    ------------
    data: array
        2D array of map data.
    eff_beam: float
        Effective beam FWHM in arcmin.
    outer_extent: float
        Outer extent of the aperture in arcmin.
    inttype: str, optional
        Type of aperture to compute flux density from: 
        'circle' or 'annulus'. Default: 'circle'.
    inner_extent: float, optional
        Inner extent of the annulus in arcmin. Required if inttype is 'annulus'.
    pix_size: float, optional
        Pixel size in arcmin. Default: 1.5 arcmin.
    Returns
    -----------
        sum_area: float
            Sum of pixel values within the specified aperture.
        pix_num: int
            Number of pixels within the aperture.
    """
    if pix_size is None:
        pix_size = 1.5  # arcmin
    num_pix = data.shape[0]
    if inttype == "circle":
        area = int_area(
            data, eff_beam, num_pix, extent=outer_extent, pixel_size=pix_size
        )
    elif inttype == "annulus":
        area = int_annuli(
            data,
            eff_beam,
            num_pix,
            inner=inner_extent,
            outer=outer_extent,
            pixel_size=pix_size,
        )
    pix_num = 0
    a = area.reshape(data.shape[0] ** 2)
    for j in np.arange(data.shape[0] ** 2):
        if a[j] != 0:
            pix_num += 1
    sum_area = np.sum(area)
    return sum_area, pix_num


def radialprofile(maps, numpix):
    """
    Compute radial profile of the input map (radial profile of whatever is
    contained in the 2D map/array).
    Parameters
    ------------
    maps: array
        2D array of map data.
    numpix: int
        Number of pixels along one side of the map.
    Returns
    -----------
        krad: array
            Radial distances in pixels.
        radprofile: array
            Radial profile values.
    """
    num_pix = numpix  # int(map_size*60/pixel_size)
    center = np.array([num_pix // 2, num_pix // 2])
    x, y = np.indices((num_pix, num_pix))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r_int = r.astype(int)
    weight = np.bincount(r_int.reshape(-1))
    k = np.copy(r)
    krad = np.bincount(r_int.reshape(-1), k.reshape(-1)) / weight
    radprofile = np.bincount(r_int.reshape(-1), maps.reshape(-1)) / weight
    return krad, radprofile


def pa_aperture(maps, fwhm, ap_extent=1.5, bg_subtraction=True):
    """
    Compute polarisation angle from IQU maps using Aperture Photometry. Stokes Q & U in COSMO convention.

    Parameters
    ------------
    maps: array
        Array of dimension (nnz, npix, npix) where npix==Number of pixels per side.
    fwhm: float
        Beam FWHM in arcmin.
    ap_extent: float, optional
        Extent of the circular aperture over which to compute. Aperture = ap_extent*(np.sqrt(theta_tau**2+fwhm**2))/2.
    bg_subtraction: bool, optional
        Condition whether background subtraction should be applied. Default: True

    Returns
    -----------
    Polarisation angle in IAU convention in degrees
    """
    sumi = np.zeros((3))
    fa = 1.4
    flux_bg = 0.0
    for j in np.arange(3):
        sum_circle_i, numc = sum_pixels(
            maps[j],
            eff_beam=fwhm,
            outer_extent=ap_extent,
            inttype="circle",
            inner_extent=None,
            pix_size=1.5,
        )
        if bg_subtraction:
            f_bg = int_annuli(
                maps[j],
                eff_beam=fwhm,
                num_pix=maps[j].shape[0],
                inner=1.5 * ap_extent,
                outer=2 * ap_extent,
                pixel_size=1.5,
            )
            flux_bg = np.median(f_bg[f_bg != 0]) * jysr_area(numc, 1.5) * fa
        sumi[j] = sum_circle_i * jysr_area(1, 1.5) * fa - flux_bg
    return np.degrees(0.5 * np.arctan2(-sumi[2], sumi[1]))
