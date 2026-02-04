"""
Mapmaking script to make Stokes IQU binned maps from Planck TOD with
pixel size of 1.5' and 80 pixels along one side.
"""

import numpy as np
from . import npipe_utils as utils
from scipy.linalg import pinv

nnz = 3  # IQU
freq = [100, 143, 217, 353] #GHz 30, 44, 70, 
npix = 80
pixel_size = 1.5 #arcmin
splits = ['A', 'B']
path = "/home/vmura/npipe/data/maps/"

for s in splits:
    for f in freq:
        # get detector names that are part of frequency channel
        dets = utils.get_dets_split(f, s)

        # Get TOD & coordinates on a grid
        print("Extracting TOD for {}GHz freq channel".format(f))
        x, y, xbinning, ybinning, signal, pixweights = utils.get_tod_ongrid(dets, npix, pixel_size)
        print("Making map for {}GHz freq channel".format(f))

        bmap=np.zeros((npix, npix,3))
        for i in np.arange(npix):
            m = np.logical_and(x>xbinning[i],x<xbinning[i+1])
            yx = y*m
            for j in np.arange(npix):
                m = np.logical_and(np.abs(yx)>ybinning[j],yx<ybinning[j+1])
                PTP = pixweights[:,m]@pixweights[:,m].T
                mp = pinv(PTP)@pixweights[:,m]@signal[m]
                bmap[j,i] = mp

        print("Made map for {}GHz freq channel. Writing map to file..".format(f))
        utils.create_fits(path+"{}GHz_{}pix_{}_grid.fits".format(f,npix,s), bmap.T)