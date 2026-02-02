import healpy as hp
import numpy as np
from . import npipe_utils as utils

nside = 2048
nnz = 3  # IQU
freq = 100 #GHz
path = "/home/vmura/npipe/data/maps/"
detsA = utils.get_dets_split(freq, 'A')
detsB = utils.get_dets_split(freq, 'B')

def make_map(signal, pixels, pixweights, name, nside=2048, nnz=3):
    isamp = 0
    imap = 0  # IQU
    bmap = np.zeros((12*nside**2,nnz))  # binned map
    nmap = np.zeros((12*nside**2,nnz))  # hits map
    nsamp = signal.shape[0]
    for isamp in range(nsamp):
        pix = pixels[isamp]
        if pix < 0:
            continue
        for imap in range(nnz):
            bmap[pix, imap] += signal[isamp] * pixweights.T[isamp, imap]
            nmap[pix, imap] += 1
    hp.write_map(path + name, bmap.T, nest=True, overwrite=True)
    hp.write_map(path + "hits_" + name, nmap.T, nest=True, overwrite=True)


for suffix, dets in (("A", detsA), ("B", detsB)):
    mapname = f"tau-A_{freq}hpbinning_{suffix}.fits"
    sig, pixx, weights = utils.get_tod(dets=dets)
    make_map(sig, pixx, weights, mapname)
    print(f"Written map {mapname}")