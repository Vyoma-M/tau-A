import sys
import os
# make the tau-A sources importable (adjust if running from repo root)
sys.path.append(os.path.join(os.path.dirname(__file__).removesuffix("/examples"), "/tau-A"))
from ..tauA.gettod import Get_TOD, GetTODConfig

#FIX: #1) Configure (replace data_path with your local data folder)
cfg = GetTODConfig(
    instrument="HFI",
    freq=100,
    data_path=None,   # must contain M1/ and PR2-3/ subfolders
    withcc=True,
    bg_subtraction=True,
    nside=2048,                       # required for assemble_tod
)

# 2) Instantiate loader
loader = Get_TOD(config=cfg)

# 3) Load TOD (returns TOD dataclass)
tod = loader.tod()  # uses cfg.freq and cfg.nside
print("tod:", tod.signal.shape, tod.pixels.shape, tod.weights.shape)

# 4) Load colour-corrected TOD
tod_cc = loader.tod_withcc()
print("tod_withcc:", tod_cc.signal.shape, tod_cc.pixels.shape)

# 5) Project onto a small image grid around cfg.coord
x, y, xbin, ybin, grid_signal, grid_pixweights = loader.tod_ongrid(
    npix=80, pixsize=1.5
)
print("tod_ongrid:", grid_signal.shape)