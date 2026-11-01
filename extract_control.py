"""Frame-order control for the pilot: same extraction as extract.py, but the 16 sampled frames are put in a different order
before the encoder. MODE = rev (reversed) or shuf (fixed random permutation, seed 0).
Usage: python extract_control.py MODE VIDEO_ROOT OUT_DIR"""
import sys, glob, os
import numpy as np, torch
import extract as ex

MODE = sys.argv[1] if len(sys.argv) > 1 else "rev"
order = np.arange(ex.NFR)[::-1] if MODE == "rev" else np.random.default_rng(0).permutation(ex.NFR)
orig = ex.Clips.__getitem__

def getitem(self, i):
    v, pv = orig(self, i)                       # pv (16,3,224,224): frames in time order
    return v, pv[torch.as_tensor(order.copy())]
ex.Clips.__getitem__ = getitem
if __name__ == "__main__":
    ex.main(sys.argv[2], sys.argv[3])
