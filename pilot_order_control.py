"""Do the 8 time vectors track frame CONTENT or time SLOT? Compare original clips with clips whose 16 frames were reversed / shuffled.
Run: python -I pilot_order_control.py ORIG_FEAT REV_FEAT SHUF_FEAT   (dirs of <id>.npy (8,d))"""
import sys, glob, os, numpy as np
ids = [os.path.basename(f) for f in sorted(glob.glob(sys.argv[1] + "/*.npy"))]
L = lambda d: np.stack([np.load(f"{d}/{i}").astype(np.float64) for i in ids])
O, Rv, S = L(sys.argv[1]), L(sys.argv[2]), L(sys.argv[3])
cen = lambda A: A - A.mean(1, keepdims=True)
cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
mO = O.mean(1); between = np.mean([np.linalg.norm(mO[i] - mO[j]) for i in range(len(O)) for j in range(i)])
print(f"{len(ids)} videos. mean pairwise distance between video-mean vectors (original): {between:.3f}")
for nm, A in (("reversed", Rv), ("shuffled", S)):
    d = np.mean([np.linalg.norm(A[i].mean(0) - mO[i]) for i in range(len(O))])
    print(f"{nm}: distance of clip-mean vector from the original clip-mean, same video: {d:.3f} (= {d/between:.2f} x between-video distance)")
dO, dR = cen(O), cen(Rv)
same = cos(dO, dR); flip = cos(dO, dR[:, ::-1])
print(f"reversed clip, within-video deviations: cosine with original in the SAME slot {same:+.3f}; with the slot-reversed original (content-tracking prediction) {flip:+.3f}")
dd = np.diff(O, axis=1); ddr = np.diff(Rv, axis=1)
print(f"step vectors x_(t+1)-x_t: cosine(original step t, reversed clip step 6-t, sign flipped) {cos(dd, -ddr[:, ::-1]):+.3f} ; cosine(original step t, reversed clip step t) {cos(dd, ddr):+.3f}")
print(f"per-video temporal std of original (mean over dims) {O.std(1).mean():.4f}; reversed {Rv.std(1).mean():.4f}; shuffled {S.std(1).mean():.4f}")
