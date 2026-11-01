"""Pilot: within-video temporal variance vs between-video variance of the 8 time vectors. Run: python -I pilot_variance.py FEAT_DIR"""
import sys, glob, numpy as np
Xs = [np.load(f) .astype(np.float64) for f in sorted(glob.glob(sys.argv[1] + "/*.npy"))]
A = np.stack(Xs)                                    # (V, 8, d)
V, T, d = A.shape
tot = A.reshape(-1, d).var(0)                       # total variance per dim, all frames pooled
within = A.var(1).mean(0)                           # mean over videos of the temporal variance, per dim
between = A.mean(1).var(0)                          # variance across videos of the video means
print(f"V={V} videos, T={T}, d={d}")
print(f"mean over dims: within / total = {np.mean(within / tot):.4f}   between / total = {np.mean(between / tot):.4f}  (within + between*(V-1)/V should be ~ total)")
print(f"summed variance: within {within.sum():.4f}  between {between.sum():.4f}  total {tot.sum():.4f}  within/between {within.sum()/between.sum():.4f}")
# unbiased within with T-1 dof (8 points) so it is comparable
w2 = A.var(1, ddof=1).mean(0); print(f"within (ddof=1) / total summed: {w2.sum()/tot.sum():.4f}")
# temporal structure of the 8 vectors: is the within-video variation a smooth trend or noise? lag-1 autocorrelation of deviations
dev = A - A.mean(1, keepdims=True)
num = (dev[:, 1:] * dev[:, :-1]).sum(); den = (dev ** 2).sum(); print(f"lag-1 autocorrelation of within-video deviations (pooled): {num/den:.3f}")
# fraction of within-video variance in the top singular directions of the deviations
S = np.linalg.svd(dev.reshape(-1, d), compute_uv=False) ** 2; print("within-video deviation variance share in top 1,3,7 directions:", np.round(np.cumsum(S)[[0, 2, 6]] / S.sum(), 3))
