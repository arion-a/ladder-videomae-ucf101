"""Checks for ladder_final.decompose. Run: python -I test_decompose.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ladder_final as lf

D, KS = 24, 3                                   # frame dim; slow block = 1 real mode (0.95) + 1 rotating pair (0.9, angle 0.5)


def world(seed, nvid=150, T=60, slow_scale=0.6):
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.standard_normal((D, D)))[0] * 1.0
    th, rho = 0.5, 0.9
    Rm = rho * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Xs, Ss = [], []
    for _ in range(nvid):
        z = np.zeros((T, D)); z[0] = rng.standard_normal(D)
        for t in range(1, T):
            z[t, 0] = 0.95 * z[t - 1, 0] + np.sqrt(1 - 0.95 ** 2) * rng.standard_normal()
            z[t, 1:3] = Rm @ z[t - 1, 1:3] + np.sqrt(1 - rho ** 2) * rng.standard_normal(2)
            z[t, 3:] = 0.1 * z[t - 1, 3:] + np.sqrt(1 - 0.01) * rng.standard_normal(D - 3)
        z[:, :3] *= slow_scale                  # slow block has the SMALLER variance, so PCA picks fast directions
        Xs.append(z @ A.T); Ss.append(z[:, :3] @ A[:, :3].T)
    return Xs, Ss


def main():
    ok = True
    def chk(name, cond, info=""):
        nonlocal ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)

    Xs, Ss = world(0)
    fit, held = slice(0, 100), slice(100, 150)
    r = lf.run_ladder(Xs[fit], [1], 0.5)
    M, mu = r["maps"][0], r["mu"]
    chk("ladder finds the planted 3-dim block", M.shape[0] == 3, f"dim {M.shape[0]}")
    out, info = lf.decompose(Xs[held], M, mu, 1)
    lam = info["lam"]
    print("   eigenvalues", np.round(lam, 3), " cond(V)", round(info["cond"], 2))

    # 1. algebra
    e1 = max(np.abs(o["C"] @ info["V"].T - (x - mu) @ M.T).max() for o, x in zip(out, Xs[held]))
    P = np.linalg.pinv(M) @ M
    e2 = max(np.abs(o["rebuilt"] - (x - mu) @ P).max() for o, x in zip(out, Xs[held]))
    e3 = max(np.abs((o["rebuilt"] ** 2).sum(1) + (o["remainder"] ** 2).sum(1) - ((x - mu) ** 2).sum(1)).max() for o, x in zip(out, Xs[held]))
    chk("sum c_i v_i = s", e1 < 1e-9, f"max err {e1:.1e}")
    chk("rebuild = orthogonal projection pinv(M)M", e2 < 1e-9, f"max err {e2:.1e}")
    chk("rebuilt^2 + remainder^2 = frame^2 (Pythagoras)", e3 < 1e-8, f"max err {e3:.1e}")

    # 2. held-out persistence of each amount: lambda_hat = sum conj(c_t) c_{t+1} / sum |c_t|^2
    num = np.zeros(len(lam), complex); den = np.zeros(len(lam))
    for o in out:
        Cc = o["C"]; num += (Cc[:-1].conj() * Cc[1:]).sum(0); den += (np.abs(Cc[:-1]) ** 2).sum(0)
    lh = num / den
    err = np.abs(lh - lam).max()
    chk("held-out one-step ratio of c_i equals lambda_i", err < 0.06, f"max |lam_hat - lam| {err:.3f}")
    ph = np.mean([np.angle(o["C"][1:, k] / o["C"][:-1, k]).mean() for o in out for k in [int(np.argmax(np.abs(lam.imag)))]])
    chk("phase advances by angle(lambda) per step", abs(abs(ph) - 0.5) < 0.1, f"{ph:.3f} vs +-0.5")

    # 3. planted parts: rebuilt block vs the true slow part, against PCA of the same size
    Xa = np.vstack(Xs[fit]); w, U = np.linalg.eigh(np.cov((Xa - Xa.mean(0)).T)); Pp = U[:, np.argsort(-w)[:3]]
    num_l = num_p = den_ = 0.0
    for o, x, s in zip(out, Xs[held], Ss[held]):
        s = s - s.mean(0) * 0                       # true slow part is already zero-mean
        num_l += ((o["rebuilt"] - s) ** 2).sum()
        num_p += ((((x - x.mean(0)) @ Pp) @ Pp.T - s) ** 2).sum(); den_ += (s ** 2).sum()
    el, ep = num_l / den_, num_p / den_
    chk("rebuild recovers the planted slow part better than PCA-3", el < 0.5 * ep, f"rel err ladder {el:.3f}  PCA {ep:.3f}")

    # 4. per-frame share is not constant (whitening makes the AVERAGE trivial, the per-frame value is the readout)
    sh = np.concatenate([o["share"] for o in out])
    chk("share varies across frames", sh.std() > 0.02, f"mean {sh.mean():.3f} std {sh.std():.3f}")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
