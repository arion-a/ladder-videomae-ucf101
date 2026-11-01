"""Synthetic rotating-block test with known angular velocity, shaped like the pilot features (T=8 vectors per video, lags 1 and 2).
Frame = A [rotating plane (omega, rho, random phase/amplitude per video), static per-video appearance vector, fast noise] + small observation noise.
Run: python -I rotating_test.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ladder_final as lf

D, T = 128, 8
for V in (7500, 500):
    for omega in (0.2, 0.6, 1.2):
        rng = np.random.default_rng(1); rho = 0.95
        A = np.linalg.qr(rng.standard_normal((D, D)))[0]; R = rho * np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]])
        Xs = []
        for _ in range(V):
            z = np.zeros((T, D)); p = rng.standard_normal(2) * 1.5
            for t in range(T):
                z[t, :2] = p; p = R @ p + 0.1 * rng.standard_normal(2)
            z[:, 2:10] = 2.0 * rng.standard_normal(8)                       # static appearance (per video), large variance
            for t in range(T): z[t, 10:] = (0.3 * (z[t - 1, 10:] if t else 0) + rng.standard_normal(D - 10))   # fast noise
            Xs.append(z @ A.T)
        r = lf.run_ladder(Xs, [1, 2], 0.5)
        dims = [int(x) for x in r["dims"]]
        lam_all = np.linalg.eigvals(lf.lag_moment(lf.whiten(Xs)[0], 1))
        cplx = lam_all[(lam_all.imag > 0.05)]; top = cplx[np.argmax(np.abs(cplx))] if len(cplx) else 0j
        s = f"V={V:5d} omega={omega:.1f}  rung dims {dims}  top complex eigenvalue of lag-1 matrix: modulus {abs(top):.3f} angle {np.angle(top):.3f}"
        if dims and dims[0] > 0:
            M = r["maps"][0]; out, info = lf.decompose(Xs, M, r["mu"], 1); lam = info["lam"]; c = lam[lam.imag > 0.05]
            if len(c):
                k = int(np.argmax(np.abs(c))); j = int(np.where(lam == c[k])[0][0])
                ph = np.mean([np.angle(o["C"][1:, j] / o["C"][:-1, j]).mean() for o in out])
                s += f" | level-1 F: largest complex modulus {abs(c[k]):.3f} angle {np.angle(c[k]):.3f}; mean measured phase step of c {ph:.3f}"
        print(s, f"(planted: modulus ~{rho}, angle {omega})", flush=True)
