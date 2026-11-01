"""Merged ladder of invariants: champion a005 plus audited contributions. numpy only, pure functions.

Layout
  PART 1  champion functions, copied verbatim from ladder_champion.py (a005), except `cut` (extended below,
          default behaviour identical) and `lift` (renamed lift_quadratic).
  PART 2  contributions, each a separate switchable function with its limits in the docstring.
  Defaults reproduce the champion exactly (every contribution is OFF unless asked for); see FINAL_SOLUTION.md
  for where each one helps and where it hurts.
"""
import numpy as np

# ===================================================================== PART 1: champion (verbatim)
def haar(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def inv_sqrt(C, tol=1e-10):
    """symmetric C^{-1/2}, pseudo-inverse on numerically null directions."""
    w, V = np.linalg.eigh(C)
    keep = w > tol * w.max()
    return (V[:, keep] * w[keep] ** -0.5) @ V[:, keep].T


def whiten(Xs):
    """list of (T_v,d) -> (list of whitened arrays, W symmetric, mu)."""
    A = np.vstack(Xs); mu = A.mean(0); Ac = A - mu
    W = inv_sqrt(Ac.T @ Ac / len(Ac))
    return [(X - mu) @ W for X in Xs], W, mu


def lag_moment(Rs, tau):
    """ordered pair moment B[i,j] = mean r_{t+tau,i} r_{t,j}, pairs inside one video."""
    num, N = 0.0, 0
    for R in Rs:
        if len(R) > tau:
            num = num + R[tau:].T @ R[:-tau]; N += len(R) - tau
    return num / N


def shuffled_moment(Rs, rng):
    """independent-pair control: same marginals, pairing destroyed, frame dependence also destroyed."""
    R = np.vstack(Rs)
    return R[rng.permutation(len(R))].T @ R / len(R)


def dyn_floor(Rs, rng, reps=3):
    """noise level of the dynamic part that respects autocorrelation: demean each video (removes per-video
    constants), pair every frame with a frame of the same video at least T/4 away (circular shift).
    Marginal autocorrelation is kept, pairing is destroyed. Returns max |eig| averaged over draws."""
    out = []
    for _ in range(reps):
        num, N = 0.0, 0
        for R in Rs:
            Rc = R - R.mean(0); T = len(Rc); s = int(rng.integers(T // 4, 3 * T // 4))
            num = num + np.roll(Rc, -s, axis=0).T @ Rc; N += T
        out.append(np.abs(np.linalg.eigvals(num / N)).max())
    return float(np.mean(out))


def _orth_rows(V):
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    return U



def cut(B, theta, sign_mode="modulus"):
    """CONTRIBUTION 2 (sign convention). rows Q (k x d, orthonormal) spanning the B-closed slow functionals.
    sign_mode="modulus" (champion, DEFAULT): keep left eigenvectors with |eigenvalue| >= theta, complex pairs whole.
        Meaning of 'the same': predictable and slow in absolute value, so an alternating mode (-0.95) counts as slow.
    sign_mode="real": keep REAL eigenvalues with real part >= theta; keep complex pairs by modulus.
        Meaning: slow means persistent with the same sign (a mode of -0.95 is a fast alternation, not an invariant).
    sign_mode="halfplane" (ADDED POST HOC after the pre-registered test of "real" failed, see FINAL_SOLUTION.md):
        like "real", but a complex pair is kept by modulus only if its angle is below pi/2 (real part > 0). Reason: a
        repeated negative eigenvalue (-0.95 twice) is split by sampling noise into a complex pair of modulus ~0.95
        and angle ~pi, which "real" keeps as a rotation. Limit: a true rotation by more than pi/2 per lag is dropped.
    Limits: which one is right is a definition of 'the same', not a statistical question (see FINAL_SOLUTION.md);
    in 'real' mode a negative real eigenvalue is dropped even if it is exactly closed. Odd lags flip sign of
    negative modes, even lags do not, so 'modulus' is lag-parity independent and 'real' is not."""
    lam, V = np.linalg.eig(B.T)
    if sign_mode == "modulus":
        keep = np.abs(lam) >= theta
    elif sign_mode == "real":
        isreal = np.abs(lam.imag) <= 1e-9 * np.maximum(1.0, np.abs(lam))
        keep = np.where(isreal, lam.real >= theta, np.abs(lam) >= theta)
    elif sign_mode == "halfplane":
        isreal = np.abs(lam.imag) <= 1e-9 * np.maximum(1.0, np.abs(lam))
        keep = np.where(isreal, lam.real >= theta, (np.abs(lam) >= theta) & (lam.real > 0))
    else:
        raise ValueError(sign_mode)
    k = int(keep.sum())
    if k == 0:
        return np.zeros((0, B.shape[0]))
    Vk = V[:, keep]
    return _orth_rows(np.hstack([Vk.real, Vk.imag]))[:, :k].T


def cut_real(B, theta):
    return cut(B, theta, "real")


def cut_sym(B, theta):
    """literal SplitJEPA difference loss: top eigenspace of (B+B^T)/2."""
    w, V = np.linalg.eigh((B + B.T) / 2)
    return V[:, w >= theta].T


def cut_k(B, k, sym=False):
    if sym:
        w, V = np.linalg.eigh((B + B.T) / 2); return V[:, np.argsort(-w)[:k]].T
    lam, V = np.linalg.eig(B.T)
    idx = np.argsort(-np.abs(lam))
    Vk = V[:, idx]
    return _orth_rows(np.hstack([Vk.real, Vk.imag]))[:, :k].T


def closure_defect(B, Q):
    """|| Q B (I - Q^T Q) ||_F^2 : how much of the block's own next-step prediction is borrowed from
    the part that was thrown away. 0 iff the block is closed."""
    return float(np.linalg.norm(Q @ B @ (np.eye(B.shape[0]) - Q.T @ Q)) ** 2)


def split(Rs, tau, theta, cutter=cut):
    return cutter(lag_moment(Rs, tau), theta)


def ladder(Xs, taus, theta, cutter=cut):
    """REFERENCE implementation on frames (keeps all frames in memory, one pass per level).
    Returns list of (k_j x d) maps acting on the centred embedding; nested by construction."""
    Rs, W, mu = whiten(Xs)
    M = W.copy(); maps = []
    for tau in taus:
        Q = split(Rs, tau, theta, cutter)
        if Q.shape[0] == 0:
            maps.append(np.zeros((0, M.shape[1]))); break
        if Q.shape[0] < Rs[0].shape[1]:
            Rs, W2, _ = whiten([R @ Q.T for R in Rs])
            M = W2 @ Q @ M
        maps.append(M.copy())
    return maps


# ---- the scalable form: a commutative monoid of sums, then d x d algebra only
def moments_of(X, taus):
    """sufficient statistics of one video (or shard): n, sum x, sum x x^T, and per lag
    (sum x_{t+tau} x_t^T, sum x_{t+tau}, sum x_t, pair count)."""
    m = dict(n=len(X), sx=X.sum(0), sxx=X.T @ X, lag={})
    for t in taus:
        if len(X) > t:
            m['lag'][t] = (X[t:].T @ X[:-t], X[t:].sum(0), X[:-t].sum(0), len(X) - t)
    return m


def combine(a, b):
    """monoid operation: add statistics (across videos, shards, devices)."""
    lag = {}
    for t in set(a['lag']) | set(b['lag']):
        p = [m['lag'][t] for m in (a, b) if t in m['lag']]
        lag[t] = tuple(sum(x) for x in zip(*p))
    return dict(n=a['n'] + b['n'], sx=a['sx'] + b['sx'], sxx=a['sxx'] + b['sxx'], lag=lag)


def build_moments(Xs, taus):
    m = None
    for X in Xs:
        mi = moments_of(X, taus); m = mi if m is None else combine(m, mi)
    return m


def finalize(m, taus):
    """centred covariance and centred ordered lag matrices of the raw embedding, from the sums."""
    n = m['n']; mu = m['sx'] / n; C = m['sxx'] / n - np.outer(mu, mu); B = {}
    for t in taus:
        S, sa, sb, N = m['lag'][t]
        B[t] = (S - np.outer(sa, mu) - np.outer(mu, sb) + N * np.outer(mu, mu)) / N
    return mu, C, B


def ladder_m(mom, taus, theta, cutter=cut):
    """The ladder from the sums only (no frames). Level j lives on cached d x d matrices:
    child lag matrix = M B_tau M^T, child covariance = (Q M) C (Q M)^T. Returns (maps, mu)."""
    mu, C, B = finalize(mom, taus)
    M = inv_sqrt(C); maps = []
    for tau in taus:
        Q = cutter(M @ B[tau] @ M.T, theta)
        if Q.shape[0] == 0:
            maps.append(np.zeros((0, M.shape[1]))); break
        QM = Q @ M
        if Q.shape[0] < M.shape[0]:
            M = inv_sqrt(QM @ C @ QM.T) @ QM
        maps.append(M.copy())
    return maps, mu


def levels(maps):
    dims = [m.shape[0] for m in maps]
    return dims, 1 + sum(1 for a, b in zip(dims[:-1], dims[1:]) if b < a) if dims else 0


def sin_theta(P, E):
    a = np.linalg.qr(P.T)[0]; b = np.linalg.qr(E.T)[0]
    c = np.linalg.svd(a.T @ b, compute_uv=False)
    k = min(a.shape[1], b.shape[1])
    return float(np.sqrt(max(0.0, k - (c[:k] ** 2).sum())))


def boot_sin(Xs, taus, theta, nb=20, seed=0):
    """DATA-ONLY error estimate for every level: resample videos with replacement, rerun the ladder from the
    sums, and report the mean sin Theta between the full-sample level and the resampled level (same size only).
    Replaces the uncomputable ||B_hat - B|| in the residual bound. Cost: nb extra runs of ladder_m."""
    rng = np.random.default_rng(seed)
    mom = [moments_of(X, taus) for X in Xs]

    def run(idx):
        m = None
        for i in idx:
            m = mom[i] if m is None else combine(m, mom[i])
        return ladder_m(m, taus, theta)[0]
    full = run(range(len(Xs))); out = [[] for _ in full]
    for _ in range(nb):
        mb = run(rng.integers(0, len(Xs), len(Xs)))
        for j, (a, b) in enumerate(zip(full, mb)):
            if a.shape[0] == b.shape[0] and a.shape[0] > 0:
                out[j].append(sin_theta(a, b))
    return full, [float(np.mean(o)) if o else np.nan for o in out]


def cluster_times(ts, gap):
    """group implied times by log-gaps larger than `gap` (no truth needed). Returns list of arrays."""
    ts = np.sort(np.asarray(ts)); cuts = np.where(np.diff(np.log(ts)) > gap)[0] + 1
    return np.split(ts, cuts)


def heldout_r2(P, Xtr, Ttr, Xte, Tte, mu):
    f = lambda Xs: np.vstack([(X - mu) @ P.T for X in Xs])
    Ftr, Fte = f(Xtr), f(Xte)
    Ftr1 = np.hstack([Ftr, np.ones((len(Ftr), 1))]); Fte1 = np.hstack([Fte, np.ones((len(Fte), 1))])
    T1, T2 = np.vstack(Ttr), np.vstack(Tte)
    beta = np.linalg.lstsq(Ftr1, T1, rcond=None)[0]
    res = ((T2 - Fte1 @ beta) ** 2).sum(); tot = ((T2 - T2.mean(0)) ** 2).sum()
    return float(1 - res / tot)


def implied_times(B, tau, lo, hi=0.9):
    """abstraction, Gaussian/Markov case: t_i = -tau / ln|lambda_i(B_tau)| for eigenvalues in the band
    (lo, hi). No index matching is needed (the set of times is a multiset). |lambda| >= 0.99 is flagged
    as shared content (t = infinity), |lambda| <= lo as below the noise floor (not reported)."""
    lam = np.abs(np.linalg.eigvals(B))
    band = lam[(lam > lo) & (lam < hi)]
    return -tau / np.log(band), int((lam >= 0.99).sum())


def implied_time(Rs, tau):
    """non-Markov dictionary version (two lags): top eigenvalue only, ratio removes the prefactor."""
    lam1 = np.abs(np.linalg.eigvals(lag_moment(Rs, tau))).max()
    lam2 = np.abs(np.linalg.eigvals(lag_moment(Rs, 2 * tau))).max()
    return -tau / np.log(lam2 / lam1)


def truth_idx(mods, tau, theta):
    """indices of latent coordinates whose persistence |mu|^tau is at least theta."""
    return np.where(mods >= theta ** (1.0 / tau) - 1e-12)[0]


def lift_quadratic(R):
    d = R.shape[1]; iu = np.triu_indices(d)
    Q = (R[:, :, None] * R[:, None, :])[:, iu[0], iu[1]]
    return np.hstack([R, Q])

def dyn_floor_x(Rs, rng, reps=3):
    """noise level of the dynamic part, cross-video null: demean each video, pair video v with a DIFFERENT video
    (same positions). Independent videos have exactly zero cross-correlation, so any eigenvalue here is sampling noise,
    and slow modes inside a video cannot inflate it (the T/4-shift null dyn_floor could)."""
    out = []
    for _ in range(reps):
        p = rng.permutation(len(Rs)); num = 0.0; N = 0
        for i, R in enumerate(Rs):
            S = Rs[p[i]] if p[i] != i else Rs[p[(i + 1) % len(Rs)]]
            T = min(len(R), len(S)); a = R[:T] - R[:T].mean(0); b = S[:T] - S[:T].mean(0)
            num = num + b.T @ a; N += T
        out.append(np.abs(np.linalg.eigvals(num / N)).max())
    return float(np.mean(out))



# ===================================================================== PART 2: contributions
from functools import partial


# ---- 1. irreversibility test and cutter choice ---------------------------------------------------
def irreversibility_pvalue(Rs, tau=1, nperm=499, rng=None):
    """CONTRIBUTION 1. Exact randomisation test of time-reversal symmetry (detailed balance) of the lag-tau pair
    moment. Reversing a whole video turns its pair sum F_v into F_v^T exactly, for any within-video dependence;
    under reversibility the reversal of each video is a symmetry, so videos can be flipped independently.
    Statistic ||F - F^T||_F, p = (1 + #{null >= observed}) / (1 + nperm). Rs: list of whitened (T_v,d) arrays.
    Limits: needs several independent videos (size is exact only for the video-flip group); tests the second
    order (pair) moment only, so a non-reversible process with a symmetric lag-tau pair moment passes; power at
    lag 1 can be low when the rotation per frame is small; needs per-video pair sums F_v (V x d x d), a small
    sufficient statistic but not the pooled sums of the monoid."""
    rng = np.random.default_rng(0) if rng is None else rng
    F = np.stack([R[tau:].T @ R[:-tau] for R in Rs if len(R) > tau])
    n = sum(len(R) - tau for R in Rs if len(R) > tau)
    FT = np.transpose(F, (0, 2, 1))
    stat = lambda S: np.linalg.norm(S - S.T)
    s0 = stat(F.sum(0) / n); cnt = 0
    for _ in range(nperm):
        f = rng.random(len(F)) < 0.5
        cnt += stat(np.where(f[:, None, None], FT, F).sum(0) / n) >= s0
    return (1 + cnt) / (1 + nperm)


def choose_cutter(Rs, tau=1, alpha=0.05, rng=None, sign_mode="modulus", nperm=499):
    """CONTRIBUTION 1. Symmetric cut (cut_sym) if the reversibility p-value >= alpha, ordered cut otherwise.
    Returns (cutter, p). Limits: the symmetric cut is exact only under detailed balance (T2); failing to reject
    is not proof of reversibility, so with few videos or weak asymmetry the gate falls to the symmetric cut on
    non-reversible data (error then up to 0.14-0.7 sinTheta in the audits). alpha is a conventional size, not tuned."""
    p = irreversibility_pvalue(Rs, tau, nperm, rng)
    return (cut_sym if p >= alpha else partial(cut, sign_mode=sign_mode)), p


# ---- 3. difference-moment / same-video content rung ----------------------------------------------
def within_video_moment(Rs):
    """CONTRIBUTION 3. Same-video difference moment over ALL pairs in a video:
    sum_v sum_{s,t} (r_s-r_t)(r_s-r_t)^T / (2 sum_v T_v^2) = pooled within-video covariance of whitened frames
    (exact identity)."""
    num = 0.0; N = 0
    for R in Rs:
        Rc = R - R.mean(0); num = num + Rc.T @ Rc; N += len(R)
    return num / N


def content_rung(Xs, tol=0.1):
    """CONTRIBUTION 3. Exact invariant block of the same-video family: directions of whitened frames with
    within-video variance <= tol (population value is exactly 0 for a per-video constant). Returns (map acting on
    the centred raw embedding, eigenvalues ascending). Needs no lag and has no end effects.
    Limits: a slow mode with relaxation time comparable to the video length has small within-video variance
    (about 1 - 2*t_relax/T) and is then counted as content; frame noise raises the content eigenvalue above 0 so
    tol must exceed the noise fraction; sees only the second moment."""
    Rs, W, mu = whiten(Xs)
    w, V = np.linalg.eigh(within_video_moment(Rs))
    return V[:, w <= tol].T @ W, w


# ---- 4. window-mean pre-processing and diagonal lift ----------------------------------------------
def window_mean(X, w):
    """CONTRIBUTION 4. Non-overlapping window mean of w frames (T -> T//w). Limits: lags are then in window units;
    removes within-window dynamics; helps only when the signal of interest is a slowly varying LEVEL of a feature
    (e.g. a variance after diag_lift) buried in fast noise; it can hide fast structure."""
    if w <= 1:
        return X
    n = len(X) // w
    return X[:n * w].reshape(n, w, -1).mean(1)


def diag_lift(R):
    """CONTRIBUTION 4. [r, r^2] (2d features, no cross terms). Scales as O(d). Limits: sees regimes that change
    the VARIANCE of coordinates; blind to regimes that rotate or couple coordinates (needs lift_quadratic);
    whitened raw coordinates are not guaranteed to be the coordinates in which the regime acts diagonally."""
    return np.hstack([R, R ** 2])


def make_features(Xs, window=1, diag=False):
    """CONTRIBUTION 4. Returns (list of feature videos, feature function). Centring/whitening for the lift uses Xs
    only; the function can be applied to held-out videos. window=1, diag=False is the identity."""
    if not diag and window <= 1:
        return list(Xs), (lambda X: X)
    A = np.vstack(Xs); mu = A.mean(0); W = inv_sqrt(np.cov(A.T, bias=True))

    def feat(X):
        F = (X - mu) @ W
        if diag:
            F = diag_lift(F)
        return window_mean(F, window)
    return [feat(X) for X in Xs], feat


# ---- 5. detection rule, debias, feasibility, data-derived floor -----------------------------------
def detectable(rho, d, m_eff):
    """CONTRIBUTION 5. Random-matrix detection rule: a mode with persistence rho (= eigenvalue of the lag moment)
    is recoverable only if rho > sqrt(d/m_eff), m_eff the effective number of independent pairs. Limits: derived
    for an isolated spike in an iid bulk; m_eff of consecutive frames is smaller than the pair count."""
    return rho > np.sqrt(d / m_eff)


def debias_persistence(rho_hat, c):
    """CONTRIBUTION 5. Invert the finite-sample shift of the top eigenvalue of the SYMMETRISED lag moment:
    with l = 1 - rho_hat, population 1-rho is the smaller root of u^2 - (1 + l - c) u + l = 0, c = d/m_eff.
    Limits: derived for the symmetric statistic and an iid bulk; for the ordered (non-symmetric) cut the eigenvalue
    is not shifted the same way (tested in test_final.py, H5a); meaningless when rho_hat is below the edge."""
    l = 1.0 - rho_hat; b = 1.0 + l - c
    return 1.0 - 0.5 * (b - np.sqrt(max(b * b - 4 * l, 0.0)))


def rung_feasible(theta, d_in, n_pairs, kappa=1.0, rule="bbp"):
    """CONTRIBUTION 5. Per-rung feasibility flag. rule 'bbp': theta > sqrt(d_in*kappa/n_pairs) (a010); rule 'a003':
    (1-theta) <= 2(1 - sqrt(d_in*kappa/n_pairs)). kappa >= 1 converts consecutive pairs into independent ones
    (kappa=1 is optimistic). False means: do not trust this rung (it can return wrong dims silently)."""
    c = d_in * kappa / n_pairs
    return bool(theta > np.sqrt(c)) if rule == "bbp" else bool((1 - theta) <= 2 * (1 - np.sqrt(c)))


def data_floor(Rs, rng=None, reps=3):
    """CONTRIBUTION 5. Data-derived floor on |eigenvalue| of the lag moment: max of the shuffled-pair null
    (independent pairs, a002) and dyn_floor (autocorrelation-aware, demeaned same-video shifted pairs).
    Returns dict. Limits: a null, not a threshold with a stated error rate; dyn_floor can be inflated by modes
    slower than T/4; heavy tails inflate both."""
    rng = np.random.default_rng(0) if rng is None else rng
    sh = float(np.mean([np.abs(np.linalg.eigvals(shuffled_moment(Rs, rng))).max() for _ in range(reps)]))
    dy = dyn_floor(Rs, rng, reps)
    return dict(shuffle=sh, dyn=dy, floor=max(sh, dy))


def theta_from_floor(theta, floor, factor=1.5):
    """CONTRIBUTION 5. Effective threshold: never below factor x data floor (factor 1.5 is the champion's own
    `lo = 1.5 * floor` in check_abstraction, not tuned here)."""
    return max(theta, factor * floor)


# ---- 6. consensus level reading --------------------------------------------------------------------
def plateau_levels(dims, d, min_run=2):
    """dimensions held on >= min_run consecutive lags (excluding 0 and d)."""
    out = []; i = 0
    while i < len(dims):
        j = i
        while j + 1 < len(dims) and dims[j + 1] == dims[i]:
            j += 1
        if j - i + 1 >= min_run and dims[i] not in (0, d):
            out.append(int(dims[i]))
        i = j + 1
    return out


def consensus_levels(mom, taus, thetas=(0.4, 0.45, 0.5, 0.55, 0.6), cutter=cut, min_run=2):
    """CONTRIBUTION 6. Level dimensions that form a plateau along a dense lag grid at EVERY threshold in a band
    (a009). Returns sorted list, largest first. taus must be a dense (e.g. geometric ratio 1.5) grid.
    Limits: reports only levels separated by a gap wide enough to hold >= min_run grid lags at all band thresholds,
    so it can miss real but crowded levels (lower recall, fewer wrong levels); cost = len(thetas) ladder runs."""
    d = mom['sx'].shape[0]
    lists = []
    for th in thetas:
        maps, _ = ladder_m(mom, taus, th, cutter)
        lists.append(set(plateau_levels([m.shape[0] for m in maps], d, min_run)))
    return sorted(set.intersection(*lists), reverse=True)


def geometric_lags(lo, hi, ratio=1.5):
    return tuple(sorted(set(int(round(lo * ratio ** k)) for k in range(int(np.log(hi / lo) / np.log(ratio)) + 1))))


# ---- 7. lemmas as checkable functions ---------------------------------------------------------------
def stability_ratio(eps, rel=0.1):
    """CONTRIBUTION 7. Level count under theta = 1 - eps changing by +-rel*eps: a mode of relaxation time T is kept
    at lag tau iff T >= tau/kappa(eps), kappa(e) = -ln(1-e). Two clusters can sit in different levels for every eps
    in the band at some common lag only if T_b/T_a > kappa(eps(1+rel))/kappa(eps(1-rel)). Returns that ratio.
    (a007 quotes 1.41; this function computes the ratio for the champion's convention theta = 1 - eps.)"""
    kap = lambda e: -np.log(1 - e)
    return float(kap(eps * (1 + rel)) / kap(eps * (1 - rel)))


def tolerance_stable(times, eps, rel=0.1):
    """CONTRIBUTION 7. True iff sorted adjacent relaxation times differ by more than stability_ratio(eps, rel)."""
    t = np.sort(np.asarray(times, float))
    return bool(np.all(t[1:] / t[:-1] > stability_ratio(eps, rel))) if len(t) > 1 else True


def closure_bound(B, P, t):
    """CONTRIBUTION 7. Lemma: || P B^t P - (P B P)^t ||_2  <=  (t-1) * min(|| (I-P) B P ||_2, || P B (I-P) ||_2)
    for an orthogonal projector P and ||B||_2 <= 1 (telescoping sum). Returns (lhs, rhs)."""
    I = np.eye(len(B))
    lhs = np.linalg.norm(P @ np.linalg.matrix_power(B, t) @ P - np.linalg.matrix_power(P @ B @ P, t), 2)
    rhs = (t - 1) * min(np.linalg.norm((I - P) @ B @ P, 2), np.linalg.norm(P @ B @ (I - P), 2))
    return float(lhs), float(rhs)


def closure_defect_identity(B, k):
    """CONTRIBUTION 7. Let Q = top-k eigenspace of Sym(B), Qf its complement, K = Q Anti(B) Qf^T. Then
    Q B^2 Q^T - (Q B Q^T)^2 = -K K^T exactly. Returns (norm of the difference, ||K K^T||). Limit: Q is the
    symmetric-part eigenspace, which is the champion's block only when B is reversible."""
    Sy, An = (B + B.T) / 2, (B - B.T) / 2
    w, V = np.linalg.eigh(Sy); V = V[:, np.argsort(-w)]
    Q, Qf = V[:, :k].T, V[:, k:].T
    K = Q @ An @ Qf.T
    mem = Q @ B @ B @ Q.T - (Q @ B @ Q.T) @ (Q @ B @ Q.T)
    return float(np.linalg.norm(mem + K @ K.T)), float(np.linalg.norm(K @ K.T))


# ---- pipeline, one-shot ---------------------------------------------------------------------------------
def ladder_flags(mom, taus, theta, cutter=cut, kappa=1.0):
    """ladder_m with a per-rung feasibility flag (CONTRIBUTION 5). Returns (maps, mu, flags); flags[j] = dict(tau,
    d_in, n_pairs, bbp, a003). The maps equal ladder_m's exactly."""
    mu, C, B = finalize(mom, taus)
    M = inv_sqrt(C); maps = []; flags = []
    for tau in taus:
        npairs = mom['lag'][tau][3]
        flags.append(dict(tau=tau, d_in=M.shape[0], n_pairs=npairs,
                          bbp=rung_feasible(theta, M.shape[0], npairs, kappa, 'bbp'),
                          a003=rung_feasible(theta, M.shape[0], npairs, kappa, 'a003')))
        Q = cutter(M @ B[tau] @ M.T, theta)
        if Q.shape[0] == 0:
            maps.append(np.zeros((0, M.shape[1]))); break
        QM = Q @ M
        if Q.shape[0] < M.shape[0]:
            M = inv_sqrt(QM @ C @ QM.T) @ QM
        maps.append(M.copy())
    return maps, mu, flags


def one_shot(mom, tau, theta, cutter=cut):
    """single cut at one lag from the whitened full-d moment (no iteration). Map on the centred embedding."""
    mu, C, B = finalize(mom, [tau]); M = inv_sqrt(C)
    return cutter(M @ B[tau] @ M.T, theta) @ M


def run_ladder(Xs, taus, theta=0.5, *, sign_mode="modulus", choose=False, alpha=0.05, floor_factor=None,
               window=1, diag=False, rng=None, kappa=1.0, nperm=499):
    """Whole pipeline with every contribution switchable. Defaults = champion (ladder_m on raw frames).
    Returns dict(maps, mu, dims, n_levels, feat, p_irrev, theta_eff, floor, flags, ...). taus are in window units
    when window > 1; maps act on the (centred) FEATURES returned by `feat`."""
    rng = np.random.default_rng(0) if rng is None else rng
    F, feat = make_features(Xs, window, diag)
    mom = build_moments(F, taus)
    Rs, _, _ = whiten(F)
    cutter = partial(cut, sign_mode=sign_mode); p = None; fl = None; th = theta
    if choose:
        cutter, p = choose_cutter(Rs, min(taus), alpha, rng, sign_mode, nperm)
    if floor_factor:
        fl = data_floor(Rs, rng); th = theta_from_floor(theta, fl['floor'], floor_factor)
    maps, mu, flags = ladder_flags(mom, taus, th, cutter, kappa)
    dims, nl = levels(maps)
    return dict(maps=maps, mu=mu, dims=dims, n_levels=nl, feat=feat, F=F, p_irrev=p, theta_eff=th, floor=fl,
                flags=flags, cutter=cutter, mom=mom)


# ---- recommended configuration (decided from test_final.py results; see FINAL_SOLUTION.md) -------------------
RECOMMENDED = dict(choose=True, sign_mode="modulus", floor_factor=None, window=1, diag=False)
"""Contributions that passed their head-to-head and hurt nowhere in the tested worlds are ON here (the reversibility gate).
Sign mode stays 'modulus' (lag-parity independent). Floor is OFF (no measurable effect at theta=0.5). window/diag are
world-dependent (help for variance/hidden-regime and energy-loss worlds, neutral elsewhere) and are chosen by the user."""


def run_recommended(Xs, taus, theta=0.5, **kw):
    """run_ladder with RECOMMENDED switches; any keyword overrides."""
    cfg = dict(RECOMMENDED); cfg.update(kw)
    return run_ladder(Xs, taus, theta, **cfg)


# ===================================================================== PART 3: decomposition read-out (shares and rebuild)
def decompose(Xs, M, mu, tau, fit=None):
    """Spectral decomposition of one ladder level: x - mu ~ sum_i c_i u_i, plus a remainder.
    M (m x d) = a level map from run_ladder (acts on x - mu), mu its centring vector, tau = the lag that level was cut at.
    s = M (x - mu) is the level's state (m-dim, unit covariance over the pooled frames). F = E[s_{t+tau} s_t^T] is the
    block's own dynamics. F = V diag(lam) V^{-1}: columns v_i = patterns (F v_i = lam_i v_i), rows w_i of V^{-1} =
    measurements (w_i^T F = lam_i w_i^T, w_i^T v_j = delta_ij). Per frame c_i = w_i^T s, so s = sum c_i v_i exactly and
    the predicted next amount is lam_i c_i. In frame space u_i = pinv(M) v_i, and
        x - mu = sum_i c_i u_i + (x - mu - P(x - mu)),   P = pinv(M) M the orthogonal projector on the rows of M,
    so the rebuilt part and the remainder are orthogonal and their squared norms add (Pythagoras).
    A complex pair (lam, conj lam) is one rotating plane: its contribution is 2 Re(c u), amplitude |c|, phase angle(c);
    the phase advances by angle(lam) per lag. Returns (list of dicts, one per video: C (T x m, complex), amp, phase,
    rebuilt (T x d), remainder (T x d), share (T,)), and a dict with lam, V, Winv (the left measurements), U (d x m).
    Limits: the individual amounts c_i are on the oblique basis, so per-mode energies do not add to the total share
    unless F is symmetric (compare `mode_share`, which is each mode alone); if F is defective (repeated eigenvalue with
    a missing eigenvector) V is ill-conditioned and `cond` is large - do not read c_i then. Since s is whitened, the
    average share of ANY m-dim block is m/d: read the per-frame and per-video shares, not the average."""
    s_list = [(X - mu) @ M.T for X in Xs]
    F = lag_moment(s_list if fit is None else [(X - mu) @ M.T for X in fit], tau)   # fit: data the dynamics are estimated on (default Xs itself)
    lam, V = np.linalg.eig(F)
    Winv = np.linalg.inv(V)
    Mp = np.linalg.pinv(M)
    U = Mp @ V
    out = []
    for X, S in zip(Xs, s_list):
        C = S @ Winv.T                                    # C[t,i] = w_i^T s_t  (complex for rotating modes)
        Xc = X - mu
        rebuilt = (C @ U.T).real                          # sum_i c_i u_i, conjugate partners add to 2 Re
        rem = Xc - rebuilt
        share = (rebuilt ** 2).sum(1) / np.maximum((Xc ** 2).sum(1), 1e-300)
        out.append(dict(C=C, amp=np.abs(C), phase=np.angle(C), rebuilt=rebuilt, remainder=rem, share=share))
    return out, dict(lam=lam, V=V, Winv=Winv, U=U, F=F, cond=float(np.linalg.cond(V)))


def mode_share(U, C, Xc):
    """each mode alone: |contribution of mode i|^2 / |x - mu|^2 per frame (T x m). Conjugate partners are counted
    together in the first of the pair and set to 0 in the second, so the columns are one per plane or real mode."""
    T, m = C.shape
    sh = np.zeros((T, m)); done = set()
    tot = np.maximum((Xc ** 2).sum(1), 1e-300)
    lam_idx = range(m)
    for i in lam_idx:
        if i in done: continue
        j = None
        if np.abs(U[:, i].imag).max() > 1e-12:           # complex: find conjugate partner
            d = np.abs(U - U[:, [i]].conj()).sum(0); d[i] = np.inf; j = int(np.argmin(d)); done.add(j)
            z = 2 * (C[:, [i]] * U[:, i][None, :]).real
        else:
            z = (C[:, [i]] * U[:, i][None, :]).real
        sh[:, i] = (z ** 2).sum(1) / tot
    return sh
