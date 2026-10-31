"""UCF101 split 1: ladder on frozen VideoMAE V2 features. Run: python fit_probe.py FEAT_DIR SPLITS_DIR OUT_JSON
FEAT_DIR has features/<id>.npy (8,d) and native/<id>.npy (d). SPLITS_DIR = ucfTrainTestlist.
Fit set = train groups 8-21; validation = train groups 22-25 (held out by group); test = official testlist01.
Linear probe = multinomial L2 logistic regression (same objective as sklearn, C chosen on validation only), solved by L-BFGS on GPU."""
import sys, os, re, json
import numpy as np, torch
import torch.nn.functional as TF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_final as lf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAUS, THETA, VAL_GROUPS, NCLS = [1, 2], 0.5, {22, 23, 24, 25}, 101
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def vid(p): return os.path.basename(p).replace(".avi", "")
def group(v): return int(re.search(r"_g(\d+)_", v).group(1))


def load_lists(spl):
    cls = {l.split()[1]: int(l.split()[0]) - 1 for l in open(f"{spl}/classInd.txt") if l.strip()}
    rd = lambda f: [l.split()[0] for l in open(f"{spl}/{f}") if l.strip()]
    tr, te = rd("trainlist01.txt"), rd("testlist01.txt")
    lab = lambda p: cls[p.split("/")[0]]
    return [vid(p) for p in tr], np.array([lab(p) for p in tr]), [vid(p) for p in te], np.array([lab(p) for p in te])


def rep(Xs, M, mu):
    """per video [mean_t z, std_t z], z = M (x - mu); M=None: original features."""
    out = []
    for X in Xs:
        Z = X - mu if M is None else (X - mu) @ M.T
        out.append(np.concatenate([Z.mean(0), Z.std(0)]))
    return np.array(out)


def train_softmax(F, y, C, iters=300):
    X = torch.tensor(F, dtype=torch.float32, device=DEV); Y = torch.tensor(y, device=DEV)
    W = torch.zeros(X.shape[1], NCLS, device=DEV, requires_grad=True); b = torch.zeros(NCLS, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)
    lam = 1.0 / (2 * C * len(X))                       # sklearn objective C*sum(loss)+0.5|w|^2, divided by C*n

    def closure():
        opt.zero_grad(); loss = TF.cross_entropy(X @ W + b, Y) + lam * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()


def predict(Wb, F):
    W, b = Wb
    return (torch.tensor(F, dtype=torch.float32, device=DEV) @ W + b).argmax(1).cpu().numpy()


def arm(name, Ffit, yfit, Fval, yval, Fall, yall, Fte, yte):
    """scaler from the fit set only; C chosen on validation; final probe trained on all train videos."""
    mu, sd = Ffit.mean(0), Ffit.std(0) + 1e-8
    z = lambda F: (F - mu) / sd
    best = max(((predict(train_softmax(z(Ffit), yfit, C), z(Fval)) == yval).mean(), C) for C in CS)
    pred = predict(train_softmax(z(Fall), yall, best[1]), z(Fte))
    acc = (pred == yte).mean()
    print(f"{name:34s} dim {Ffit.shape[1]:5d}  val {best[0]:.4f}  C {best[1]:g}  TEST {acc:.4f}", flush=True)
    return dict(dim=int(Ffit.shape[1]), val=float(best[0]), C=best[1], test=float(acc), correct=(pred == yte))


def boot_diff(c1, c2, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(c1); d = c1.astype(float) - c2.astype(float)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, [2.5, 97.5])


def main(feat, spl, out_json):
    trv, ytr, tev, yte = load_lists(spl)
    isv = np.array([group(v) in VAL_GROUPS for v in trv])
    fit_ids = [v for v, m in zip(trv, isv) if not m]; val_ids = [v for v, m in zip(trv, isv) if m]
    assert not (set(fit_ids) | set(val_ids)) & set(tev), "leakage: train/val ids overlap test ids"
    ld = lambda ids: [np.load(f"{feat}/features/{v}.npy").astype(np.float64) for v in ids]
    ldn = lambda ids: np.array([np.load(f"{feat}/native/{v}.npy").astype(np.float64) for v in ids])
    X_fit, X_val, X_all, X_te = ld(fit_ids), ld(val_ids), ld(trv), ld(tev)
    yfit, yval = ytr[~isv], ytr[isv]
    A = np.vstack(X_fit); mu = A.mean(0)
    R = {}
    run = lambda name, f: R.__setitem__(name, arm(name, f(X_fit, fit_ids), yfit, f(X_val, val_ids), yval, f(X_all, trv), ytr, f(X_te, tev), yte))

    run("0 native pooled vector (ref)", lambda Xs, ids: ldn(ids))
    run("1 mean", lambda Xs, ids: np.array([X.mean(0) for X in Xs]))
    run("2 mean+std", lambda Xs, ids: rep(Xs, None, mu))

    def levels(Xs):
        r = lf.run_ladder(Xs, TAUS, THETA)
        print("   rung dims", r["dims"], "flags", r["flags"], flush=True)
        return [(j, M) for j, M in enumerate(r["maps"], 1) if M.shape[0] > 0], r["mu"], [int(x) for x in r["dims"]]

    lev, lmu, dims = levels(X_fit)
    rows = []
    for j, M in lev:
        k = f"ladder level {j} (m={M.shape[0]})"
        run(k, lambda Xs, ids: rep(Xs, M, lmu))
        rows.append(dict(level=j, dim=int(M.shape[0]), **{q: R[k][q] for q in ("val", "test")}, key=k))
    if not rows:
        print("ladder returned no non-empty level"); json.dump(dict(ladder_dims=dims), open(out_json, "w")); return
    head = max(rows, key=lambda r: r["val"]); m = head["dim"]
    print(f"headline level {head['level']} (m={m}), chosen on validation", flush=True)
    w, U = np.linalg.eigh(np.cov((A - mu).T)); Pm = U[:, np.argsort(-w)[:m]].T
    run("3 PCA-m mean+std", lambda Xs, ids: rep(Xs, Pm, mu))
    R["4 ladder"] = R[head["key"]]

    rng = np.random.default_rng(0)                    # control: ladder fitted on time-shuffled fit sequences
    lev_s, mu_s, dims_s = levels([X[rng.permutation(len(X))] for X in X_fit])
    shuf = []
    for j, M in lev_s:
        k = f"shuffled-fit level {j} (m={M.shape[0]})"
        run(k, lambda Xs, ids: rep(Xs, M, mu_s)); shuf.append(dict(level=j, dim=int(M.shape[0]), val=R[k]["val"], test=R[k]["test"]))

    print("\n=== TEST top-1 (official split 1, %d videos) ===" % len(tev))
    for k in ["0 native pooled vector (ref)", "1 mean", "2 mean+std", "3 PCA-m mean+std", "4 ladder"]:
        print(f"{k:30s} dim {R[k]['dim']:5d}  {100 * R[k]['test']:.2f}")
    diffs = {}
    for base in ["2 mean+std", "3 PCA-m mean+std", "1 mean", "0 native pooled vector (ref)"]:
        dm, ci = boot_diff(R["4 ladder"]["correct"], R[base]["correct"])
        diffs[base] = [float(100 * dm), float(100 * ci[0]), float(100 * ci[1])]
        print(f"ladder - {base:28s} {100 * dm:+.2f} pts  95% CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]")
    pcl = lambda k: {int(c): float(R[k]["correct"][yte == c].mean()) for c in np.unique(yte)}
    json.dump(dict(ladder_dims=dims, shuffled_dims=dims_s, headline=head, levels=rows, shuffled_fit=shuf,
                   test={k: v["test"] for k, v in R.items()}, diffs=diffs,
                   per_class_ladder=pcl("4 ladder"), per_class_mean=pcl("1 mean")), open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:4])
