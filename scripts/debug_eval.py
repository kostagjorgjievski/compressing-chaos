# debug_eval.py
import numpy as np
import torch


@torch.no_grad()
def latent_mismatch_report(vae, diffusion, real_ds, device, n_real=5000, n_diff=5000, seed=42):
    rng = np.random.default_rng(seed)
    n_real = min(n_real, len(real_ds))
    idx = rng.integers(0, len(real_ds), size=n_real)

    # Collect real windows
    xs = []
    for i in idx:
        x, _ = real_ds[int(i)]
        xs.append(x.unsqueeze(0))  # [1, T, 1]
    X = torch.cat(xs, dim=0).to(device)  # [N, T, 1]

    # Encode real
    # Make sure x is [B, T, C]
    if x.ndim == 2:
        # either [T, C] or [B, T]
        # if last dim looks like channels (usually 1), assume [T, C]
        if x.shape[-1] <= 8:   # safe heuristic for your case (C=1)
            x = x.unsqueeze(0)  # [1, T, C]
        else:
            x = x.unsqueeze(-1) # [B, T, 1]
    elif x.ndim == 1:
        # [T] -> [1, T, 1]
        x = x.unsqueeze(0).unsqueeze(-1)

    enc = vae.encode(x)


    mu = logvar = None

    if isinstance(enc, (tuple, list)):
        if len(enc) >= 2:
            mu, logvar = enc[0], enc[1]
    elif isinstance(enc, dict):
        mu = enc.get("mu", None)
        logvar = enc.get("logvar", None)

    if mu is None or logvar is None:
        raise RuntimeError(
            f"vae.encode output format not recognized. Got type={type(enc)}; "
            f"expected (mu, logvar, ...) or dict with mu/logvar."
        )

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z_sample_real = mu + std * eps

    # Diffusion samples
    z_diff = diffusion.sample(num_samples=n_diff, device=device)

    # Move to numpy
    mu_np = mu.detach().cpu().numpy()
    zs_np = z_sample_real.detach().cpu().numpy()
    zd_np = z_diff.detach().cpu().numpy()

    def summarize(Z, tag):
        per_dim_mean = Z.mean(axis=0)
        per_dim_std = Z.std(axis=0)
        avg_var = float(np.mean(per_dim_std ** 2))
        cov_trace = float(np.trace(np.cov(Z, rowvar=False)))
        norms = np.linalg.norm(Z, axis=1)
        print(f"\nLATENT SUMMARY: {tag}")
        print(f"  Z shape: {Z.shape}")
        print(f"  per-dim mean: mean={per_dim_mean.mean():.4f} std_across_dims={per_dim_mean.std():.4f}")
        print(f"  per-dim std : mean={per_dim_std.mean():.4f} std_across_dims={per_dim_std.std():.4f}")
        print(f"  avg_var(per-dim): {avg_var:.4f}")
        print(f"  cov_trace: {cov_trace:.4f}")
        print(f"  ||Z||: mean={norms.mean():.4f} std={norms.std():.4f} p10/p50/p90={np.quantile(norms, [0.1, 0.5, 0.9])}")

    summarize(mu_np, "real z_mu")
    summarize(zs_np, "real z_sample (mu+sigma*eps)")
    summarize(zd_np, "diffusion z_sample")

    real_norm_med = np.quantile(np.linalg.norm(zs_np, axis=1), 0.5)
    diff_norm_med = np.quantile(np.linalg.norm(zd_np, axis=1), 0.5)
    ratio = diff_norm_med / (real_norm_med + 1e-8)
    print(f"\nMEDIAN NORM RATIO (diff / real_sample): {ratio:.3f}")
    if ratio < 0.85:
        print("Verdict: diffusion latents look under-spread vs real posterior samples (under-dispersion).")
    elif ratio > 1.15:
        print("Verdict: diffusion latents look over-spread vs real posterior samples (may cause extremes).")
    else:
        print("Verdict: latent scale roughly matched. Tail mismatch may still exist (check p90/p99).")


def _squeeze(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    assert x.ndim == 2, f"Expected [N,T], got {x.shape}"
    return x


def moments(x):
    x = np.asarray(x).reshape(-1)
    m = x.mean()
    s = x.std(ddof=1) + 1e-12
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4) - 3.0
    return dict(mean=float(m), std=float(s), skew=float(skew), excess_kurt=float(kurt))


def quantiles(x, qs=(0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)):
    x = np.asarray(x).reshape(-1)
    return {f"q{int(q * 1000):04d}": float(np.quantile(x, q)) for q in qs}


def acf_1d(x, max_lag=20):
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    res = []
    for k in range(1, max_lag + 1):
        res.append(float(np.dot(x[:-k], x[k:]) / denom))
    return np.array(res)


def mean_acf_windows(X, max_lag=20):
    acfs = np.stack([acf_1d(w, max_lag=max_lag) for w in X], axis=0)
    return acfs.mean(axis=0), acfs.std(axis=0, ddof=1)


def ks_statistic(a, b):
    a = np.sort(np.asarray(a).reshape(-1))
    b = np.sort(np.asarray(b).reshape(-1))
    n, m = len(a), len(b)
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < n and j < m:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / n
        else:
            j += 1
            cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))
    return float(d)


def realized_vol(window):
    w = np.asarray(window)
    return float(np.sqrt(np.mean(w * w)))


def var_es(x, alpha=0.01):
    x = np.asarray(x).reshape(-1)
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    es = tail.mean() if len(tail) > 0 else np.nan
    return float(q), float(es)


def max_drawdown(window):
    """
    Correct MDD for LOG-RETURNS:
      equity_t = exp(cumsum(r_t))
    """
    r = np.asarray(window).reshape(-1)
    log_eq = np.cumsum(r)
    eq = np.exp(log_eq - np.max(log_eq))  # stabilize; scaling doesn't change drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    return float(dd.min())


def print_report(name, out):
    print("\n" + "=" * 90)
    print(f"{name}")
    print("=" * 90)
    mr, mg = out["moments_real"], out["moments_gen"]
    print(f"Mean/Std (real): {mr['mean']:+.4e} / {mr['std']:.4e}")
    print(f"Mean/Std (gen) : {mg['mean']:+.4e} / {mg['std']:.4e}   | std_ratio={out['std_ratio']:.3f}")
    print(f"Skew/Kurt (real): {mr['skew']:+.3f} / {mr['excess_kurt']:+.3f}")
    print(f"Skew/Kurt (gen) : {mg['skew']:+.3f} / {mg['excess_kurt']:+.3f}")
    print(f"KS: {out['ks']:.4f}")
    print(f"VaR1% real/gen: {out['VaR1_real']:+.4e} / {out['VaR1_gen']:+.4e}")
    print(f"ES1%  real/gen: {out['ES1_real']:+.4e} / {out['ES1_gen']:+.4e}")
    print(f"ACF lags 1..5 real: {out['acf_mean_real_lag1_5']}")
    print(f"ACF lags 1..5 gen : {out['acf_mean_gen_lag1_5']}")
    print(f"MDD median real/gen: {out['MDD_real_median']:+.3f} / {out['MDD_gen_median']:+.3f}")
    print(f"MDD p95    real/gen: {out['MDD_real_p95']:+.3f} / {out['MDD_gen_p95']:+.3f}")


def summarize_block(name, Xr, Xg):
    Xr = np.asarray(Xr)
    Xg = np.asarray(Xg)

    r_flat = Xr.reshape(-1)
    g_flat = Xg.reshape(-1)

    out = {}
    out["moments_real"] = moments(r_flat)
    out["moments_gen"] = moments(g_flat)
    out["quant_real"] = quantiles(r_flat)
    out["quant_gen"] = quantiles(g_flat)

    acf_mean_r, _ = mean_acf_windows(Xr, max_lag=20)
    acf_mean_g, _ = mean_acf_windows(Xg, max_lag=20)
    out["acf_mean_real_lag1_5"] = [float(x) for x in acf_mean_r[:5]]
    out["acf_mean_gen_lag1_5"] = [float(x) for x in acf_mean_g[:5]]

    out["ks"] = ks_statistic(r_flat, g_flat)

    var_r, es_r = var_es(r_flat, alpha=0.01)
    var_g, es_g = var_es(g_flat, alpha=0.01)
    out["VaR1_real"] = var_r
    out["ES1_real"] = es_r
    out["VaR1_gen"] = var_g
    out["ES1_gen"] = es_g

    dd_r = np.array([max_drawdown(w) for w in Xr])
    dd_g = np.array([max_drawdown(w) for w in Xg])
    out["MDD_real_median"] = float(np.median(dd_r))
    out["MDD_real_p95"] = float(np.quantile(dd_r, 0.95))
    out["MDD_gen_median"] = float(np.median(dd_g))
    out["MDD_gen_p95"] = float(np.quantile(dd_g, 0.95))

    out["std_ratio"] = float(out["moments_gen"]["std"] / (out["moments_real"]["std"] + 1e-12))
    print_report(name, out)
    return out


def inspect_vix(vix: np.ndarray, k: int = 10, name: str = "VIX"):
    vix = np.asarray(vix).reshape(-1)
    print(f"\nVIX INSPECT: {name}")
    print(f"shape={vix.shape} mean={vix.mean():.6f} std={vix.std():.6f} min={vix.min():.6f} max={vix.max():.6f}")
    idx = np.linspace(0, len(vix) - 1, num=min(k, len(vix)), dtype=int)
    print("sample values:", vix[idx])


def vix_bins(vix: np.ndarray, mode: str = "auto"):
    vix = np.asarray(vix).astype(np.float32).reshape(-1)

    if mode == "auto":
        mode = "pct" if float(vix.min()) < 0.0 else "raw"

    if mode == "raw":
        bins = {
            "low(<15)": (vix < 15.0),
            "mid(15-25)": (vix >= 15.0) & (vix < 25.0),
            "high(25-35)": (vix >= 25.0) & (vix < 35.0),
            "extreme(>=35)": (vix >= 35.0),
        }
        return bins, {}

    p50 = float(np.percentile(vix, 50))
    p80 = float(np.percentile(vix, 80))
    p95 = float(np.percentile(vix, 95))
    cutoffs = {"p50": p50, "p80": p80, "p95": p95}

    bins = {
        f"low(<p50:{p50:.3f})": (vix < p50),
        f"mid(p50-p80:{p50:.3f}-{p80:.3f})": (vix >= p50) & (vix < p80),
        f"high(p80-p95:{p80:.3f}-{p95:.3f})": (vix >= p80) & (vix < p95),
        f"extreme(>=p95:{p95:.3f})": (vix >= p95),
    }
    return bins, cutoffs


def stress_sweep(X_real, X_gen, vix, name="MODEL", bin_mode="auto", min_n=50):
    X_real = np.asarray(X_real)
    X_gen = np.asarray(X_gen)
    vix = np.asarray(vix).reshape(-1)

    if X_real.ndim == 3:
        X_real = _squeeze(X_real)
    if X_gen.ndim == 3:
        X_gen = _squeeze(X_gen)

    assert X_real.shape[0] == X_gen.shape[0] == len(vix), "vix and windows must align"

    bins, cutoffs = vix_bins(vix, mode=bin_mode)

    print("\n" + "#" * 90)
    print(f"STRESS SWEEP BY VIX BINS: {name}")
    print("#" * 90)

    if cutoffs:
        print("\nUsing PERCENTILE VIX bins:")
        print("cutoffs:", cutoffs)

    for bname, mask in bins.items():
        n = int(mask.sum())
        print(f"\n[{bname}] n={n}" + ("" if n >= min_n else " (too small, skipping reliable stats)"))
        if n < min_n:
            continue

        xr = X_real[mask]
        xg = X_gen[mask]

        r = xr.reshape(n, -1)
        g = xg.reshape(n, -1)

        mean_r, std_r = float(r.mean()), float(r.std() + 1e-8)
        mean_g, std_g = float(g.mean()), float(g.std() + 1e-8)
        std_ratio = std_g / std_r

        def skew_kurt(a):
            a = a.reshape(-1)
            mu = a.mean()
            s = a.std() + 1e-8
            z = (a - mu) / s
            skew = float((z ** 3).mean())
            kurt = float((z ** 4).mean() - 3.0)
            return skew, kurt

        skew_r, kurt_r = skew_kurt(r)
        skew_g, kurt_g = skew_kurt(g)

        var1_r = float(np.quantile(r, 0.01))
        var1_g = float(np.quantile(g, 0.01))
        es1_r = float(r[r <= var1_r].mean()) if np.any(r <= var1_r) else float("nan")
        es1_g = float(g[g <= var1_g].mean()) if np.any(g <= var1_g) else float("nan")

        dd_r = np.array([max_drawdown(w) for w in xr])
        dd_g = np.array([max_drawdown(w) for w in xg])

        print("\n" + "=" * 90)
        print(f"{name} | {bname}")
        print("=" * 90)
        print(f"Mean/Std (real): {mean_r:+.4e} / {std_r:.4e}")
        print(f"Mean/Std (gen) : {mean_g:+.4e} / {std_g:.4e}   | std_ratio={std_ratio:.3f}")
        print(f"Skew/Kurt (real): {skew_r:+.3f} / {kurt_r:+.3f}")
        print(f"Skew/Kurt (gen) : {skew_g:+.3f} / {kurt_g:+.3f}")
        print(f"VaR1% real/gen: {var1_r:+.4e} / {var1_g:+.4e}")
        print(f"ES1%  real/gen: {es1_r:+.4e} / {es1_g:+.4e}")
        print(f"MDD median real/gen: {np.median(dd_r):+.3f} / {np.median(dd_g):+.3f}")
        print(f"MDD p95    real/gen: {np.quantile(dd_r,0.95):+.3f} / {np.quantile(dd_g,0.95):+.3f}")


def conditioning_check(X_gen, vix, name="MODEL", bin_mode="auto"):
    X_gen = _squeeze(X_gen)
    vix = np.asarray(vix).reshape(-1)

    vols = np.array([realized_vol(w) for w in X_gen])
    corr = np.corrcoef(vix, vols)[0, 1]

    print("\n" + "#" * 90)
    print(f"CONDITIONING CHECK: {name}")
    print("#" * 90)
    print(f"Corr(VIX, realized_vol_gen) = {corr:.3f}")
    print("Vol quantiles by VIX bin (gen):")

    bins, _ = vix_bins(vix, mode=bin_mode)
    for bname, mask in bins.items():
        if mask.sum() < 50:
            continue
        q10, q50, q90 = np.quantile(vols[mask], [0.1, 0.5, 0.9])
        print(f"  {bname:24s} n={int(mask.sum()):4d}  vol q10/q50/q90: {q10:.4e} / {q50:.4e} / {q90:.4e}")


if __name__ == "__main__":
    raise SystemExit("Import and call stress_sweep(X_real, X_gen, vix) from your training notebook/script.")
