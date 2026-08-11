"""
03_validate.py  --  Task 3: validate the imputation.

Three masking designs, because a single design would answer only one question:

  Design A  MCAR cell masking      - 10% of observed cells hidden at random.
                                     Tests the easy, sporadic-missing case.
  Design B  block hold-out          - whole landmark blocks hidden one subject
                                     at a time, round-robin, so every
                                     (block, subject) pair is held out exactly
                                     once.  This mimics the real mechanism.
  Design C  burden-matched blocks   - a block is hidden in as many subjects as
                                     are genuinely missing it, so the fitting
                                     sample is as small as it is in production.
                                     The pessimistic, honest case.

Metrics: RMSE, MAE, normalised RMSE (RMSE / observed SD), bias, and the
imputed-vs-true correlation, reported per variable, per measurement type and
per anatomical region -- never as a single pooled number.  Interval coverage
is computed from the m=20 draws of the primary method.

Run:  python src/03_validate.py
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import utils as U
from imputers import SINGLE_METHODS, impute_hybrid_mi

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140

SEED = U.RANDOM_SEED
N_REP_A = 20
N_REP_C = 20
MASK_FRAC_A = 0.10
M_DRAWS = 20

# Methods run in the full comparison.  missForest is included but with fewer
# replicates because a single fit takes ~95 s against ~7 s for the others.
FAST_METHODS = ["median", "knn_k3", "knn_k5", "mice_ridge",
                "hybrid_pmm_nolog", "hybrid_pmm", "hybrid_pmm_m20"]
SLOW_METHODS = ["missforest"]
N_REP_SLOW = 3
CACHE = U.TAB / "_cache"
CACHE.mkdir(exist_ok=True)
# missForest costs ~95 s per fit, so it is evaluated on Design B only, which
# is the design that actually mimics the real mechanism.
SLOW_DESIGN = "B_block_holdout"


# ==========================================================================
# Mask designs
# ==========================================================================
def blocks_from_data(X: pd.DataFrame) -> list[dict]:
    """Recover the landmark blocks as sets of co-missing variables."""
    na = X.isna()
    groups: dict[tuple, list[str]] = {}
    for c in X.columns:
        groups.setdefault(tuple(na[c].to_numpy()), []).append(c)
    out = []
    for key, cols in groups.items():
        if sum(key) == 0:
            continue
        out.append({"vars": cols,
                    "missing_rows": np.array(key, dtype=bool),
                    "n_missing": int(sum(key))})
    return sorted(out, key=lambda d: -d["n_missing"])


def design_A(X, rng, n_rep=N_REP_A, frac=MASK_FRAC_A):
    """MCAR: hide a random `frac` of the observed cells."""
    obs = np.argwhere(X.notna().to_numpy())
    n = int(round(frac * len(obs)))
    for _ in range(n_rep):
        pick = obs[rng.choice(len(obs), n, replace=False)]
        mask = np.zeros(X.shape, bool)
        mask[pick[:, 0], pick[:, 1]] = True
        yield mask


def design_B(X):
    """Block hold-out, round-robin.

    Replicate r hides, for every block, the block's variables in the r-th
    subject that currently has them.  Every (block, subject) pair is held out
    exactly once across replicates, and several blocks are hidden at once so
    the run count stays modest.
    """
    blocks = blocks_from_data(X)
    cols = list(X.columns)
    avail = [np.flatnonzero(~b["missing_rows"]) for b in blocks]
    n_rep = max(len(a) for a in avail)
    for r in range(n_rep):
        mask = np.zeros(X.shape, bool)
        used = False
        for b, av in zip(blocks, avail):
            if r >= len(av):
                continue
            i = av[r]
            for v in b["vars"]:
                mask[i, cols.index(v)] = True
            used = True
        if used:
            yield mask


def design_C(X, rng, n_rep=N_REP_C):
    """Burden-matched: hide each block in as many *additional* subjects as are
    already missing it (capped so at least 6 fitting subjects survive)."""
    blocks = blocks_from_data(X)
    cols = list(X.columns)
    for _ in range(n_rep):
        mask = np.zeros(X.shape, bool)
        for b in blocks:
            av = np.flatnonzero(~b["missing_rows"])
            n_hide = min(b["n_missing"], max(0, len(av) - 6))
            if n_hide == 0:
                continue
            for i in rng.choice(av, n_hide, replace=False):
                for v in b["vars"]:
                    mask[i, cols.index(v)] = True
        yield mask


# ==========================================================================
# Scoring
# ==========================================================================
def collect(X, meta, aux, mask, method_name, rng):
    """Apply one method to one masked dataset; return per-cell truths/preds."""
    Xm = X.mask(mask)
    fn = SINGLE_METHODS[method_name]
    F = fn(Xm, meta, aux, rng)
    ii, jj = np.where(mask)
    return pd.DataFrame({
        "row": ii, "col": jj,
        "variable": np.array(X.columns)[jj],
        "true": X.to_numpy()[ii, jj],
        "pred": F.to_numpy()[ii, jj],
        "method": method_name,
    })


def score(recs: pd.DataFrame, obs_sd: pd.Series) -> pd.DataFrame:
    """Per-variable error metrics."""
    recs = recs.dropna(subset=["true", "pred"])
    g = recs.groupby(["method", "variable"])
    out = g.apply(lambda d: pd.Series({
        "n": len(d),
        "rmse": float(np.sqrt(np.mean((d.pred - d.true) ** 2))),
        "mae": float(np.mean(np.abs(d.pred - d.true))),
        "bias": float(np.mean(d.pred - d.true)),
        "r": float(np.corrcoef(d.pred, d.true)[0, 1]) if d.true.nunique() > 2 else np.nan,
    })).reset_index()
    out["sd_obs"] = out.variable.map(obs_sd)
    out["nrmse"] = out.rmse / out.sd_obs
    out["rel_bias_pct"] = out.bias / out.sd_obs * 100
    out["type"] = out.variable.map(U.measure_type)
    out["region"] = out.variable.map(U.region)
    return out


def summarise(sc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (sc.groupby(by)
            .agg(n_vars=("variable", "nunique"),
                 n_cells=("n", "sum"),
                 median_nrmse=("nrmse", "median"),
                 mean_nrmse=("nrmse", "mean"),
                 mean_mae=("mae", "mean"),
                 mean_abs_rel_bias=("rel_bias_pct", lambda x: np.mean(np.abs(x))),
                 mean_r=("r", "mean"))
            .round(3).reset_index())


# ==========================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    df = U.load_raw()
    meas = U.measurement_columns(df)
    X = df[meas].astype(float)
    meta = U.variable_metadata(df)
    aux = df[["age_years", "sex_M"]].astype(float)
    obs_sd = X.std(ddof=1)

    designs = {
        "A_mcar_cells": list(design_A(X, rng)),
        "B_block_holdout": list(design_B(X)),
        "C_burden_matched": list(design_C(X, rng)),
    }
    for k, v in designs.items():
        cells = int(np.mean([m.sum() for m in v]))
        print(f"{k}: {len(v)} replicates, ~{cells} held-out cells each")

    # Each design's per-cell results are cached to disk.  Re-running the
    # script resumes from whatever is already on disk, which makes the long
    # missForest replicates restartable.
    all_scores = []
    for dname, masks in designs.items():
        cache = U.TAB / f"validation_cells_{dname}.csv"
        recs = []
        for r, mask in enumerate(masks):
            # per-replicate cache, so a long run can be stopped and resumed
            rc = CACHE / f"{dname}_rep{r:02d}.csv"
            if rc.exists():
                recs.append(pd.read_csv(rc))
                continue
            part = [collect(X, meta, aux, mask, meth,
                            np.random.default_rng(SEED + 1000 * r))
                    for meth in FAST_METHODS]
            if r < N_REP_SLOW:
                part += [collect(X, meta, aux, mask, meth,
                                 np.random.default_rng(SEED + 1000 * r))
                         for meth in SLOW_METHODS]
            part = pd.concat(part, ignore_index=True)
            part.to_csv(rc, index=False)
            recs.append(part)
            print(f"  {dname} rep {r + 1}/{len(masks)}", flush=True)
        recs = pd.concat(recs, ignore_index=True)
        recs.to_csv(cache, index=False)
        sc = score(recs, obs_sd)
        sc.insert(0, "design", dname)
        all_scores.append(sc)
        print(f"  scored {dname}")

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(U.TAB / "validation_by_variable.csv", index=False)

    meth_sum = summarise(scores, ["design", "method"])
    meth_sum.to_csv(U.TAB / "validation_method_comparison.csv", index=False)
    print("\n=== Method comparison (lower nRMSE is better) ===")
    print(meth_sum.pivot(index="method", columns="design",
                         values="median_nrmse").round(3).to_string())

    best = "hybrid_pmm_m20"
    type_sum = summarise(scores[scores.method == best], ["design", "type"])
    reg_sum = summarise(scores[scores.method == best], ["design", "region"])
    type_sum.to_csv(U.TAB / "validation_by_type.csv", index=False)
    reg_sum.to_csv(U.TAB / "validation_by_region.csv", index=False)
    print(f"\n=== {best}: error by measurement type ===")
    print(type_sum.to_string(index=False))
    print(f"\n=== {best}: error by anatomical region ===")
    print(reg_sum.to_string(index=False))

    # interval coverage from the m=20 draws (Design B)
    cov_cache = U.TAB / "validation_coverage_cells.csv"
    cov_rows = []
    for r, mask in enumerate(designs["B_block_holdout"]):
        rc = CACHE / f"cov_rep{r:02d}.csv"
        if rc.exists():
            cov_rows.append(pd.read_csv(rc))
            continue
        Xm = X.mask(mask)
        frames, _ = impute_hybrid_mi(Xm, meta, aux,
                                     np.random.default_rng(SEED + r),
                                     m=M_DRAWS, return_all=True)
        arr = np.stack([f.to_numpy(float) for f in frames])
        ii, jj = np.where(mask)
        lo = np.percentile(arr[:, ii, jj], 2.5, axis=0)
        hi = np.percentile(arr[:, ii, jj], 97.5, axis=0)
        truth = X.to_numpy()[ii, jj]
        part = pd.DataFrame({
            "variable": np.array(X.columns)[jj],
            "true": truth, "lo": lo, "hi": hi,
            "width": hi - lo,
            "covered": (truth >= lo) & (truth <= hi),
        })
        part.to_csv(rc, index=False)
        cov_rows.append(part)
    cov = pd.concat(cov_rows, ignore_index=True).dropna(subset=["true"])
    cov["type"] = cov.variable.map(U.measure_type)
    cov["region"] = cov.variable.map(U.region)
    cov["rel_width"] = cov.width / cov.variable.map(obs_sd)
    cov_sum = (cov.groupby(["type", "region"])
               .agg(n=("covered", "size"),
                    coverage_95=("covered", "mean"),
                    mean_rel_width=("rel_width", "mean"))
               .round(3).reset_index())
    cov.to_csv(U.TAB / "validation_coverage_cells.csv", index=False)
    cov_sum.to_csv(U.TAB / "validation_coverage.csv", index=False)
    print("\n=== 95% interval coverage of the m=20 predictive distribution ===")
    print(cov_sum.to_string(index=False))
    print(f"overall coverage = {cov.covered.mean():.3f}")

    # 3.3  are anatomical relationships preserved?
    imp = pd.read_csv(U.OUT / "craniofacial_imputed.csv")[meas]
    inc = [c for c in meas if X[c].isna().any()]
    cc_rows = X[inc].notna().all(axis=1)
    corr_cc = X.loc[cc_rows, inc].corr(method="spearman")
    corr_imp = imp[inc].corr(method="spearman")
    iu = np.triu_indices(len(inc), 1)
    d_corr = corr_imp.to_numpy()[iu] - corr_cc.to_numpy()[iu]
    print(f"\nCorrelation structure among the {len(inc)} incomplete variables: "
          f"mean |delta rho| = {np.nanmean(np.abs(d_corr)):.3f}, "
          f"max = {np.nanmax(np.abs(d_corr)):.3f} "
          f"(complete-case n = {int(cc_rows.sum())})")
    pd.DataFrame({"delta_rho": d_corr}).to_csv(
        U.TAB / "validation_corr_shift.csv", index=False)

    # bilateral asymmetry index: should stay centred near zero
    asym = []
    for _, r in meta.iterrows():
        if not isinstance(r["mirror"], str) or r["side"] != "L":
            continue
        L, R = imp[r["variable"]], imp[r["mirror"]]
        was_imp = X[r["variable"]].isna() | X[r["mirror"]].isna()
        ai = (L - R) / ((L + R) / 2) * 100
        asym.append(pd.DataFrame({"variable": r["variable"], "ai": ai,
                                  "source": np.where(was_imp, "imputed", "observed")}))
    asym = pd.concat(asym, ignore_index=True).dropna()
    asym.to_csv(U.TAB / "validation_asymmetry_index.csv", index=False)
    print("\nBilateral asymmetry index (%, (L-R)/mean):")
    print(asym.groupby("source")["ai"].describe()[["count", "mean", "std"]]
          .round(2).to_string())

    # fig08 method comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=meth_sum, x="method", y="median_nrmse", hue="design", ax=ax)
    ax.axhline(1.0, ls="--", c="grey", lw=1)
    ax.text(0.01, 1.02, "nRMSE = 1 -> no better than the variable's own SD",
            transform=ax.get_yaxis_transform(), fontsize=7, color="grey")
    ax.set_ylabel("median nRMSE (RMSE / observed SD)")
    ax.set_title("Imputation accuracy by method and masking design")
    plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(U.FIG / "fig08_method_comparison.png"); plt.close(fig)

    # fig09 error by type and region, primary method
    sub = scores[(scores.method == best)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=sub, x="type", y="nrmse", hue="design", ax=axes[0])
    axes[0].axhline(1, ls="--", c="grey", lw=1)
    axes[0].set_title("Distances impute well; angles do not")
    sns.boxplot(data=sub[sub.design == "B_block_holdout"], x="region", y="nrmse",
                ax=axes[1], color="#4c72b0")
    axes[1].axhline(1, ls="--", c="grey", lw=1)
    axes[1].set_title("Design B: error by anatomical region")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(f"Per-variable normalised error, {best}")
    fig.tight_layout(); fig.savefig(U.FIG / "fig09_error_by_type_region.png"); plt.close(fig)

    # fig10 imputed vs true scatter (Design B)
    cellsB = pd.read_csv(U.TAB / "validation_cells_B_block_holdout.csv")
    cb = cellsB[cellsB.method == best].dropna()
    cb["type"] = cb.variable.map(U.measure_type)
    cb["z_true"] = cb.groupby("variable")["true"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) else 1))
    cb["z_pred"] = (cb["pred"] - cb.groupby("variable")["true"].transform("mean")) / \
                   cb.groupby("variable")["true"].transform(
                       lambda s: s.std(ddof=1) if s.std(ddof=1) else 1)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sns.scatterplot(data=cb, x="z_true", y="z_pred", hue="type", alpha=0.6, s=22, ax=ax)
    lim = [cb.z_true.min() - .3, cb.z_true.max() + .3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("true value (z, within variable)")
    ax.set_ylabel("imputed value (z)")
    ax.set_title("Design B: imputed vs held-out true values")
    fig.tight_layout(); fig.savefig(U.FIG / "fig10_imputed_vs_true.png"); plt.close(fig)

    # fig11 observed vs imputed distributions for exemplar variables
    show = ["N-ANS", "ANS-PNS", "SNA", "S-N-ANS", "TM-TMR", "JL-JR"]
    show = [v for v in show if v in meas]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, v in zip(axes.ravel(), show):
        o = X[v].dropna()
        i = imp.loc[X[v].isna(), v]
        sns.kdeplot(o, ax=ax, fill=True, label=f"observed (n={len(o)})", cut=0)
        if len(i) > 1:
            sns.kdeplot(i, ax=ax, fill=True, label=f"imputed (n={len(i)})", cut=0)
        ks = stats.ks_2samp(o, i) if len(i) > 1 else None
        ttl = v if ks is None else f"{v}   KS p={ks.pvalue:.2f}"
        ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("")
    fig.suptitle("Observed vs imputed distributions (a legitimate shift is possible "
                 "under MAR, so this is a plausibility check, not a test)", fontsize=9)
    fig.tight_layout(); fig.savefig(U.FIG / "fig11_distributions.png"); plt.close(fig)

    # fig12 correlation preservation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(corr_cc.to_numpy()[iu], corr_imp.to_numpy()[iu], s=6, alpha=0.4)
    axes[0].plot([-1, 1], [-1, 1], "k--", lw=1)
    axes[0].set_xlabel("Spearman rho, complete cases only")
    axes[0].set_ylabel("Spearman rho, after imputation")
    axes[0].set_title(f"Pairwise correlations preserved\nmean |delta| = "
                      f"{np.nanmean(np.abs(d_corr)):.3f}")
    sns.histplot(data=asym, x="ai", hue="source", stat="density", common_norm=False,
                 element="step", ax=axes[1], bins=30)
    axes[1].set_xlabel("bilateral asymmetry index (%)")
    axes[1].set_title("Left-right asymmetry: observed vs imputed pairs")
    fig.tight_layout(); fig.savefig(U.FIG / "fig12_structure_preservation.png"); plt.close(fig)

    # fig13 coverage
    fig, ax = plt.subplots(figsize=(7, 4))
    cs = cov.groupby(["type", "region"]).covered.mean().reset_index()
    sns.barplot(data=cs, x="region", y="covered", hue="type", ax=ax)
    ax.axhline(0.95, ls="--", c="red", lw=1, label="nominal 95%")
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Coverage of the 95% predictive interval (Design B)")
    fig.tight_layout(); fig.savefig(U.FIG / "fig13_coverage.png"); plt.close(fig)

    print(f"\nWrote validation tables to {U.TAB} and figures to {U.FIG}")


if __name__ == "__main__":
    main()
