"""
01_diagnostics.py  --  Task 1: explore and diagnose the missing data.

Outputs
  tables/  missing_by_variable.csv, missing_by_subject.csv,
           missing_patterns.csv, landmark_missingness.csv,
           mechanism_tests.csv, unexplained_cells.csv,
           bilateral_comissingness.csv
  figures/ fig01..fig07

Run:  python src/01_diagnostics.py
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
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import utils as U

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 140


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = p[idx] * n / (n - rank + 1)
        prev = min(prev, val)
        adj[idx] = prev
    return np.clip(adj, 0, 1)


def main() -> None:
    df = U.load_raw()
    meas = U.measurement_columns(df)
    X = df[meas]
    na = X.isna()
    meta = U.variable_metadata(df)
    meta.to_csv(U.TAB / "variable_metadata.csv", index=False)

    n_sub, n_var = X.shape
    print(f"Subjects: {n_sub} | measurements: {n_var} | "
          f"cells missing: {na.values.sum()} "
          f"({na.values.mean() * 100:.2f}%)")

    # Missingness per variable
    by_var = (meta[["variable", "type", "region", "side", "mirror",
                    "n_missing", "pct_missing"]]
              .sort_values(["pct_missing", "variable"], ascending=[False, True]))
    by_var.to_csv(U.TAB / "missing_by_variable.csv", index=False)

    # Missingness per subject
    by_sub = pd.DataFrame({
        U.ID_COL: df[U.ID_COL],
        "age_years": df["age_years"].round(3),
        "sex": df["gender"],
        "n_missing": na.sum(axis=1).values,
        "pct_missing": (na.mean(axis=1) * 100).round(2).values,
    }).sort_values("pct_missing", ascending=False)
    by_sub.to_csv(U.TAB / "missing_by_subject.csv", index=False)

    # Outlier flag: > median + 1.5*IQR of subject-level missingness
    # Subject-level missingness is strongly bimodal (a group of complete
    # records and a group with whole blocks absent), so a Tukey fence on the
    # pooled distribution is uninformative.  We therefore report both: the
    # Tukey fence, and the fence computed within the affected subjects only.
    q1, q3 = np.percentile(by_sub["pct_missing"], [25, 75])
    thresh_all = q3 + 1.5 * (q3 - q1)
    aff = by_sub.loc[by_sub.pct_missing > 0, "pct_missing"]
    aq1, aq3 = np.percentile(aff, [25, 75])
    thresh = aq3 + 1.5 * (aq3 - aq1)
    print(f"Subjects with any missingness: {len(aff)}/{n_sub} "
          f"(range {aff.min():.1f}-{aff.max():.1f}%); "
          f"{n_sub - len(aff)} subjects are complete.")
    print(f"Tukey fence over all subjects = {thresh_all:.1f}% (flags none); "
          f"fence within affected subjects = {thresh:.1f}% -> flags "
          f"{by_sub.loc[by_sub.pct_missing > thresh, U.ID_COL].tolist()}")

    # Distinct column-missingness patterns == co-missing blocks
    lm_to_vars: dict[str, list[str]] = {}
    for c in meas:
        for lm in U.landmarks(c):
            lm_to_vars.setdefault(lm, []).append(c)

    lm_mat_all = U.landmark_missing_matrix(df)

    pattern_of: dict[tuple, list[str]] = {}
    for c in meas:
        pattern_of.setdefault(tuple(na[c].values), []).append(c)

    rows = []
    for key, cols in sorted(pattern_of.items(), key=lambda kv: -sum(kv[0])):
        if sum(key) == 0:
            continue
        subs = [int(df[U.ID_COL].iloc[i]) for i, b in enumerate(key) if b]
        # A landmark "drives" a block when its inferred availability pattern
        # is exactly the block's missingness pattern: undigitised in precisely
        # the subjects that lost the block, present in all the others.
        ind_arr = np.array(key, dtype=bool)
        common = sorted(lm for lm in lm_mat_all.columns
                        if np.array_equal(lm_mat_all[lm].to_numpy(dtype=bool),
                                          ind_arr))
        rows.append({
            "block_id": f"B{len(rows) + 1}",
            "n_variables": len(cols),
            "n_subjects_missing": sum(key),
            "pct_subjects_missing": round(sum(key) / n_sub * 100, 1),
            "subjects": ";".join(map(str, subs)),
            "driving_landmarks": ";".join(common) or "(none)",
            "variables": ";".join(sorted(cols)),
        })
    patterns = pd.DataFrame(rows)
    patterns.to_csv(U.TAB / "missing_patterns.csv", index=False)
    print(f"\nDistinct missingness blocks: {len(patterns)}")
    print(patterns[["block_id", "n_variables", "n_subjects_missing",
                    "driving_landmarks"]].to_string(index=False))

    # Landmark-level view + how many cells the landmark rule explains
    lm_mat = lm_mat_all
    lm_mat.to_csv(U.TAB / "landmark_missingness.csv")
    missing_lms = lm_mat.loc[:, lm_mat.any(axis=0)]

    explained = np.zeros_like(na.values, dtype=bool)
    for j, c in enumerate(meas):
        lms = U.landmarks(c)
        flag = np.zeros(n_sub, dtype=bool)
        for lm in lms:
            if lm in lm_mat.columns:
                flag |= lm_mat[lm].to_numpy(dtype=bool)
        explained[:, j] = flag
    n_missing = na.values.sum()
    n_explained = (na.values & explained).sum()
    n_unexpl = (na.values & ~explained).sum()
    n_false_pos = (~na.values & explained).sum()
    print(f"\nLandmark rule: {n_explained}/{n_missing} missing cells explained "
          f"({n_explained / n_missing * 100:.1f}%); "
          f"{n_unexpl} unexplained; {n_false_pos} observed-but-predicted-missing")

    unexpl_idx = np.argwhere(na.values & ~explained)
    unexpl = pd.DataFrame([
        {U.ID_COL: int(df[U.ID_COL].iloc[i]), "variable": meas[j],
         "note": "missing but all its landmarks observed elsewhere"}
        for i, j in unexpl_idx])
    unexpl.to_csv(U.TAB / "unexplained_cells.csv", index=False)
    if len(unexpl):
        print(unexpl.to_string(index=False))

    # Bilateral co-missingness: for each L/R pair, does one side survive?
    bil = []
    for _, r in meta.iterrows():
        if pd.isna(r["mirror"]) or r["side"] not in ("L", "R"):
            continue
        a, b = na[r["variable"]], na[r["mirror"]]
        bil.append({
            "variable": r["variable"], "mirror": r["mirror"],
            "region": r["region"],
            "both_missing": int((a & b).sum()),
            "only_this_missing": int((a & ~b).sum()),
            "only_mirror_missing": int((~a & b).sum()),
        })
    bil_df = pd.DataFrame(bil)
    bil_df.to_csv(U.TAB / "bilateral_comissingness.csv", index=False)
    if len(bil_df):
        tot_pairs = bil_df[["both_missing", "only_this_missing"]].sum()
        print(f"\nBilateral pairs: {tot_pairs['both_missing']} cells missing on "
              f"BOTH sides vs {tot_pairs['only_this_missing']} cells where the "
              f"contralateral homologue survives.")

    # Mechanism tests.  Under MCAR the missingness indicator must be
    # independent of every OBSERVED quantity.  We therefore test each block
    # indicator against: acquisition order (Patient ID), age, sex, and all
    # 101 fully observed measurements.
    complete_cols = [c for c in meas if na[c].sum() == 0]
    tests = []
    for _, blk in patterns.iterrows():
        cols = blk["variables"].split(";")
        ind = na[cols[0]].values.astype(int)
        if ind.sum() < 2 or (1 - ind).sum() < 2:
            continue

        rho_id, p_id = stats.spearmanr(ind, df[U.ID_COL])
        rho_age, p_age = stats.spearmanr(ind, df["age_years"])
        tab = pd.crosstab(ind, df["gender"])
        p_sex = stats.fisher_exact(tab)[1] if tab.shape == (2, 2) else np.nan

        pvals = []
        for c in complete_cols:
            a = df[c][ind == 1]
            b = df[c][ind == 0]
            pvals.append(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
        pvals = np.array(pvals)
        q = bh_fdr(pvals)

        tests.append({
            "block_id": blk["block_id"],
            "driving_landmarks": blk["driving_landmarks"],
            "n_missing": int(ind.sum()),
            "spearman_rho_PatientID": round(rho_id, 3), "p_PatientID": round(p_id, 4),
            "spearman_rho_age": round(rho_age, 3), "p_age": round(p_age, 4),
            "p_sex_fisher": None if np.isnan(p_sex) else round(p_sex, 4),
            "n_complete_vars_tested": len(complete_cols),
            "n_raw_p<0.05": int((pvals < 0.05).sum()),
            "n_FDR_q<0.05": int((q < 0.05).sum()),
            "min_FDR_q": round(float(q.min()), 4),
        })
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(U.TAB / "mechanism_tests.csv", index=False)
    print("\n--- Mechanism tests (MCAR would predict no association) ---")
    print(tests_df.to_string(index=False))

    # ================================================================ FIGURES
    # fig01: variable-level missingness
    inc = by_var[by_var.n_missing > 0]
    fig, ax = plt.subplots(figsize=(7, 11))
    colors = sns.color_palette("crest", n_colors=len(inc.region.unique()))
    cmap = dict(zip(sorted(inc.region.unique()), colors))
    ax.barh(inc["variable"], inc["pct_missing"],
            color=[cmap[r] for r in inc["region"]])
    ax.invert_yaxis()
    ax.set_xlabel("% of subjects missing")
    ax.set_title(f"Missingness by variable ({len(inc)} of {n_var} affected;\n"
                 f"{n_var - len(inc)} variables fully observed)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[r]) for r in cmap]
    ax.legend(handles, list(cmap), title="region", fontsize=7, loc="lower right")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(); fig.savefig(U.FIG / "fig01_missing_by_variable.png"); plt.close(fig)

    # fig02: subject-level missingness
    fig, ax = plt.subplots(figsize=(8, 4))
    s = by_sub.sort_values(U.ID_COL)
    ax.bar(s[U.ID_COL].astype(str), s["pct_missing"],
           color=np.where(s["pct_missing"] > thresh, "#c0392b", "#4c72b0"))
    ax.axhline(thresh, ls="--", c="grey", lw=1,
               label=f"Tukey outlier threshold ({thresh:.1f}%)")
    ax.set_xlabel("Patient ID"); ax.set_ylabel("% of measurements missing")
    ax.set_title("Missingness by subject")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(U.FIG / "fig02_missing_by_subject.png"); plt.close(fig)

    # fig03: missingness matrix (subjects x variables), variables grouped
    ordered = (meta.sort_values(["region", "type", "variable"])["variable"].tolist())
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(na[ordered].astype(int), cbar=False, cmap=["#f0f0f0", "#b2182b"],
                linewidths=0.15, linecolor="white", ax=ax,
                yticklabels=df[U.ID_COL].astype(str))
    ax.set_xticks(np.arange(len(ordered)) + 0.5)
    ax.set_xticklabels(ordered, rotation=90, fontsize=4)
    ax.set_ylabel("Patient ID"); ax.set_xlabel("")
    ax.set_title("Missingness matrix (red = missing), variables ordered by anatomical region")
    fig.tight_layout(); fig.savefig(U.FIG / "fig03_missingness_matrix.png"); plt.close(fig)

    # fig04: co-missingness (Jaccard) among incomplete variables, clustered
    inc_cols = [c for c in meas if na[c].sum() > 0]
    A = na[inc_cols].values.astype(float)
    inter = A.T @ A
    cnt = A.sum(0)
    union = cnt[:, None] + cnt[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    D = 1 - J
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    idx = leaves_list(Z)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pd.DataFrame(J[np.ix_(idx, idx)],
                            index=[inc_cols[i] for i in idx],
                            columns=[inc_cols[i] for i in idx]),
                cmap="rocket_r", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Jaccard similarity of missingness"})
    ax.tick_params(labelsize=5)
    ax.set_title("Do variables go missing together?\n"
                 "Jaccard similarity of missingness indicators (hierarchically ordered)")
    fig.tight_layout(); fig.savefig(U.FIG / "fig04_comissingness.png"); plt.close(fig)

    # fig05: landmark availability matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(missing_lms.astype(int), cmap=["#f0f0f0", "#b2182b"], cbar=False,
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title("Landmark-level missingness\n(red = landmark not digitised for that subject)")
    ax.set_xlabel("landmark"); ax.set_ylabel("Patient ID")
    fig.tight_layout(); fig.savefig(U.FIG / "fig05_landmark_missingness.png"); plt.close(fig)

    # fig06: missingness vs acquisition order and age
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df[U.ID_COL], na.mean(axis=1) * 100, c="#b2182b")
    axes[0].set_xlabel("Patient ID (acquisition order)")
    axes[0].set_ylabel("% missing")
    r0 = stats.spearmanr(df[U.ID_COL], na.mean(axis=1))
    axes[0].set_title(f"vs acquisition order  (Spearman rho={r0.statistic:.2f}, p={r0.pvalue:.3f})")
    axes[1].scatter(df["age_years"], na.mean(axis=1) * 100, c="#2166ac")
    r1 = stats.spearmanr(df["age_years"], na.mean(axis=1))
    axes[1].set_xlabel("age (years)"); axes[1].set_ylabel("% missing")
    axes[1].set_title(f"vs age  (Spearman rho={r1.statistic:.2f}, p={r1.pvalue:.3f})")
    fig.suptitle("Subject-level missingness is explained by acquisition order, not by age")
    fig.tight_layout(); fig.savefig(U.FIG / "fig06_missing_vs_id_age.png"); plt.close(fig)

    # fig07: block indicators vs a size proxy (are missing subjects different?)
    size_proxy = df["S-N"]
    fig, axes = plt.subplots(1, len(patterns), figsize=(3 * len(patterns), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, blk) in zip(axes, patterns.iterrows()):
        ind = na[blk["variables"].split(";")[0]]
        sns.boxplot(x=ind.map({True: "missing", False: "observed"}),
                    y=size_proxy, ax=ax, width=0.5, palette="Set2")
        sns.stripplot(x=ind.map({True: "missing", False: "observed"}),
                      y=size_proxy, ax=ax, color="black", size=3, alpha=0.6)
        ax.set_title(f"{blk['block_id']}: {blk['driving_landmarks'][:18]}", fontsize=8)
        ax.set_xlabel("")
    axes[0].set_ylabel("S-N (cranial base length, mm)")
    fig.suptitle("Is missingness related to craniofacial size? (MCAR check)")
    fig.tight_layout(); fig.savefig(U.FIG / "fig07_block_vs_size.png"); plt.close(fig)

    print(f"\nWrote figures to {U.FIG} and tables to {U.TAB}")


if __name__ == "__main__":
    main()
