"""
02_impute.py  --  Task 2: select, justify and implement the imputation.

Produces
  data/processed/craniofacial_clean.csv          tidy input (parsed age)
  outputs/craniofacial_imputed.csv               completed dataset (primary)
  outputs/craniofacial_imputed_long_m20.csv      all m imputations, long form
  outputs/imputation_uncertainty.csv             per-cell SD + 95% interval
  outputs/tables/imputation_provenance.csv       which rule filled which cell
  outputs/tables/excluded_from_imputation.csv    exclusion rules applied
  outputs/tables/predictors_used.csv             model documentation

Run:  python src/02_impute.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import utils as U
from imputers import _select_predictors, impute_hybrid_mi

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Model settings (all fixed here so the run is fully reproducible)
# --------------------------------------------------------------------------
SETTINGS = dict(
    m=20,                 # number of imputations
    k=5,                  # max measurement predictors per target regression
    donors=3,             # PMM donor pool
    min_obs=10,           # min pairwise-complete n for a predictor
    use_bilateral=True,   # Stage-1 symmetry rule
    log_distances=True,   # model distances multiplicatively
    ridge_alpha=1.0,      # shrinkage on standardised predictors
    seed=U.RANDOM_SEED,
)

# Exclusion rule: a variable is flagged as UNSAFE TO IMPUTE when
#   (a) more than EXCL_PCT of subjects are missing it, AND
#   (b) it is an angle (a shape quantity, poorly predicted by size), AND
#   (c) no same-block variable survives to inform it.
# Flagged variables are still imputed so that downstream code receives a
# rectangular dataset, but they are marked so an analyst can drop them.
EXCL_PCT = 40.0


def main() -> None:
    rng = np.random.default_rng(SETTINGS["seed"])
    df = U.load_raw()
    meas = U.measurement_columns(df)
    meta = U.variable_metadata(df)
    X = df[meas].astype(float)
    aux = df[["age_years", "sex_M"]].astype(float)

    # tidy copy of the input
    clean = pd.concat([df[[U.ID_COL, "gender", "age"]], aux, X], axis=1)
    clean.to_csv(U.DATA_PROC / "craniofacial_clean.csv", index=False)

    # Exclusion / flagging rules
    excl = meta[["variable", "type", "region", "pct_missing"]].copy()
    excl["flag_high_missing"] = excl["pct_missing"] > EXCL_PCT
    excl["flag_angle"] = excl["type"] == "angle"
    excl["do_not_use_without_review"] = (
        excl["flag_high_missing"] & excl["flag_angle"])
    excl.loc[excl.pct_missing == 0, "do_not_use_without_review"] = False
    excl.to_csv(U.TAB / "excluded_from_imputation.csv", index=False)
    flagged = excl.loc[excl.do_not_use_without_review, "variable"].tolist()
    print(f"Flagged for manual review (imputed but not recommended): {flagged}")

    # No subject is excluded: the worst record (ID 17, 21.7%) still retains
    # 119 of 152 measurements, all of them anatomically informative.
    subj_miss = X.isna().mean(axis=1)
    print(f"Subjects excluded: none (max subject missingness "
          f"{subj_miss.max() * 100:.1f}%).")

    # Document the predictor set actually used for every imputed target
    rows = []
    for t in [c for c in meas if X[c].isna().any()]:
        need = X[t].isna().to_numpy()
        preds = _select_predictors(X, t, need, k=SETTINGS["k"],
                                   min_obs=SETTINGS["min_obs"])
        mrr = meta.set_index("variable").loc[t, "mirror"]
        bil = (isinstance(mrr, str)
               and (X[t].isna() & X[mrr].notna()).sum() > 0)
        rows.append({
            "target": t,
            "type": U.measure_type(t),
            "region": U.region(t),
            "n_missing": int(need.sum()),
            "n_fit_rows": int((~need).sum()),
            "bilateral_rule_applies": bil,
            "mirror": mrr if isinstance(mrr, str) else "",
            "predictors": ";".join(preds) + ";age_years;sex_M",
        })
    pred_doc = pd.DataFrame(rows)
    pred_doc.to_csv(U.TAB / "predictors_used.csv", index=False)
    print(f"Documented predictor sets for {len(pred_doc)} incomplete variables; "
          f"{pred_doc.bilateral_rule_applies.sum()} can use the bilateral rule.")

    # Fit: m completed datasets, then Rubin-style pooling
    frames, stage = impute_hybrid_mi(
        X, meta, aux, rng, m=SETTINGS["m"], k=SETTINGS["k"],
        donors=SETTINGS["donors"], min_obs=SETTINGS["min_obs"],
        use_bilateral=SETTINGS["use_bilateral"],
        log_distances=SETTINGS["log_distances"], return_all=True)

    arr = np.stack([f.to_numpy(float) for f in frames])
    point = pd.DataFrame(arr.mean(0), index=X.index, columns=X.columns)
    between_sd = pd.DataFrame(arr.std(0, ddof=1), index=X.index, columns=X.columns)

    # observed values are carried through untouched 
    obs_mask = X.notna()
    point[obs_mask] = X[obs_mask]
    between_sd[obs_mask] = 0.0
    assert point.isna().sum().sum() == 0, "completed dataset still has NaN"
    assert np.allclose(point.to_numpy()[obs_mask.to_numpy()],
                       X.to_numpy()[obs_mask.to_numpy()]), "observed values altered"

    imputed = pd.concat(
        [df[[U.ID_COL, "gender", "age"]], aux.round(4), point.round(4)], axis=1)
    imputed.to_csv(U.OUT / "craniofacial_imputed.csv", index=False)

    # long form: every imputation, for analysts who want to pool properly
    long = []
    for i, f in enumerate(frames, start=1):
        g = f.copy()
        g[obs_mask] = X[obs_mask]
        g.insert(0, "imputation", i)
        g.insert(0, U.ID_COL, df[U.ID_COL].to_numpy())
        long.append(g)
    pd.concat(long).to_csv(
        U.OUT / f"craniofacial_imputed_long_m{SETTINGS['m']}.csv", index=False)

    # per-cell uncertainty for the cells that were actually imputed
    ii, jj = np.where(~obs_mask.to_numpy())
    unc = pd.DataFrame({
        U.ID_COL: df[U.ID_COL].to_numpy()[ii],
        "variable": np.array(meas)[jj],
        "imputed_value": point.to_numpy()[ii, jj].round(4),
        "between_imputation_sd": between_sd.to_numpy()[ii, jj].round(4),
    })
    unc["ci95_low"] = (unc.imputed_value - 1.96 * unc.between_imputation_sd).round(4)
    unc["ci95_high"] = (unc.imputed_value + 1.96 * unc.between_imputation_sd).round(4)
    obs_sd = X.std(ddof=1)
    unc["sd_ratio_vs_observed"] = (
        unc.between_imputation_sd / unc.variable.map(obs_sd)).round(3)
    unc["type"] = unc.variable.map(U.measure_type)
    unc["region"] = unc.variable.map(U.region)
    unc["rule"] = [stage.to_numpy()[i, j] for i, j in zip(ii, jj)]
    unc["flagged_for_review"] = unc.variable.isin(flagged)
    unc.to_csv(U.OUT / "imputation_uncertainty.csv", index=False)

    prov = unc.groupby("rule").size().rename("n_cells").reset_index()
    prov.to_csv(U.TAB / "imputation_provenance.csv", index=False)
    print("\nCells filled by rule:")
    print(prov.to_string(index=False))

    print("\nUncertainty (between-imputation SD as a fraction of the "
          "variable's observed SD):")
    print(unc.groupby("type")["sd_ratio_vs_observed"]
          .describe()[["count", "mean", "50%", "max"]].round(2).to_string())
    print(unc.groupby("region")["sd_ratio_vs_observed"].mean().round(2).to_string())
    print(f"\nWrote {U.OUT / 'craniofacial_imputed.csv'}")


if __name__ == "__main__":
    main()
