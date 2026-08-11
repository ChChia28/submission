"""
imputers.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor

# Baselines
def impute_median(X: pd.DataFrame, meta=None, aux=None, rng=None) -> pd.DataFrame:
    """Unconditional median imputation -- the simple baseline."""
    out = SimpleImputer(strategy="median").fit_transform(X)
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def impute_knn(X: pd.DataFrame, meta=None, aux=None, rng=None, k: int = 5) -> pd.DataFrame:
    """KNN imputation on z-scored variables (so mm and degrees are comparable).

    Distances between subjects are computed with nan_euclidean, i.e. over the
    dimensions both subjects share, rescaled by the number of such dimensions.
    """
    mu, sd = X.mean(), X.std(ddof=1).replace(0, 1)
    Z = (X - mu) / sd
    out = KNNImputer(n_neighbors=k, weights="distance").fit_transform(Z)
    return pd.DataFrame(out, index=X.index, columns=X.columns) * sd + mu


def impute_mice_ridge(X: pd.DataFrame, meta=None, aux=None, rng=None,
                      max_iter: int = 10) -> pd.DataFrame:
    """sklearn IterativeImputer with a BayesianRidge -- a MICE-style engine.

    Included as the 'off the shelf MICE' comparator.  Predictors are capped at
    the 10 most correlated variables, without which the regressions would have
    151 predictors for 24 rows.
    """
    seed = int(rng.integers(1e6)) if rng is not None else 0
    mu, sd = X.mean(), X.std(ddof=1).replace(0, 1)
    Z = (X - mu) / sd
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,
                           n_nearest_features=10, sample_posterior=False,
                           random_state=seed, initial_strategy="median")
    out = imp.fit_transform(Z)
    return pd.DataFrame(out, index=X.index, columns=X.columns) * sd + mu


def impute_missforest(X: pd.DataFrame, meta=None, aux=None, rng=None,
                      max_iter: int = 6, n_trees: int = 100) -> pd.DataFrame:
    """Random-forest iterative imputation (a missForest analogue).

    Non-parametric, so it captures curvature in growth relationships, but it
    cannot extrapolate beyond the observed range of the donors.
    """
    seed = int(rng.integers(1e6)) if rng is not None else 0
    mu, sd = X.mean(), X.std(ddof=1).replace(0, 1)
    Z = (X - mu) / sd
    imp = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_trees, max_depth=6,
                                        min_samples_leaf=2, n_jobs=-1,
                                        random_state=seed),
        max_iter=max_iter, n_nearest_features=10, random_state=seed,
        initial_strategy="median")
    out = imp.fit_transform(Z)
    return pd.DataFrame(out, index=X.index, columns=X.columns) * sd + mu


# Primary method
def build_plan(X: pd.DataFrame, targets: list[str], k: int,
               min_obs: int) -> dict[str, list[str]]:
    """Choose predictors for every incomplete target in one vectorised pass.

    A candidate must be observed for **every** subject needing imputation
    (otherwise it is unusable at prediction time) and share at least
    `min_obs` pairwise-complete rows with the target.  Candidates are ranked
    by |Spearman rho|, because craniofacial growth relationships are monotone
    but not always linear.

    Spearman correlations and pairwise sample sizes are computed once for the
    whole matrix rather than per target; pandas' pairwise deletion means each
    rho already uses exactly the rows where both variables are observed.
    """
    cols = list(X.columns)
    rho = X.corr(method="spearman").to_numpy().copy()
    np.fill_diagonal(rho, np.nan)
    obs = X.notna().to_numpy()
    npair = obs.T.astype(int) @ obs.astype(int)      # pairwise-complete counts
    idx = {c: i for i, c in enumerate(cols)}

    plan: dict[str, list[str]] = {}
    for t in targets:
        j = idx[t]
        miss_rows = ~obs[:, j]
        # usable at prediction time: no NaN in the rows we must predict
        usable = ~(( ~obs[miss_rows] ).any(axis=0))
        score = np.abs(rho[j])
        score[~usable] = np.nan
        score[npair[j] < min_obs] = np.nan
        score[j] = np.nan
        order = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
        plan[t] = [cols[i] for i in order[:k] if np.isfinite(score[i])]
    return plan


def _select_predictors(X: pd.DataFrame, target: str, miss_rows: np.ndarray,
                       k: int, min_obs: int) -> list[str]:
    """Single-target wrapper around :func:`build_plan` (used for documentation
    output in 02_impute.py)."""
    return build_plan(X, [target], k=k, min_obs=min_obs)[target]


def _pmm_draw(y_obs: np.ndarray, yhat_obs: np.ndarray, yhat_miss: np.ndarray,
              rng: np.random.Generator, donors: int = 3) -> np.ndarray:
    """Predictive mean matching: return an observed value from the `donors`
    nearest predicted means.  Guarantees imputations are real, anatomically
    attainable measurements and preserves the marginal distribution's shape.
    """
    out = np.empty(len(yhat_miss))
    for i, yh in enumerate(yhat_miss):
        d = np.abs(yhat_obs - yh)
        pool = np.argsort(d)[:min(donors, len(d))]
        out[i] = y_obs[rng.choice(pool)]
    return out


def impute_hybrid_mi(X: pd.DataFrame, meta: pd.DataFrame, aux: pd.DataFrame,
                     rng: np.random.Generator, m: int = 1, k: int = 5,
                     donors: int = 3, min_obs: int = 10,
                     use_bilateral: bool = True, log_distances: bool = True,
                     boot_predictors: bool = True, return_all: bool = False):
    """Stage 1 bilateral rule + Stage 2 PMM multiple imputation.

    Parameters
    ----------
    m       : number of imputations (m=1 gives a single completed dataset)
    k       : max regression predictors drawn from the measurement pool
    donors  : PMM donor-pool size
    min_obs : minimum pairwise-complete n for a predictor to be eligible
    use_bilateral : apply the symmetry rule where the homologue is observed
    log_distances : model linear distances on the log scale (multiplicative
                    growth), angles on the raw scale
    return_all : if True return the list of m completed frames

    Returns
    -------
    (point_estimate, between_imputation_sd) or list of frames
    """
    meta_idx = meta.set_index("variable")
    is_dist = {v: meta_idx.loc[v, "type"] == "distance" for v in X.columns}
    mirror = {v: (meta_idx.loc[v, "mirror"]
                  if isinstance(meta_idx.loc[v, "mirror"], str) else None)
              for v in X.columns}

    targets = [c for c in X.columns if X[c].isna().any()]

    # Predictor selection depends only on the observed data and the
    # missingness pattern, not on the random draw, so it is computed ONCE and
    # reused across all m imputations.  This keeps the m draws comparable and
    # makes m=20 barely more expensive than m=1.
    plan_full = build_plan(X, targets, k=k, min_obs=min_obs)

    completed = []

    for draw in range(max(m, 1)):
        # Which variables are the "right" predictors is itself estimated from
        # 24 subjects.  Re-selecting predictors on a bootstrap resample for
        # each imputation propagates that model-selection uncertainty into the
        # spread of the m draws, which a fixed predictor set would ignore.
        if boot_predictors and m > 1 and draw > 0:
            bs_rows = rng.integers(0, len(X), len(X))
            Xb = X.iloc[bs_rows]
            try:
                plan = build_plan(Xb, targets, k=k, min_obs=min_obs)
                plan = {t: (v if v else plan_full[t]) for t, v in plan.items()}
            except Exception:
                plan = plan_full
        else:
            plan = plan_full

        F = X.copy()
        stage = pd.DataFrame("observed", index=X.index, columns=X.columns)
        stage[X.isna()] = "pending"

        # Stage 1: bilateral symmetry
        # For a missing left-side value whose right-side homologue is
        # observed, regress left on right using subjects where both exist.
        # A regression (rather than a straight copy) absorbs the systematic
        # left-right asymmetry that is normal in craniofacial anatomy.
        if use_bilateral:
            for t in targets:
                mrr = mirror[t]
                if mrr is None or mrr not in X.columns:
                    continue
                need = X[t].isna().to_numpy() & X[mrr].notna().to_numpy()
                if not need.any():
                    continue
                fit = X[t].notna().to_numpy() & X[mrr].notna().to_numpy()
                if fit.sum() < min_obs:
                    continue
                xa, ya = X.loc[fit, mrr].to_numpy(), X.loc[fit, t].to_numpy()
                # bootstrap the fitting rows -> parameter uncertainty
                bs = rng.integers(0, len(xa), len(xa))
                b1, b0 = np.polyfit(xa[bs], ya[bs], 1)
                resid = ya - (b0 + b1 * xa)
                sigma = resid.std(ddof=2) if len(resid) > 2 else resid.std()
                pred = b0 + b1 * X.loc[need, mrr].to_numpy()
                if m > 1:                       # add residual noise for MI
                    pred = pred + rng.normal(0, sigma, len(pred))
                F.loc[need, t] = pred
                stage.loc[need, t] = "bilateral"

        #  Stage 2: PMM multiple imputation 
        for t in targets:
            need = F[t].isna().to_numpy()
            if not need.any():
                continue
            preds = plan.get(t, [])
            fit_rows = X[t].notna().to_numpy()
            if not preds or fit_rows.sum() < 5:
                F.loc[need, t] = X[t].median()
                stage.loc[need, t] = "median_fallback"
                continue

            P = pd.concat([X[preds], aux], axis=1)
            # any residual NaN among predictors -> column median (rare)
            P = P.fillna(P.median())

            y = X[t].to_numpy(float)
            transform = is_dist[t] and log_distances and np.nanmin(y) > 0
            yt = np.log(y) if transform else y

            A = P.to_numpy(float)
            mu, sd = A[fit_rows].mean(0), A[fit_rows].std(0, ddof=1)
            sd[sd == 0] = 1.0
            A = (A - mu) / sd

            # Bootstrap the complete cases -> proper parameter uncertainty
            idx = np.flatnonzero(fit_rows)
            bs = rng.choice(idx, len(idx), replace=True) if m > 1 else idx
            model = Ridge(alpha=1.0).fit(A[bs], yt[bs])

            yhat_obs = model.predict(A[fit_rows])
            yhat_miss = model.predict(A[need])
            drawn = _pmm_draw(yt[fit_rows], yhat_obs, yhat_miss, rng, donors)
            F.loc[need, t] = np.exp(drawn) if transform else drawn
            stage.loc[need, t] = "pmm"

        completed.append(F)

    if return_all:
        return completed, stage

    arr = np.stack([f.to_numpy(float) for f in completed])
    point = pd.DataFrame(arr.mean(0), index=X.index, columns=X.columns)
    sd = pd.DataFrame(arr.std(0, ddof=1) if len(completed) > 1
                      else np.zeros_like(arr[0]), index=X.index, columns=X.columns)
    point[X.notna()] = X[X.notna()]      # observed values are never altered
    return point, sd


# Registry used by the validation script 
SINGLE_METHODS = {
    "median": impute_median,
    "knn_k3": lambda X, meta, aux, rng: impute_knn(X, k=3),
    "knn_k5": lambda X, meta, aux, rng: impute_knn(X, k=5),
    "mice_ridge": impute_mice_ridge,
    "missforest": impute_missforest,
    "hybrid_pmm": lambda X, meta, aux, rng: impute_hybrid_mi(
        X, meta, aux, rng, m=1)[0],
    "hybrid_pmm_m20": lambda X, meta, aux, rng: impute_hybrid_mi(
        X, meta, aux, rng, m=20)[0],
    "hybrid_pmm_nolog": lambda X, meta, aux, rng: impute_hybrid_mi(
        X, meta, aux, rng, m=1, log_distances=False)[0],
}
