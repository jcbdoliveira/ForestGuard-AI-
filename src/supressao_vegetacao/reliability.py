"""
Validação retrospectiva (v17).

Walk-forward por ano de corte, skill contra persistência (repetir o ano anterior)
e regra explícita de quando a previsão municipal não deve ser usada.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .projection_core import build_lag_features_log, ridge_fit, ridge_predict


def metrics_ha(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {"mae_ha": float("nan"), "rmse_ha": float("nan"), "bias_ha": float("nan"), "n": 0}
    err = y_pred - y_true
    return {
        "mae_ha": round(float(np.mean(np.abs(err))), 2),
        "rmse_ha": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "bias_ha": round(float(np.mean(err)), 2),
        "n": int(len(y_true)),
    }


def skill_rmse(rmse_model: float, rmse_baseline: float) -> float:
    if rmse_baseline is None or not np.isfinite(rmse_baseline) or rmse_baseline <= 1e-9:
        return float("nan")
    return round(1.0 - float(rmse_model) / float(rmse_baseline), 4)


def judge_reliability(
    skill: float,
    n_eval: int,
    n_cutoffs: int,
    min_cutoffs: int = 5,
) -> Tuple[bool, str]:
    if n_cutoffs < min_cutoffs:
        return False, "cortes_insuficientes"
    if n_eval < max(10, min_cutoffs):
        return False, "amostra_insuficiente"
    if not np.isfinite(skill):
        return False, "baseline_degenerada"
    if skill <= 0:
        return False, "skill_rmse_nao_supera_persistencia"
    if skill < 0.10:
        return False, "skill_positivo_mas_marginal"
    return True, "skill_superior_a_persistencia_com_margem"


def _stack_for_years(
    series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    years_keep: set,
    n_lags: int,
    anchor: float,
) -> List[Tuple[str, int, np.ndarray, float, float]]:
    """Lista (ibge, year, features, y_ha, persistence_ha) restrita a years_keep."""
    rows = []
    for code, (years, values, remnant) in series.items():
        year_to_val = {int(y): float(v) for y, v in zip(years, values)}
        X, _y_log, year_list = build_lag_features_log(years, values, remnant, n_lags=n_lags, anchor_year=anchor)
        for i, yr in enumerate(year_list):
            if yr not in years_keep:
                continue
            prev = year_to_val.get(yr - 1)
            if prev is None:
                continue
            t_idx = int(np.where(years.astype(int) == yr)[0][0])
            rows.append((code, yr, X[i], float(values[t_idx]), prev))
    return rows


def walk_forward_municipal(
    series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_lags: int = 3,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 50.0),
) -> Dict[str, Any]:
    """
    Para cada ano de corte T (com validação em T-1 e treino estritamente anterior):
    ajusta o alpha em T-1, mede o erro em T e compara com a persistência.
    """
    all_years = sorted({int(y) for years, _v, _r in series.values() for y in years})
    if len(all_years) < n_lags + 3:
        return {"cutoffs": [], "by_cutoff": [], "overall": {}, "by_city": {}}

    min_y = all_years[0]
    cutoffs = [y for y in all_years if (y - 1) in all_years and (y - 1) >= min_y + n_lags + 1]

    by_city_true: Dict[str, List[float]] = {}
    by_city_pred: Dict[str, List[float]] = {}
    by_city_base: Dict[str, List[float]] = {}
    by_city_mean: Dict[str, List[float]] = {}
    by_cutoff = []
    all_true: List[float] = []
    all_pred: List[float] = []
    all_base: List[float] = []
    all_mean: List[float] = []
    chosen_alpha = float(alphas[0])

    for T in cutoffs:
        anchor = float(T - 2)
        train_rows = _stack_for_years(series, {y for y in all_years if y <= T - 2}, n_lags, anchor)
        valid_rows = _stack_for_years(series, {T - 1}, n_lags, anchor)
        test_rows = _stack_for_years(series, {T}, n_lags, anchor)
        if len(train_rows) < 8 or len(valid_rows) < 3 or len(test_rows) < 3:
            continue

        X_tr = np.vstack([r[2] for r in train_rows])
        y_tr = np.log1p(np.array([r[3] for r in train_rows]))
        best_a = float(alphas[0])
        best_rmse = float("inf")
        y_va = np.array([r[3] for r in valid_rows])
        X_va = np.vstack([r[2] for r in valid_rows])
        for a in alphas:
            w = ridge_fit(X_tr, y_tr, alpha=float(a))
            pred = np.maximum(np.expm1(ridge_predict(X_va, w)), 0.0)
            rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_a = float(a)
        chosen_alpha = best_a
        w = ridge_fit(X_tr, y_tr, alpha=best_a)
        X_te = np.vstack([r[2] for r in test_rows])
        pred = np.maximum(np.expm1(ridge_predict(X_te, w)), 0.0)
        yt = np.array([r[3] for r in test_rows], dtype=np.float64)
        yb = np.array([r[4] for r in test_rows], dtype=np.float64)
        # Baseline de média histórica por município: usa apenas anos anteriores ao corte.
        mean_preds = []
        for row in test_rows:
            code = row[0]
            years_city, values_city, _ = series[code]
            past = values_city[years_city < T]
            mean_preds.append(float(np.mean(past)) if len(past) else float(row[4]))
        yh = np.asarray(mean_preds, dtype=np.float64)
        m_model = metrics_ha(yt, pred)
        m_base = metrics_ha(yt, yb)
        by_cutoff.append(
            {
                "year": int(T),
                "alpha": best_a,
                "n": m_model["n"],
                "mae_ha": m_model["mae_ha"],
                "rmse_ha": m_model["rmse_ha"],
                "bias_ha": m_model["bias_ha"],
                "rmse_persistence_ha": m_base["rmse_ha"],
                "mae_persistence_ha": m_base["mae_ha"],
                "rmse_historical_mean_ha": metrics_ha(yt, yh)["rmse_ha"],
                "mae_historical_mean_ha": metrics_ha(yt, yh)["mae_ha"],
                "skill_rmse": skill_rmse(m_model["rmse_ha"], m_base["rmse_ha"]),
            }
        )
        for row, p, mean_p in zip(test_rows, pred, yh):
            code = row[0]
            by_city_true.setdefault(code, []).append(row[3])
            by_city_pred.setdefault(code, []).append(float(p))
            by_city_base.setdefault(code, []).append(row[4])
            by_city_mean.setdefault(code, []).append(float(mean_p))
        all_true.extend(yt.tolist())
        all_pred.extend(pred.tolist())
        all_base.extend(yb.tolist())
        all_mean.extend(yh.tolist())

    overall_m = metrics_ha(np.array(all_true), np.array(all_pred))
    overall_b = metrics_ha(np.array(all_true), np.array(all_base))
    overall_h = metrics_ha(np.array(all_true), np.array(all_mean))
    best_overall_baseline_rmse = min(overall_b["rmse_ha"], overall_h["rmse_ha"])
    overall_skill = skill_rmse(overall_m["rmse_ha"], best_overall_baseline_rmse)
    overall_ok, overall_reason = judge_reliability(overall_skill, overall_m["n"], len(by_cutoff))
    overall = {
        **overall_m,
        "rmse_persistence_ha": overall_b["rmse_ha"],
        "mae_persistence_ha": overall_b["mae_ha"],
        "rmse_historical_mean_ha": overall_h["rmse_ha"],
        "best_baseline_rmse_ha": best_overall_baseline_rmse,
        "mae_historical_mean_ha": overall_h["mae_ha"],
        "skill_rmse": overall_skill,
        "n_cutoffs": len(by_cutoff),
        "reliable": overall_ok,
        "reason": overall_reason,
        "alpha_last_cutoff": chosen_alpha,
    }

    by_city: Dict[str, Any] = {}
    for code in by_city_true:
        yt = np.array(by_city_true[code])
        yp = np.array(by_city_pred[code])
        yb = np.array(by_city_base[code])
        yh = np.array(by_city_mean[code])
        mh = metrics_ha(yt, yh)
        mm = metrics_ha(yt, yp)
        mb = metrics_ha(yt, yb)
        best_baseline_rmse = min(mb["rmse_ha"], mh["rmse_ha"])
        sk = skill_rmse(mm["rmse_ha"], best_baseline_rmse)
        ok, reason = judge_reliability(sk, mm["n"], mm["n"])
        by_city[code] = {
            "ibge_code": code,
            **mm,
            "n_cutoffs": mm["n"],
            "rmse_persistence_ha": mb["rmse_ha"],
            "mae_persistence_ha": mb["mae_ha"],
            "rmse_historical_mean_ha": mh["rmse_ha"],
            "mae_historical_mean_ha": mh["mae_ha"],
            "best_baseline_rmse_ha": best_baseline_rmse,
            "skill_rmse": sk,
            "reliable": ok,
            "reason": reason,
            "cutoffs": [c["year"] for c in by_cutoff],
        }

    return {
        "cutoffs": [c["year"] for c in by_cutoff],
        "by_cutoff": by_cutoff,
        "overall": overall,
        "by_city": by_city,
        "baseline": "persistencia_ano_anterior",
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return round(float(ece), 6)


def fit_platt(scores: np.ndarray, y: np.ndarray, min_positives: int = 15) -> Optional[Dict[str, Any]]:
    """Regressão logística 1D (Platt). None se não houver eventos suficientes."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos = int(y.sum())
    if pos < min_positives or pos > len(y) - min_positives:
        return None
    x = np.column_stack([np.ones(len(scores)), scores])
    w = np.zeros(2, dtype=np.float64)
    for _ in range(30):
        z = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.maximum(p * (1.0 - p), 1e-6)
        z_adj = z + (y - p) / W
        Xw = x * W[:, None]
        A = x.T @ Xw
        b = x.T @ (W * z_adj)
        A[0, 0] += 1e-4
        A[1, 1] += 1e-4
        w_new = np.linalg.solve(A, b)
        if float(np.max(np.abs(w_new - w))) < 1e-8:
            w = w_new
            break
        w = w_new
    return {"method": "platt", "a": round(float(w[0]), 6), "b": round(float(w[1]), 6), "n": int(len(y)), "positives": pos}


def apply_platt(scores: np.ndarray, calibrator: Dict[str, Any]) -> np.ndarray:
    z = calibrator["a"] + calibrator["b"] * np.asarray(scores, dtype=np.float64)
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def map_probability_policy(features_imputed: bool, calibrator: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A AUC do treino não descreve o mapa quando features foram imputadas."""
    calibrated = bool(calibrator) and calibrator.get("method") == "platt" and not features_imputed
    if features_imputed:
        source = "spatial_raw_with_imputed_features"
    elif calibrated:
        source = "spatial_platt_on_validation"
    else:
        source = "spatial_raw_proba_uncalibrated"
    return {
        "metrics_apply_to_displayed_map": not features_imputed,
        "calibrated": calibrated,
        "probability_source": source,
        "reason": (
            "Features de raster ausentes foram imputadas com a mediana do treino; "
            "ROC-AUC/Brier do painel GEE não descrevem este mapa."
            if features_imputed
            else (
                "Platt ajustado no ano de validação e aplicado à probabilidade bruta."
                if calibrated
                else "Sem calibrador (eventos insuficientes no ano de validação) ou modelo geométrico."
            )
        ),
    }
