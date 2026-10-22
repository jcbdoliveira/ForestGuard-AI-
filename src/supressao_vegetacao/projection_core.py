"""
Núcleo numérico único da projeção municipal (v17).

O mesmo contrato é reproduzido em src/projectionCore.mjs.
tests/test_v17_reliability.py exige igualdade entre os dois.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Remnant = Union[float, np.ndarray]


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Matriz de treino vazia.")
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    n_feat = X_aug.shape[1]
    reg = alpha * np.eye(n_feat)
    reg[0, 0] = 0.0
    A = X_aug.T @ X_aug + reg
    b = X_aug.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def ridge_predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    return X_aug @ weights


def build_lag_features_log(
    years: np.ndarray,
    values: np.ndarray,
    remnant: Remnant,
    n_lags: int = 3,
    anchor_year: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Features log1p. year_norm usa a âncora informada (último ano conhecido no treino)."""
    n = len(values)
    width = n_lags + 4
    if n <= n_lags:
        return np.zeros((0, width)), np.zeros(0), []

    years = np.asarray(years, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if np.isscalar(remnant):
        remnant_arr = np.full(n, float(remnant), dtype=np.float64)
    else:
        remnant_arr = np.asarray(remnant, dtype=np.float64)
        if len(remnant_arr) != n:
            raise ValueError("remnant precisa ter um valor por ano da série.")

    anchor = float(anchor_year if anchor_year is not None else years[-1])
    log_vals = np.log1p(values)
    rows: List[List[float]] = []
    targets: List[float] = []
    year_idx: List[int] = []

    for t in range(n_lags, n):
        lags = [float(log_vals[t - k]) for k in range(1, n_lags + 1)]
        window = log_vals[t - n_lags : t]
        ma = float(np.mean(window))
        x_local = np.arange(n_lags, dtype=np.float64)
        slope = float(np.polyfit(x_local, window, 1)[0]) if n_lags >= 2 else 0.0
        rel_rate = float(values[t - 1] / max(float(remnant_arr[t - 1]), 1.0))
        year_norm = (float(years[t]) - anchor) / 10.0
        rows.append(lags + [ma, slope, rel_rate, year_norm])
        targets.append(float(log_vals[t]))
        year_idx.append(int(years[t]))

    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), year_idx


def feature_names(n_lags: int) -> List[str]:
    return (
        [f"log1p_lag_{k}" for k in range(1, n_lags + 1)]
        + ["log1p_moving_avg", "log_local_slope", "rel_rate_to_remnant", "year_norm"]
    )


def project_series(
    hist: Dict[str, float],
    remnant_ha: float,
    weights: Sequence[float],
    n_lags: int,
    anchor_year: float,
    horizons: Sequence[int],
    pressure_multiplier: float = 1.0,
) -> Dict[str, object]:
    """Projeção autorregressiva. Espelho obrigatório de projectionCore.mjs."""
    years_list = sorted(int(y) for y in hist.keys())
    values = [float(hist[str(y)]) for y in years_list]
    if len(values) < n_lags:
        raise ValueError(f"Série com menos de {n_lags} anos.")

    w = np.asarray(weights, dtype=np.float64)
    hist_max = max(values)
    hist_mean = sum(values) / len(values)
    annual_cap = max(2.0 * hist_max, 3.0 * hist_mean)
    max_h = max(int(h) for h in horizons)
    series_ha = list(values)
    year_cursor = years_list[-1]
    annual: Dict[int, float] = {}
    p_mul = max(0.5, min(2.0, float(pressure_multiplier)))

    for _step in range(1, max_h + 1):
        year_cursor += 1
        log_series = np.log1p(np.asarray(series_ha, dtype=np.float64))
        lags = [float(log_series[-k]) for k in range(1, n_lags + 1)]
        window = log_series[-n_lags:]
        ma = float(np.mean(window))
        x_local = np.arange(n_lags, dtype=np.float64)
        slope = float(np.polyfit(x_local, window, 1)[0]) if n_lags >= 2 else 0.0
        rel_rate = float(series_ha[-1] / max(float(remnant_ha), 1.0))
        year_norm = (year_cursor - float(anchor_year)) / 10.0
        feat = np.asarray(lags + [ma, slope, rel_rate, year_norm], dtype=np.float64)
        pred_ha = float(np.expm1(float(ridge_predict(feat.reshape(1, -1), w)[0])))
        pred_ha = max(0.0, pred_ha) * p_mul
        pred_ha = min(pred_ha, annual_cap)
        series_ha.append(pred_ha)
        annual[year_cursor] = round(pred_ha, 2)

    base_year = years_list[-1]
    result: Dict[str, object] = {
        "base_year": base_year,
        "historical_series_ha": {str(y): round(v, 2) for y, v in zip(years_list, values)},
        "annual_projections_ha": {str(y): v for y, v in annual.items()},
        "annual_cap_ha": round(annual_cap, 2),
        "horizon_cumulative_ha": {},
    }
    horizons_out = result["horizon_cumulative_ha"]
    assert isinstance(horizons_out, dict)
    for h in horizons:
        h = int(h)
        cumulative = sum(annual[base_year + k] for k in range(1, h + 1))
        cumulative = min(cumulative, float(remnant_ha) * 0.95)
        horizons_out[f"{h}y"] = {
            "horizon_years": h,
            "target_year": base_year + h,
            "expected_suppressed_ha": round(cumulative, 2),
            "percent_of_remnant": round(100.0 * cumulative / max(float(remnant_ha), 1.0), 2),
        }
    return result
