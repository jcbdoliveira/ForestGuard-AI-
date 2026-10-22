"""
Treinamento da projeção municipal (v17).

Fonte: perda anual de floresta do MapBiomas Coleção 11
(`approx_annual_forest_loss_ha`), não as séries clonadas do catálogo.
O catálogo legado permanece só como metadado cadastral.

Validação: walk-forward com vários anos de corte. O último corte entra nas
métricas `metrics`; o retrospectivo completo fica em `walk_forward`.
Skill é 1 - RMSE_modelo / RMSE_persistência (repetir o ano anterior).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..projection_core import (
    build_lag_features_log,
    feature_names,
    project_series,
    ridge_fit,
    ridge_predict,
)
from ..reliability import metrics_ha, walk_forward_municipal
from ..sources.mapbiomas_catalog import INDEX_PATH, load_forest_index

# Reexporta o núcleo para testes e chamadas antigas.
__all__ = [
    "PanelSample",
    "TrainedModel",
    "build_lag_features_log",
    "build_panel",
    "get_trained_model",
    "project_municipality_series",
    "ridge_fit",
    "ridge_predict",
    "train_deforestation_model",
]


def index_sha256() -> str:
    if not INDEX_PATH.exists():
        return ""
    return hashlib.sha256(INDEX_PATH.read_bytes()).hexdigest()


def observed_series() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Séries observadas. Ignora o molde repetido de real_prodes_data.py."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for code, rec in load_forest_index().items():
        loss = rec.get("approx_annual_forest_loss_ha") or {}
        forest = rec.get("forest_ha_by_year") or {}
        if len(loss) < 5:
            continue
        years = sorted(int(y) for y in loss.keys())
        values = np.array([float(loss[str(y)]) for y in years], dtype=np.float64)
        remnant = np.array(
            [
                float(forest.get(str(y - 1), rec.get("forest_remnant_ha") or 1.0))
                for y in years
            ],
            dtype=np.float64,
        )
        out[str(code)] = (np.array(years, dtype=np.float64), values, remnant)
    return out


def _sorted_series(hist: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    years = sorted(int(y) for y in hist.keys())
    values = np.array([float(hist[str(y)]) for y in years], dtype=np.float64)
    return np.array(years, dtype=np.float64), values


@dataclass
class PanelSample:
    ibge_code: str
    year: int
    features: np.ndarray
    target_log: float
    target_ha: float
    remnant_ha: float


def build_panel(n_lags: int = 3, anchor_year: Optional[int] = None) -> List[PanelSample]:
    series = observed_series()
    if not series:
        return []
    if anchor_year is None:
        anchor_year = int(max(years[-1] for years, _v, _r in series.values()))
    samples: List[PanelSample] = []
    for code, (years, values, remnant) in series.items():
        X, y_log, year_list = build_lag_features_log(
            years, values, remnant, n_lags=n_lags, anchor_year=anchor_year
        )
        year_to_idx = {int(y): i for i, y in enumerate(years)}
        for i, yr in enumerate(year_list):
            src = year_to_idx[int(yr)]
            samples.append(
                PanelSample(
                    ibge_code=code,
                    year=int(yr),
                    features=X[i],
                    target_log=float(y_log[i]),
                    target_ha=float(values[src]),
                    remnant_ha=float(remnant[src]),
                )
            )
    return samples


@dataclass
class TrainedModel:
    weights: np.ndarray
    n_lags: int
    alpha: float
    feature_names: List[str]
    metrics: Dict[str, float]
    train_n: int
    validation_details: List[Dict[str, Any]] = field(default_factory=list)
    last_year_in_data: int = 2024
    model_type: str = "RidgeRegression_log1p_lags"
    anchor_year: int = 2024
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    city_reliability: Dict[str, Any] = field(default_factory=dict)
    data_source: str = "mapbiomas_col11_annual_forest_loss"
    index_sha256: str = ""


def train_deforestation_model(
    n_lags: int = 3,
    alpha: float = 1.0,
    holdout_last_years: int = 1,
) -> TrainedModel:
    series = observed_series()
    samples = build_panel(n_lags=n_lags)
    if len(samples) < 30:
        raise RuntimeError(f"Séries MapBiomas insuficientes para treinar ({len(samples)} amostras).")

    walk = walk_forward_municipal(series, n_lags=n_lags)
    years_sorted = sorted({s.year for s in samples})
    test_years = set(years_sorted[-holdout_last_years:])
    valid_years = (
        set(years_sorted[-holdout_last_years * 2 : -holdout_last_years])
        if len(years_sorted) > holdout_last_years
        else set()
    )
    anchor = int(max(years_sorted))

    def _stack(ss: List[PanelSample]) -> Tuple[np.ndarray, np.ndarray]:
        if not ss:
            return np.zeros((0, n_lags + 4)), np.zeros(0)
        return np.vstack([s.features for s in ss]), np.array([s.target_log for s in ss])

    # Features do painel final usam a âncora global. O walk-forward acima
    # refez as features com a âncora de cada corte (sem ver o futuro).
    train_s = [s for s in samples if s.year not in test_years and s.year not in valid_years]
    test_s = [s for s in samples if s.year in test_years]
    best_alpha = float(walk.get("overall", {}).get("alpha_last_cutoff") or alpha)

    X_tr, y_tr = _stack(train_s)
    if len(train_s) >= 3 and test_s:
        weights_eval = ridge_fit(X_tr, y_tr, alpha=best_alpha)
        pred_ha = np.maximum(np.expm1(ridge_predict(np.vstack([s.features for s in test_s]), weights_eval)), 0.0)
        y_te = np.array([s.target_ha for s in test_s])
        base = []
        for s in test_s:
            years, values, _rem = series[s.ibge_code]
            prev_mask = years.astype(int) == (s.year - 1)
            base.append(float(values[prev_mask][0]) if prev_mask.any() else float(s.target_ha))
        m = metrics_ha(y_te, pred_ha)
        m_base = metrics_ha(y_te, np.array(base))
        from ..reliability import skill_rmse

        metrics = {
            "mae_ha": m["mae_ha"],
            "rmse_ha": m["rmse_ha"],
            "bias_ha": m["bias_ha"],
            "rmse_persistence_ha": m_base["rmse_ha"],
            "skill_rmse": skill_rmse(m["rmse_ha"], m_base["rmse_ha"]),
            "n_eval": m["n"],
            "n_cutoffs": len(walk.get("cutoffs") or []),
        }
        details = []
        for s, p in list(zip(test_s, pred_ha))[:40]:
            details.append(
                {
                    "year": s.year,
                    "ibge_code": s.ibge_code,
                    "y_true_ha": round(s.target_ha, 2),
                    "y_pred_ha": round(float(p), 2),
                }
            )
    else:
        metrics = {"mae_ha": float("nan"), "rmse_ha": float("nan"), "n_eval": 0, "note": "holdout insuficiente"}
        details = []

    X_all, y_all = _stack(samples)
    final_weights = ridge_fit(X_all, y_all, alpha=best_alpha)

    return TrainedModel(
        weights=final_weights,
        n_lags=n_lags,
        alpha=best_alpha,
        feature_names=feature_names(n_lags),
        metrics=metrics,
        train_n=len(samples),
        validation_details=details,
        last_year_in_data=anchor,
        anchor_year=anchor,
        walk_forward={k: v for k, v in walk.items() if k != "by_city"},
        city_reliability=walk.get("by_city") or {},
        index_sha256=index_sha256(),
    )


def project_municipality_series(
    hist: Dict[str, float],
    remnant_ha: float,
    model: TrainedModel,
    horizons: List[int],
    pressure_multiplier: float = 1.0,
) -> Dict[str, Any]:
    return project_series(
        hist=hist,
        remnant_ha=remnant_ha,
        weights=model.weights.tolist(),
        n_lags=model.n_lags,
        anchor_year=model.anchor_year,
        horizons=horizons,
        pressure_multiplier=pressure_multiplier,
    )


_MODEL_CACHE: Optional[TrainedModel] = None


def get_trained_model(force_retrain: bool = False) -> TrainedModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is None or force_retrain:
        _MODEL_CACHE = train_deforestation_model()
    return _MODEL_CACHE
