"""
Treino espacial com XGBoost (preferencial) ou GradientBoosting do scikit-learn.

- Split temporal (não aleatório)
- Métricas REAIS no conjunto de teste: ROC-AUC, PR-AUC, Brier, recall@top10%
- Persistência do modelo + metadata JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .panel import Panel, load_panel_csv, temporal_split, augment_with_spatial_neighbors

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# sklearn importado sob demanda em _fit_sklearn()


@dataclass
class SpatialTrainResult:
    model_type: str
    feature_names: List[str]
    target_col: str
    metrics_test: Dict[str, Any]
    metrics_valid: Dict[str, Any] = field(default_factory=dict)
    train_years: List[int] = field(default_factory=list)
    valid_year: Optional[int] = None
    test_year: Optional[int] = None
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    positive_rate_test: float = 0.0
    feature_importance: List[Dict[str, float]] = field(default_factory=list)
    model_path: Optional[str] = None
    # Medianas do treino — usadas na inferência quando a célula não traz o raster
    feature_medians: Dict[str, float] = field(default_factory=dict)
    horizon_years: int = 0
    train_cutoff_year: Optional[int] = None
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    training_note: str = (
        "Artefato de treinamento. Importâncias refletem o painel usado no treino "
        "(pode ser sintético de desenvolvimento). Não são pesos causais."
    )


def _round_metrics(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_metrics(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return round(obj, 6)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return round(v, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """ECE (Expected Calibration Error) com bins de probabilidade iguais."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({
                "bin_low": float(lo), "bin_high": float(hi),
                "count": 0, "avg_confidence": None, "avg_frequency": None, "gap": None,
            })
            continue
        conf = float(y_prob[mask].mean())
        freq = float(y_true[mask].mean())
        gap = abs(conf - freq)
        ece += (cnt / n) * gap
        rows.append({
            "bin_low": float(lo), "bin_high": float(hi),
            "count": cnt, "avg_confidence": conf, "avg_frequency": freq, "gap": gap,
        })
    return {"ece": float(ece), "n_bins": n_bins, "bins": rows}


def optimize_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    n_grid: int = 91,
) -> Dict[str, Any]:
    """Escolhe limiar minimizando custo esperado: cost_fp*FP + cost_fn*FN.

    Defaults: omitir um desmate (FN) custa 5× mais que um falso alarme (FP).
    Ajuste cost_fp/cost_fn à operação de fiscalização.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.linspace(min_threshold, max_threshold, n_grid)
    best_cost = None
    best_f1 = -1.0
    best_cost_thr = float(grid[0])
    best_f1_thr = float(grid[0])
    curve = []
    for thr in grid:
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        cost = float(cost_fp) * float(fp) + float(cost_fn) * float(fn)
        f1 = float(f1_score(y_true, y_hat, zero_division=0))
        prec = float(precision_score(y_true, y_hat, zero_division=0))
        rec = float(recall_score(y_true, y_hat, zero_division=0))
        curve.append({
            "threshold": float(thr),
            "cost": cost,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        })
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cost_thr = float(thr)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = float(thr)

    def _at(thr: float) -> Dict[str, Any]:
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        return {
            "threshold": float(thr),
            "cost": float(cost_fp) * float(fp) + float(cost_fn) * float(fn),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        }

    # subsample curve for JSON size (keep ~20 points + optima)
    step = max(1, len(curve) // 20)
    slim = curve[::step]
    return {
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "cost_ratio_fn_over_fp": float(cost_fn / cost_fp) if cost_fp else None,
        "best_cost": _at(best_cost_thr),
        "best_f1": _at(best_f1_thr),
        "at_0_5": _at(0.5),
        "threshold_curve_sample": slim,
        "note": (
            "Custo = cost_fp*FP + cost_fn*FN. "
            "Ajuste os custos à operação; defaults FN:FP = 5:1."
        ),
    }


def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    include_curves: bool = True,
) -> Dict[str, Any]:
    """Métricas de classificação binária + R² + PR curve + ECE + custo esperado."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
        log_loss,
        r2_score,
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_curve,
    )
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
    }
    nan_keys = [
        "roc_auc", "pr_auc", "brier", "log_loss", "r2", "mae", "rmse",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "specificity", "recall_at_top10pct",
        "tn", "fp", "fn", "tp",
    ]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        out.update({k: float("nan") for k in nan_keys})
        out["ece"] = None
        out["pr_curve"] = None
        out["threshold_optimization"] = None
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["log_loss"] = float(log_loss(y_true, y_prob))
    out["r2"] = float(r2_score(y_true, y_prob))
    out["mae"] = float(mean_absolute_error(y_true, y_prob))
    out["rmse"] = float(mean_squared_error(y_true, y_prob) ** 0.5)

    y_hat = (y_prob >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, y_hat))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_hat))
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tp"] = float(tp)
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    k = max(1, int(0.10 * len(y_prob)))
    top_idx = np.argsort(y_prob)[::-1][:k]
    out["recall_at_top10pct"] = float(y_true[top_idx].sum() / max(y_true.sum(), 1))

    # ECE
    ece_pack = expected_calibration_error(y_true, y_prob, n_bins=10)
    out["ece"] = float(ece_pack["ece"])
    out["calibration"] = ece_pack

    # PR curve (subsample)
    if include_curves:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        # thr tem len = len(prec)-1
        pts = []
        idxs = np.linspace(0, len(prec) - 1, num=min(40, len(prec)), dtype=int)
        for i in idxs:
            pts.append({
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "threshold": float(thr[i]) if i < len(thr) else None,
            })
        out["pr_curve"] = pts

    # Threshold optimization (F1 + custo esperado)
    out["threshold_optimization"] = optimize_threshold_cost(
        y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn
    )
    # atalhos úteis no topo
    opt = out["threshold_optimization"]
    out["threshold_best_f1"] = opt["best_f1"]["threshold"]
    out["threshold_best_cost"] = opt["best_cost"]["threshold"]
    out["expected_cost_at_best"] = opt["best_cost"]["cost"]
    out["expected_cost_at_0_5"] = opt["at_0_5"]["cost"]

    return _round_metrics(out)


def _fit_xgboost(
    X_tr, y_tr, X_va, y_va, seed: int = 42
) -> Tuple[Any, np.ndarray]:
    pos = max(float(y_tr.sum()), 1.0)
    neg = max(float(len(y_tr) - y_tr.sum()), 1.0)
    spw = neg / pos
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=2,
        early_stopping_rounds=40 if X_va is not None and len(y_va) else None,
    )
    if X_va is not None and len(y_va) > 0:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_tr)[:, 1]


def _fit_sklearn_gbm(X_tr, y_tr, seed: int = 42):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model


def train_spatial_model(
    panel: Panel,
    target_col: str = "y_loss_1y",
    test_year: Optional[int] = None,
    valid_year: Optional[int] = None,
    model_dir: Optional[str | Path] = None,
    prefer_xgboost: bool = True,
    seed: int = 42,
    augment_neighbors: bool = True,
    run_walk_forward: bool = True,
) -> Tuple[Any, SpatialTrainResult]:
    # v19: vizinhos só com rótulo já fechado antes de t0. Não usa y do mesmo ano,
    # de ano futuro, nem o rótulo da própria célula.
    horizon_years = 3 if str(target_col).endswith("3y") else (1 if str(target_col).endswith("1y") else 0)
    if augment_neighbors:
        # Embargo conservador alinhado ao temporal_split (t0 <= year - H - 1)
        panel = augment_with_spatial_neighbors(panel, horizon_years=horizon_years)
    split = temporal_split(panel, test_year=test_year, valid_year=valid_year, horizon_years=horizon_years)
    X_tr, y_tr = split["train"]
    if not HAS_XGB and np.isnan(X_tr).any():
        raise RuntimeError("Ha NaN em features opcionais; instale xgboost (sem imputacao arbitraria).")
    X_te, y_te = split["test"]
    years_sorted, v_year, t_year = split["years"]
    X_va = y_va = None
    if "valid" in split:
        X_va, y_va = split["valid"]

    use_xgb = prefer_xgboost and HAS_XGB
    if use_xgb:
        model, _ = _fit_xgboost(X_tr, y_tr, X_va, y_va, seed=seed)
        model_type = "XGBoostClassifier"
        proba_te = model.predict_proba(X_te)[:, 1]
        proba_va = model.predict_proba(X_va)[:, 1] if X_va is not None else None
        importances = model.feature_importances_
    else:
        model = _fit_sklearn_gbm(X_tr, y_tr, seed=seed)
        model_type = "sklearn.GradientBoostingClassifier"
        proba_te = model.predict_proba(X_te)[:, 1]
        proba_va = model.predict_proba(X_va)[:, 1] if X_va is not None else None
        importances = model.feature_importances_

    metrics_test = evaluate_binary(y_te, proba_te)
    metrics_valid = evaluate_binary(y_va, proba_va) if proba_va is not None else {}

    fi = [
        {"feature": n, "importance": round(float(v), 6)}
        for n, v in sorted(
            zip(panel.feature_names, importances),
            key=lambda kv: -kv[1],
        )
    ]

    # v22: sem medianas para imputação. Ausência é preservada como NaN.
    medians: Dict[str, float] = {}

    result = SpatialTrainResult(
        model_type=model_type,
        feature_names=list(panel.feature_names),
        target_col=target_col,
        metrics_test=metrics_test,
        metrics_valid=metrics_valid,
        train_years=list(split.get("train_years", [])),
        valid_year=v_year,
        test_year=t_year,
        n_train=int(len(y_tr)),
        n_valid=int(len(y_va)) if y_va is not None else 0,
        n_test=int(len(y_te)),
        positive_rate_test=float(y_te.mean()) if len(y_te) else 0.0,
        feature_importance=fi,
        feature_medians=medians,
        horizon_years=horizon_years,
        train_cutoff_year=split.get("train_cutoff_year"),
    )
    if run_walk_forward:
        result.walk_forward = walk_forward_spatial(panel, seed=seed, horizon_years=horizon_years)

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # 1) salva o modelo primeiro e define model_path
        if use_xgb:
            model_path = model_dir / "spatial_model.json"
            model.save_model(str(model_path))
            result.model_path = str(model_path)
        else:
            import pickle
            model_path = model_dir / "spatial_model.pkl"
            with model_path.open("wb") as f:
                pickle.dump({"model": model, "feature_names": panel.feature_names}, f)
            result.model_path = str(model_path)
        # 2) grava metadata com model_path e identificação do alvo/município
        meta_path = model_dir / "spatial_model_meta.json"
        payload = asdict(result)
        payload["data_source"] = "gee_mapbiomas_panel"
        payload["project_version"] = "v25.0.1"
        # IBGE a partir do painel (maioria das linhas) ou do nome do diretório
        ibge_from_meta = None
        try:
            codes = [str(m.get("ibge_code") or "").strip() for m in panel.meta if m.get("ibge_code")]
            if codes:
                ibge_from_meta = max(set(codes), key=codes.count)
        except Exception:
            ibge_from_meta = None
        if not ibge_from_meta:
            stem = model_dir.name.replace("_annual", "").strip()
            if stem.isdigit() and len(stem) == 7:
                ibge_from_meta = stem
        if ibge_from_meta:
            payload["ibge_code"] = ibge_from_meta
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return model, result


def train_from_csv(
    csv_path: str | Path,
    target_col: str = "y_loss_1y",
    model_dir: str | Path = "data/spatial/models",
    **kwargs,
) -> SpatialTrainResult:
    panel = load_panel_csv(csv_path, target_col=target_col)
    _, result = train_spatial_model(panel, target_col=target_col, model_dir=model_dir, **kwargs)
    return result


def walk_forward_spatial(panel: Panel, seed: int = 42, horizon_years: int = 1) -> Dict[str, Any]:
    """
    Walk-forward por ano, com embargo de rótulo e validação temporal real (v24).

    Para cada ano de teste T (horizonte H):
      - treino:  t0 <= T - H - 2  (ou o máximo disponível abaixo da validação)
      - validação: t0 == T - H - 1  (quando existir e tiver amostras suficientes)
      - teste:   t0 == T

    Comparações no **mesmo subconjunto** de células de teste (interseção com
    distância rodoviária válida quando a baseline de estradas é avaliada):
      1. Modelo completo
      2. Modelo sem features rodoviárias
      3. Baseline de prevalência histórica (treino)
      4. Baseline de distância às estradas

    Calibração Platt só usa o conjunto de validação temporal (nunca o teste).
    """
    from ..reliability import apply_platt, expected_calibration_error, fit_platt

    years = sorted({int(y) for y in panel.years})
    if len(years) < 3:
        return {"cutoffs": [], "by_cutoff": [], "note": "painel com menos de 3 anos"}

    road_names = [
        n for n in (
            "road_dist_major_km", "road_dist_secondary_km",
            "road_dist_local_km", "road_dist_track_km", "road_dist_km",
        ) if n in panel.feature_names
    ]
    # Preferir as 4 categorias; se só existir road_dist_km, usa essa.
    if any(n.startswith("road_dist_") and n != "road_dist_km" for n in road_names):
        road_names = [n for n in road_names if n != "road_dist_km"]
    road_indices = [panel.feature_names.index(n) for n in road_names]
    non_road_indices = [i for i, n in enumerate(panel.feature_names) if n not in set(road_names)]

    by_cutoff: List[Dict[str, Any]] = []
    last_calibrator = None
    H = int(horizon_years)

    for test_year in years[1:]:
        # Embargo: último t0 cujo rótulo fecha antes de T é T - H - 1.
        embargo_cutoff = int(test_year) - H - 1
        eligible_train_years = [y for y in years if y <= embargo_cutoff]
        if not eligible_train_years:
            continue

        # Validação temporal real: o ano mais recente ainda elegível.
        # Treino fica estritamente antes da validação.
        valid_year = eligible_train_years[-1]
        train_years = [y for y in eligible_train_years if y < valid_year]
        # Se não houver ano anterior, não há validação separada; treina em tudo até embargo.
        if not train_years:
            train_years = list(eligible_train_years)
            valid_year = None

        train_m = np.isin(panel.years, train_years)
        valid_m = (panel.years == int(valid_year)) if valid_year is not None else np.zeros(len(panel.years), dtype=bool)
        test_m = panel.years == int(test_year)

        if int(train_m.sum()) < 20 or int(test_m.sum()) < 10 or len(np.unique(panel.y[train_m])) < 2:
            continue

        X_tr, y_tr = panel.X[train_m], panel.y[train_m]
        X_te, y_te = panel.X[test_m], panel.y[test_m]
        X_va = panel.X[valid_m] if valid_m.any() else None
        y_va = panel.y[valid_m] if valid_m.any() else None

        # --- Modelo completo ---
        if HAS_XGB:
            model, _ = _fit_xgboost(X_tr, y_tr, X_va, y_va, seed=seed)
        else:
            model = _fit_sklearn_gbm(X_tr, y_tr, seed=seed)
        proba_te = model.predict_proba(X_te)[:, 1]
        proba_va = (
            model.predict_proba(X_va)[:, 1]
            if X_va is not None and y_va is not None and len(y_va)
            else None
        )

        # --- Modelo sem features rodoviárias (mesmos cortes) ---
        metrics_no_road: Dict[str, float] = {}
        proba_te_nr = None
        if non_road_indices and len(non_road_indices) < len(panel.feature_names):
            X_tr_nr = X_tr[:, non_road_indices]
            X_te_nr = X_te[:, non_road_indices]
            X_va_nr = X_va[:, non_road_indices] if X_va is not None else None
            model_nr = None
            if HAS_XGB:
                model_nr, _ = _fit_xgboost(X_tr_nr, y_tr, X_va_nr, y_va, seed=seed)
            else:
                tr_ok = np.isfinite(X_tr_nr).all(axis=1)
                if int(tr_ok.sum()) >= 20 and len(np.unique(y_tr[tr_ok])) >= 2:
                    model_nr = _fit_sklearn_gbm(X_tr_nr[tr_ok], y_tr[tr_ok], seed=seed)
            if model_nr is not None:
                proba_te_nr = model_nr.predict_proba(X_te_nr)[:, 1]
                metrics_no_road = evaluate_binary(y_te, proba_te_nr)

        # Máscara comum: células de teste com distância rodoviária válida
        if road_indices:
            with np.errstate(all="ignore"):
                road_te = np.nanmin(X_te[:, road_indices], axis=1)
                road_tr = np.nanmin(X_tr[:, road_indices], axis=1)
            test_road_ok = np.isfinite(road_te)
            train_road_ok = np.isfinite(road_tr)
        else:
            test_road_ok = np.ones(len(y_te), dtype=bool)
            train_road_ok = np.ones(len(y_tr), dtype=bool)
            road_te = np.full(len(y_te), np.nan)
            road_tr = np.full(len(y_tr), np.nan)

        common = test_road_ok if test_road_ok.any() else np.ones(len(y_te), dtype=bool)
        y_common = y_te[common]
        proba_common = proba_te[common]

        prevalence = float(y_tr.mean())
        metrics_full_common = evaluate_binary(y_common, proba_common)
        metrics_prev_common = evaluate_binary(y_common, np.full(len(y_common), prevalence))

        metrics_road_common: Dict[str, float] = {}
        if road_indices and int(train_road_ok.sum()) > 0 and common.any():
            road_cal = fit_platt(-road_tr[train_road_ok], y_tr[train_road_ok], min_positives=8)
            if road_cal is not None:
                metrics_road_common = evaluate_binary(
                    y_common, apply_platt(-road_te[common], road_cal)
                )

        metrics_no_road_common: Dict[str, float] = {}
        if proba_te_nr is not None:
            metrics_no_road_common = evaluate_binary(y_common, proba_te_nr[common])

        # Métricas no conjunto de teste completo (relatório principal)
        metrics = evaluate_binary(y_te, proba_te)
        metrics_const = evaluate_binary(y_te, np.full_like(proba_te, prevalence))

        # Calibração Platt: somente validação temporal, nunca o teste
        calibrator = None
        ece_cal = None
        metrics_cal: Dict[str, float] = {}
        n_pos_va = int(y_va.sum()) if y_va is not None else 0
        if proba_va is not None and y_va is not None and n_pos_va >= 15:
            calibrator = fit_platt(proba_va, y_va, min_positives=15)
            if calibrator is not None:
                last_calibrator = calibrator
                cal_te = apply_platt(proba_te, calibrator)
                ece_cal = expected_calibration_error(y_te, cal_te)
                metrics_cal = evaluate_binary(y_te, cal_te)

        by_cutoff.append(
            {
                "test_year": int(test_year),
                "valid_year": int(valid_year) if valid_year is not None else None,
                "train_years": [int(y) for y in train_years],
                "train_cutoff_year": int(max(train_years)),
                "horizon_years": H,
                "n_train": int(train_m.sum()),
                "n_valid": int(valid_m.sum()),
                "n_test": int(test_m.sum()),
                "n_test_common_road": int(common.sum()),
                # Métricas no teste completo
                "metrics_model": metrics,
                "metrics_model_no_road": metrics_no_road,
                "metrics_prevalence_baseline": metrics_const,
                # Comparação justa (mesmo subconjunto com estrada válida)
                "metrics_on_common_road_subset": {
                    "model_full": metrics_full_common,
                    "model_no_road": metrics_no_road_common,
                    "prevalence_baseline": metrics_prev_common,
                    "road_distance_baseline": metrics_road_common,
                },
                "metrics_after_platt": metrics_cal,
                "ece_raw": expected_calibration_error(y_te, proba_te),
                "ece_platt": ece_cal,
                "calibration_fit": calibrator is not None,
                "calibration_n_positives_valid": n_pos_va,
            }
        )

    aucs = [
        c["metrics_model"]["roc_auc"]
        for c in by_cutoff
        if c["metrics_model"].get("roc_auc") == c["metrics_model"].get("roc_auc")
    ]
    return {
        "cutoffs": [c["test_year"] for c in by_cutoff],
        "by_cutoff": by_cutoff,
        "n_cutoffs": len(by_cutoff),
        "horizon_years": H,
        "label_embargo": "train t0 <= test_year - horizon_years - 1; valid = last eligible year; train < valid",
        "neighbor_feature_embargo": "neighbor year_t0 <= row_year - horizon_years - 1 (aligned with label observability)",
        "median_roc_auc": round(float(np.median(aucs)), 6) if aucs else None,
        "calibrator_last_validation": last_calibrator,
        "comparison_protocol": [
            "model_full",
            "model_no_road",
            "prevalence_baseline",
            "road_distance_baseline",
        ],
        "note": (
            "Walk-forward v24: validação temporal real; Platt só na validação; "
            "comparação justa modelo completo vs sem estradas vs baselines no mesmo subconjunto; "
            "features de vizinhos com embargo conservador (t0 <= T-H-1)."
        ),
    }

