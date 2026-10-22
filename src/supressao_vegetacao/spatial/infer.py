"""
Carrega modelo espacial treinado e pontua células na inferência.

Política de seleção (prioridade):
  1. model_dir explícito (ex.: data/spatial/models/{ibge_code})
  2. modelo específico do município (se ibge_code informado)
  3. data/spatial/models/default/
  4. data/spatial/models/ (global)
  5. demais subpastas

Features ausentes nunca são preenchidas com medianas. Só células com todas as
features observadas recebem score multi-feature; as demais ficam explicitamente
sem score. A rede viária é resolvida por GeoJSON específico do município.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .panel import prepare_inference_features

ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = ROOT / "data" / "spatial" / "models"


def find_spatial_model(
    model_dir: Optional[str] = None,
    ibge_code: Optional[str] = None,
) -> Optional[Path]:
    """
    Localiza spatial_model.json|pkl.

    v16 (parecer C1/C2):fallbacks hierárquicos REMOVIDOS. Só carrega:
      1) model_dir explícito (ex.: diretório do job)
      2) modelo específico do município (data/spatial/models/{ibge_code})
    Nenhum modelo default/global/de outro município é mais usado —
    isso impede que um modelo treinado para uma cidade pontue outra.
    """
    candidates: List[Path] = []

    # 1) model_dir explícito
    if model_dir:
        candidates.append(Path(model_dir))

    # 2) modelo específico do município (prioriza pasta anual se existir)
    if ibge_code:
        code = str(ibge_code).strip()
        candidates.append(MODELS_ROOT / f"{code}_annual")
        candidates.append(MODELS_ROOT / code)

    # O modelo legado na raiz pode ser usado apenas se o manifesto o vincular
    # ao IBGE pedido; isso não reintroduz fallback para modelos de outras cidades.
    if ibge_code:
        root_meta = MODELS_ROOT / "spatial_model_meta.json"
        root_model = MODELS_ROOT / "spatial_model.json"
        if root_meta.is_file() and root_model.is_file():
            try:
                root_data = json.loads(root_meta.read_text(encoding="utf-8"))
                if (str(root_data.get("ibge_code") or "").strip() == str(ibge_code).strip()
                        and root_data.get("data_source") == "gee_mapbiomas_panel"):
                    candidates.append(MODELS_ROOT)
            except (OSError, json.JSONDecodeError):
                pass

    for d in candidates:
        d = Path(d)
        if not d.is_dir():
            continue
        for name in ("spatial_model.json", "spatial_model.pkl"):
            p = d / name
            if p.exists():
                meta_path = p.parent / "spatial_model_meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if meta.get("data_source") != "gee_mapbiomas_panel":
                    continue
                if ibge_code and str(meta.get("ibge_code") or "").strip() != str(ibge_code).strip():
                    continue
                return p
    return None


def load_spatial_model(model_path: Path) -> Tuple[Any, List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    meta_path = model_path.parent / "spatial_model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = list(meta.get("feature_names") or [])

    if model_path.suffix == ".json":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        if not feature_names and hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        return model, feature_names, meta

    import pickle
    with model_path.open("rb") as f:
        payload = pickle.load(f)
    model = payload["model"] if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and payload.get("feature_names"):
        feature_names = list(payload["feature_names"])
    return model, feature_names, meta


def cell_feature_vector(
    cell: Dict[str, Any],
    feature_names: List[str],
    feature_medians: Optional[Dict[str, float]] = None,
    allow_impute: bool = False,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Monta vetor de features a partir da célula.

    Features essenciais ausentes (ambientais) impedem o score.
    Features opcionais (estradas, vizinhança histórica) podem ser NaN —
    XGBoost trata nativamente; não imputamos 50 km nem medianas.
    """
    feats = cell.get("features") or {}
    values: List[float] = []
    imputed: List[str] = []
    nan_allowed = {
        "road_dist_km", "road_dist_major_km", "road_dist_secondary_km",
        "road_dist_local_km", "road_dist_track_km",
        "built_surface", "built_dist_km",
        "forest_fraction_t0", "valid_fraction_t0", "years_stable_forest",
        "loss_neighbors_history_1y", "loss_neighbors_history_3y",
        "loss_neighbors_persistence_3y",
    }

    for name in feature_names:
        val: Optional[float] = None
        if name in feats and feats[name] is not None:
            try:
                val = float(feats[name])
            except (TypeError, ValueError):
                val = None
        if val is not None and np.isfinite(val):
            values.append(float(val))
        elif name in nan_allowed:
            values.append(float("nan"))
        else:
            return None, []

    return np.asarray(values, dtype=np.float64), imputed



def _positive_class_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    """Extrai a probabilidade da classe positiva 1 sem presumir a ordem das classes."""
    if not hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
        return np.clip(pred, 0.0, 1.0)

    matrix = np.asarray(model.predict_proba(X), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(X):
        raise ValueError("predict_proba retornou uma matriz em formato inesperado")
    classes = list(getattr(model, "classes_", []))
    if classes:
        positive_indices = [i for i, label in enumerate(classes) if label == 1 or str(label) == "1"]
        if not positive_indices:
            return np.zeros(matrix.shape[0], dtype=np.float64)
        idx = positive_indices[0]
        if idx >= matrix.shape[1]:
            raise ValueError("classes_ não corresponde às colunas de predict_proba")
        return np.clip(matrix[:, idx], 0.0, 1.0)
    if matrix.shape[1] == 2:
        # Compatibilidade apenas para estimadores sem classes_: convenção sklearn.
        return np.clip(matrix[:, 1], 0.0, 1.0)
    raise ValueError("não é possível identificar com segurança a classe positiva do modelo")

def score_cells(
    cells: List[Dict[str, Any]],
    model_dir: Optional[str] = None,
    ibge_code: Optional[str] = None,
    allow_impute: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Retorna dict com proba por índice de célula e metadados, ou None se não aplicável.

    Preferência:
      1) Painel GEE do município + prepare_inference_features (mesma semântica do treino)
      2) Features embutidas nas células (desktop/API), sem imputação arbitraria

    Sem imputação por mediana. Células incompletas são reportadas sem score.
    """
    path = find_spatial_model(model_dir=model_dir, ibge_code=ibge_code)
    if path is None:
        return None
    try:
        model, feature_names, meta = load_spatial_model(path)
    except Exception as exc:
        return {"error": str(exc), "model_path": str(path)}

    if not feature_names:
        return None

    horizon = int(meta.get("horizon_years") or (1 if str(meta.get("target_col") or "").endswith("1y") else 3))
    X_rows: List[np.ndarray] = []
    idx_map: List[int] = []
    all_imputed: set = set()
    any_imputed = False
    unscored: List[Dict[str, Any]] = []
    used_panel = False

    # --- Caminho preferencial: painel histórico do município ---
    if ibge_code:
        panel_dir = ROOT / "data" / "spatial" / "panels" / str(ibge_code).strip()
        if panel_dir.is_dir():
            try:
                prepared = prepare_inference_features(
                    panel_dir,
                    feature_names=feature_names,
                    target_year=None,
                    horizon_years=horizon,
                    require_forest_t0=True,
                )
                # Match painel -> células do grid por proximidade de centroide
                panel_lons = np.array([float(m.get("lon") or np.nan) for m in prepared["meta"]], dtype=np.float64)
                panel_lats = np.array([float(m.get("lat") or np.nan) for m in prepared["meta"]], dtype=np.float64)
                X_panel = prepared["X"]
                if X_panel.shape[0] > 0 and np.isfinite(panel_lons).any():
                    used_panel = True
                    panel_ids = {
                        str(m.get("cell_id")): j
                        for j, m in enumerate(prepared["meta"])
                        if m.get("cell_id")
                    }
                    for i, cell in enumerate(cells):
                        if not cell.get("is_inside_boundary", True):
                            continue
                        cid = cell.get("cell_id")
                        j = panel_ids.get(str(cid)) if cid else None
                        if j is None:
                            cen = cell.get("centroid") or [None, None]
                            try:
                                clat = float(cen[0])
                                clon = float(cen[1])
                            except (TypeError, ValueError, IndexError):
                                unscored.append({"cell_index": i, "cell_id": cid, "missing_features": ["centroid"]})
                                continue
                            dist = np.hypot(panel_lons - clon, panel_lats - clat)
                            j = int(np.nanargmin(dist))
                            if not np.isfinite(dist[j]) or dist[j] > 0.015:
                                unscored.append({
                                    "cell_index": i,
                                    "cell_id": cid,
                                    "missing_features": ["panel_match"],
                                })
                                continue
                        X_rows.append(X_panel[j])
                        idx_map.append(i)
            except Exception:
                used_panel = False
                X_rows = []
                idx_map = []
                unscored = []

    # --- Fallback: features já presentes nas células ---
    if not used_panel:
        for i, cell in enumerate(cells):
            if not cell.get("is_inside_boundary", True):
                continue
            vec, imputed = cell_feature_vector(
                cell, feature_names, feature_medians=None, allow_impute=False
            )
            if vec is None:
                feats = cell.get("features") or {}
                missing = [name for name in feature_names if feats.get(name) is None]
                unscored.append({"cell_index": i, "cell_id": cell.get("cell_id"), "missing_features": missing})
                continue
            if imputed:
                any_imputed = True
                all_imputed.update(imputed)
            X_rows.append(vec)
            idx_map.append(i)

    if not X_rows:
        return {
            "scores": {},
            "model_path": str(path),
            "model_type": meta.get("model_type") or "unknown",
            "data_source": meta.get("data_source") or "gee_mapbiomas_panel",
            "feature_names": feature_names,
            "metrics_test": meta.get("metrics_test") or {},
            "metrics_apply_to_displayed_map": False,
            "feature_importance": meta.get("feature_importance") or [],
            "features_imputed": False,
            "imputed_features": [],
            "unscored_cells": unscored,
            "map_feature_mode": "road_only_or_incomplete_features_no_multifeature_score",
            "training_note": meta.get("training_note") or "",
            "probability_policy": {"calibrated": False, "metrics_apply_to_displayed_map": False,
                                   "reason": "features observadas insuficientes; sem imputação"},
            "walk_forward": meta.get("walk_forward"),
            "selection": {"requested_model_dir": model_dir, "ibge_code": ibge_code, "resolved_path": str(path)},
        }

    X = np.vstack(X_rows)
    proba = _positive_class_probabilities(model, X)
    if not np.all(np.isfinite(proba)):
        return {"error": "modelo espacial retornou escores não finitos", "model_path": str(path)}

    calibrator = meta.get("calibrator") or meta.get("calibrator_last_validation")
    from ..reliability import apply_platt, map_probability_policy

    policy = map_probability_policy(any_imputed, calibrator if isinstance(calibrator, dict) else None)
    if policy["calibrated"] and isinstance(calibrator, dict):
        proba = apply_platt(proba, calibrator)

    scores = {idx_map[j]: float(proba[j]) for j in range(len(idx_map))}
    return {
        "scores": scores,
        "model_path": str(path),
        "model_type": meta.get("model_type") or type(model).__name__,
        "data_source": meta.get("data_source") or "gee_mapbiomas_panel",
        "target_col": meta.get("target_col"),
        "horizon_years": horizon,
        "feature_names": feature_names,
        "metrics_test": meta.get("metrics_test") or {},
        "metrics_apply_to_displayed_map": policy["metrics_apply_to_displayed_map"],
        "feature_importance": meta.get("feature_importance") or [],
        "features_imputed": any_imputed,
        "imputed_features": sorted(all_imputed),
        "unscored_cells": unscored,
        "map_feature_mode": ("gee_panel_matched" if used_panel else ("observed_features_only" if not any_imputed else "imputed_features_legacy")),
        "training_note": meta.get("training_note") or "",
        "probability_policy": policy,
        "walk_forward": meta.get("walk_forward"),
        "selection": {
            "requested_model_dir": model_dir,
            "ibge_code": ibge_code,
            "resolved_path": str(path),
        },
    }
