"""
services.py — camada de negócio / inferência (Fase 2 e 3 do edital SCTEC).

Carrega o modelo uma única vez (via model_cache + lifespan) e executa a
inferência sem reler o disco a cada requisição HTTP.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from supressao_vegetacao.api import model_cache
from supressao_vegetacao.spatial.infer import _positive_class_probabilities

logger = logging.getLogger("forestguard.api.services")

ROOT = Path(__file__).resolve().parents[3]


def get_model_status() -> Dict[str, Any]:
    return model_cache.status()


def run_spatial_predict(
    ibge_code: str,
    years: int = 1,
    top_n: int = 20,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inferência espacial com o XGBoost já em RAM (preload no lifespan).

    Proibido pelo edital: abrir .json/.joblib do modelo a cada request —
    aqui só usamos model_cache.get_or_load_model (hit em memória após startup).
    """
    code = str(ibge_code).strip()
    if not 1 <= int(years) <= 20:
        raise HTTPException(400, detail="years deve estar entre 1 e 20.")

    requested_target = (target or "").strip() or "y_loss_1y"
    if requested_target not in {"y_loss_1y", "y_loss_3y"}:
        raise HTTPException(
            400, detail="target deve ser y_loss_1y (principal)."
        )
    if requested_target == "y_loss_3y" and int(years) != 3:
        raise HTTPException(400, detail="target=y_loss_3y (legado) exige years=3.")

    try:
        cached = model_cache.get_or_load_model(code)
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falha ao obter modelo em cache")
        raise HTTPException(500, detail="Não foi possível carregar o modelo treinado.") from exc

    model = cached.model
    feature_names = cached.feature_names
    meta = cached.meta
    model_path = cached.model_path

    if str(meta.get("ibge_code") or "").strip() not in ("", code):
        raise HTTPException(
            404, detail="O manifesto do modelo não corresponde ao código IBGE solicitado."
        )
    model_target = str(meta.get("target_col") or "")
    if model_target and model_target != requested_target:
        raise HTTPException(
            409,
            detail=(
                f"O modelo disponível está registrado para '{model_target}', "
                f"mas a requisição pediu '{requested_target}'."
            ),
        )

    panel_dir = ROOT / "data" / "spatial" / "panels" / code
    if not panel_dir.is_dir():
        raise HTTPException(404, detail=f"Painel de features não encontrado para o IBGE {code}.")

    horizon = int(meta.get("horizon_years") or (3 if requested_target == "y_loss_3y" else 1))
    try:
        prepared = model_cache.get_or_prepare_features(
            code,
            feature_names=feature_names,
            horizon_years=horizon,
            panel_dir=panel_dir,
            target_year=None,
            require_forest_t0=True,
        )
    except Exception as exc:
        logger.exception("Falha ao preparar features")
        raise HTTPException(500, detail=f"Falha ao preparar features: {exc}") from exc

    import numpy as np

    X = prepared.get("X")
    if X is None or getattr(X, "shape", (0,))[0] == 0:
        raise HTTPException(404, detail="Nenhuma célula com features observadas no painel.")

    proba = _positive_class_probabilities(model, np.asarray(X))
    meta_rows = prepared.get("meta") or []
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(proba):
        m = meta_rows[i] if i < len(meta_rows) else {}
        rows.append(
            {
                "rank": 0,
                "cell_id": m.get("cell_id"),
                "probability": round(float(p), 6),
                "latitude": m.get("lat"),
                "longitude": m.get("lon"),
            }
        )
    rows.sort(key=lambda r: r["probability"], reverse=True)
    for rank, r in enumerate(rows[: int(top_n)], start=1):
        r["rank"] = rank

    return {
        "ibge_code": code,
        "target": requested_target,
        "years": int(years),
        "model_path": str(model_path),
        "model_in_memory": True,
        "n_cells_scored": len(rows),
        "top_cells": rows[: int(top_n)],
        "metrics_test": meta.get("metrics_test"),
        "disclaimer": (
            "Risco relativo estimado. Não constitui desmatamento consumado nem auto de infração."
        ),
    }


def run_analysis(
    ibge_code: str,
    horizons: List[int],
    pressure: float = 1.0,
) -> Dict[str, Any]:
    """Análise municipal com cache em RAM (horizontes 1–20 pré-carregados)."""
    return model_cache.get_or_run_analysis(
        ibge_code, horizons=list(horizons), pressure=float(pressure)
    )
