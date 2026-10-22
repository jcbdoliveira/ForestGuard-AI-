"""
routers.py — APIRouter com endpoints de negócio (Fase 2 e 5 do edital SCTEC).

Rotas obrigatórias do edital:
  GET  /health
  POST /v1/predict  (registrado também em app.py; alias /predict aqui)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from supressao_vegetacao.api import services
from supressao_vegetacao.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(tags=["core"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Monitoramento de saúde",
    tags=["health"],
)
def health() -> HealthResponse:
    """200 OK se o serviço está ativo e o modelo reside em RAM (preload no lifespan)."""
    status = services.get_model_status()
    models = status.get("models") or []
    analyses = status.get("analyses") or []
    model_loaded = len(models) > 0
    from supressao_vegetacao.settings import settings

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service="ForestGuard API",
        version=str(getattr(settings, "model_version", "v27.1.0")),
        model_in_memory=model_loaded,
        models_loaded=[
            {"ibge_code": m.get("ibge_code"), "target_col": m.get("target_col")}
            for m in models
        ],
        analyses_cached=[
            {"ibge_code": a.get("ibge_code"), "horizons": a.get("horizons")}
            for a in analyses
        ],
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Inferência espacial",
    tags=["predict"],
)
def predict_alias(req: PredictRequest, request: Request) -> PredictResponse:
    """Alias de /v1/predict exigido pelo edital (POST /predict)."""
    try:
        from supressao_vegetacao.api.app import _check_api_key

        _check_api_key(request)
    except HTTPException:
        raise
    except Exception:
        pass

    result = services.run_spatial_predict(
        ibge_code=req.ibge_code,
        years=req.years,
        top_n=req.top_n,
        target=req.target,
    )
    return PredictResponse(**result)
