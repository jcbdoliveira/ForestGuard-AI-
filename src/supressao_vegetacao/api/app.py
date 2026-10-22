"""
ForestGuard API v25 — FastAPI (superfície mínima)

Apenas estas rotas estão ativas (demais respondem 404):

  GET  /health
  POST /v1/analyze
  GET  /v1/analyze/{ibge_code}/pdf
  POST /v1/simulate/annual
  POST /v1/predict
  GET  /v1/dashboard/{ibge_code}
  GET  /v1/dashboard/{ibge_code}/view

Rotas legadas (municipalities, jobs, project, metrics, models, artifacts,
download, report) permanecem no código mas são bloqueadas pelo middleware
de allowlist.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from supressao_vegetacao.spatial.dynamic_neighborhood import (
    build_knn_neighbors as _build_knn_neighbors,
    recompute_weights_dynamic as _recompute_weights_dynamic,
)

from pydantic import BaseModel, Field, field_validator

# v16 (parecer M2): logging basico configurado no modulo da API
logging.basicConfig(
    level=os.environ.get("FORESTGUARD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supressao_vegetacao.inference.predict import run_city_deforestation_analysis as run_city_analysis  # type: ignore
from supressao_vegetacao.sources.municipality import (  # type: ignore
    get_municipality,
    list_municipalities,
)
from supressao_vegetacao.desktop.pdf_report import generate_technical_report_pdf  # type: ignore
from supressao_vegetacao.settings import settings  # type: ignore
from supressao_vegetacao.spatial.infer import find_spatial_model, load_spatial_model, _positive_class_probabilities  # type: ignore
from supressao_vegetacao.spatial.panel import prepare_inference_features  # type: ignore
from supressao_vegetacao.api.dashboard import build_dashboard_payload, render_dashboard_html  # type: ignore
from supressao_vegetacao.api import model_cache  # type: ignore

try:
    from .jobs import create_job, list_jobs, load_job, run_job
    HAS_JOBS = False
except Exception:
    HAS_JOBS = False

ARTIFACTS_DIR = ROOT / "data" / "api_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def _parse_preload_codes() -> List[str]:
    """Códigos IBGE para pré-carregar: FORESTGUARD_PRELOAD_IBGE=4210100,outro.

    Se a variável não estiver definida, usa municípios com painel real
    (tipicamente 4210100) para o startup já deixar /v1/analyze e /v1/predict quentes.
    """
    raw = os.environ.get("FORESTGUARD_PRELOAD_IBGE", "").strip()
    if not raw:
        # Default: painéis reais presentes no pacote
        default_panel = ROOT / "data" / "spatial" / "panels" / "4210100"
        if default_panel.is_dir() and any(default_panel.glob("*.csv")):
            return ["4210100"]
        return []
    parts: List[str] = []
    for chunk in raw.replace(";", ",").split(","):
        code = "".join(ch for ch in chunk.strip() if ch.isdigit())
        if code:
            parts.append(code)
    # unique preserve order
    seen = set()
    out: List[str] = []
    for c in parts:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@asynccontextmanager
async def _lifespan(app: FastAPI):
    codes = _parse_preload_codes()
    log = logging.getLogger("forestguard.api")
    if codes:
        log.info(
            "Preload no startup (CSVs + modelo + features + análise 1–20): %s",
            ", ".join(codes),
        )
        results = model_cache.preload_many(codes, also_features=True)
        for r in results:
            if r.get("error"):
                log.error(
                    "Preload falhou IBGE=%s: %s", r.get("ibge_code"), r.get("error")
                )
            else:
                log.info(
                    "Preload pronto IBGE=%s csv_rows=%s cells=%s analysis=%s horizons=%s (%.2fs)",
                    r.get("ibge_code"),
                    r.get("csv_rows"),
                    r.get("cells_scored"),
                    r.get("analysis_preloaded"),
                    r.get("analysis_horizons"),
                    r.get("elapsed_s"),
                )
    else:
        log.info(
            "Nenhum preload — /v1/analyze e /v1/predict farão carga sob demanda. "
            "Use --preload-ibge 4210100 ou FORESTGUARD_PRELOAD_IBGE."
        )
    yield


app = FastAPI(
    title="ForestGuard API",
    description=(
        "ForestGuard-AI v27 — análise preditiva de risco de supressão de vegetação. "        
    ),
    version=str(getattr(settings, "model_version", "v27.1.0")),
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Allowlist estrita — qualquer outra rota devolve 404.
_ALLOWED_ROUTE_PATTERNS = [
    ("GET", re.compile(r"^/health/?$")),
    ("POST", re.compile(r"^/v1/analyze/?$")),
    ("GET", re.compile(r"^/v1/analyze/[^/]+/pdf/?$")),
    ("POST", re.compile(r"^/v1/simulate/annual/?$")),
    ("POST", re.compile(r"^/v1/predict/?$")),
    ("POST", re.compile(r"^/predict/?$")),  # alias exigido pelo edital SCTEC
    ("GET", re.compile(r"^/v1/dashboard/[^/]+/?$")),
    ("GET", re.compile(r"^/v1/dashboard/[^/]+/view/?$")),
    # Documentação Swagger / OpenAPI (critério de avaliação — endpoints no /docs)
    ("GET", re.compile(r"^/docs/?$")),
    ("GET", re.compile(r"^/docs/oauth2-redirect/?$")),
    ("GET", re.compile(r"^/redoc/?$")),
    ("GET", re.compile(r"^/openapi\.json$")),
]


@app.middleware("http")
async def _enforce_route_allowlist(request: Request, call_next):
    """Bloqueia rotas fora da superfície mínima (inclui jobs, municipalities, docs)."""
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    method = request.method.upper()
    for allowed_method, pattern in _ALLOWED_ROUTE_PATTERNS:
        if method == allowed_method and pattern.match(path):
            return await call_next(request)
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Rota não disponível nesta superfície da API.",
            "allowed": [
                "GET /health",
                "POST /v1/analyze",
                "GET /v1/analyze/{ibge_code}/pdf",
                "POST /v1/simulate/annual",
                "POST /v1/predict",
                "POST /predict",
                "GET /v1/dashboard/{ibge_code}",
                "GET /v1/dashboard/{ibge_code}/view",
                "GET /docs",
            ],
        },
    )


# v16 (parecer M1): CORS restrito a origens locais de desenvolvimento.
_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "FORESTGUARD_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Camadas do edital SCTEC: routers / schemas / services
from supressao_vegetacao.api.routers import router as core_router  # noqa: E402
app.include_router(core_router)


# v16 (parecer M1): API-key opcional nos endpoints mutantes.
# Configurar FORESTGUARD_API_KEY; se a variavel nao estiver definida (desenvolvimento
# local), a verificacao fica desativada e um aviso e emitido no boot.
_API_KEY = os.environ.get("FORESTGUARD_API_KEY")


_IS_PRODUCTION = os.environ.get("NODE_ENV", "").lower() == "production" or os.environ.get("FORESTGUARD_ENV", "").lower() == "production"
if not _API_KEY:
    if _IS_PRODUCTION:
        logging.getLogger("forestguard.api").critical(
            "FORESTGUARD_API_KEY ausente em producao; endpoints protegidos serao bloqueados."
        )
    else:
        logging.getLogger("forestguard.api").warning(
            "FORESTGUARD_API_KEY nao definida; autenticacao desativada apenas para desenvolvimento local."
        )


def _check_api_key(request: Request) -> None:
    """Falha fechada em producao; permite acesso sem chave apenas em desenvolvimento."""
    if not _API_KEY:
        if _IS_PRODUCTION:
            raise HTTPException(503, detail="Autenticacao nao configurada no servidor")
        return
    supplied = request.headers.get("X-API-Key", "")
    # compare_digest evita comparacao de segredo com tempo dependente do prefixo.
    import hmac
    if not hmac.compare_digest(supplied, _API_KEY):
        raise HTTPException(401, detail="X-API-Key invalida ou ausente")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    city: Optional[str] = Field(None, description="NOME-UF ou codigo IBGE")
    ibge_code: Optional[str] = Field(None, description="Codigo IBGE 7 digitos")
    pressure: float = Field(1.0, ge=0.5, le=2.0)
    horizons: List[int] = Field(
        default_factory=lambda: [1, 3, 5, 8, 10, 20],
        description="Horizontes em anos (1–20; no startup 1–20 ficam em RAM)",
    )

    @field_validator("horizons")
    @classmethod
    def _check_horizons(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Informe ao menos um horizonte entre 1 e 20 anos")
        if any(isinstance(x, bool) or not 1 <= int(x) <= 20 for x in v):
            raise ValueError(
                "Todos os horizontes devem estar entre 1 e 20 anos (máximo da API)"
            )
        out = list(dict.fromkeys(int(x) for x in v))
        if len(out) > 20:
            raise ValueError("Informe no máximo 20 horizontes por análise")
        return out


class ProjectRequest(BaseModel):
    """Cenario de projecao para X anos — retorna top 5 celulas criticas."""
    city: Optional[str] = Field(None, description="NOME-UF ou codigo IBGE")
    ibge_code: Optional[str] = Field(None, description="Codigo IBGE 7 digitos")
    years: int = Field(..., ge=1, le=20, description="Horizonte de projecao em anos (1–20)")
    pressure: float = Field(1.0, ge=0.5, le=2.0)
    top_n: int = Field(5, ge=1, le=50, description="Quantidade de celulas criticas (padrao 5)")


class AnnualSimulationRequest(BaseModel):
    """Simulação anual com atualização do remanescente e feedback dinâmico de vizinhança (v23)."""
    ibge_code: str = Field(..., min_length=7, max_length=7, pattern=r"^\d{7}$", description="Código IBGE de 7 dígitos")
    years: int = Field(..., ge=1, le=20, description="Horizonte da simulação anual (1 a 20 anos)")
    pressure: float = Field(1.0, ge=0.5, le=2.0, description="Multiplicador de cenário, limitado entre 0,5 e 2,0")
    top_n: int = Field(20, ge=1, le=100, description="Número de células de maior risco retornadas por ano")
    include_cell_maps: bool = Field(False, description="Inclui alocação por célula em cada ano; pode aumentar muito a resposta")
    dynamic_neighborhood: bool = Field(
        True,
        description="Ativa o Loop de Feedback Dinâmico de Vizinhança: recalcula forest_neighbor_frac e ajusta os pesos de risco a cada ano (v23)",
    )
    neighborhood_k: int = Field(
        8,
        ge=3,
        le=24,
        description="Número de vizinhos k-NN usados no cálculo dinâmico de forest_neighbor_frac",
    )
    edge_effect_beta: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description="Intensidade do efeito de borda: weight *= (1 + beta * (1 - forest_neighbor_frac))",
    )


class TrainedPredictionRequest(BaseModel):
    """Predição do classificador espacial treinado (alvo principal: y_loss_1y).

    years 1–20: o modelo anual (y_loss_1y) é o principal.
    """
    ibge_code: str = Field(..., min_length=7, max_length=7, pattern=r"^\d{7}$", description="Código IBGE de 7 dígitos")
    years: int = Field(
        1,
        ge=1,
        le=20,
        description="Horizonte (1–20). Modelo principal y_loss_1y; 3 também aceita legado y_loss_3y",
    )
    top_n: int = Field(20, ge=1, le=500, description="Número de células para retornar")
    target: Optional[str] = Field(
        None,
        description="Alvo explícito (y_loss_1y).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ibge(city: Optional[str], ibge_code: Optional[str]) -> str:
    code = (ibge_code or city or "").strip()
    if not code:
        raise HTTPException(400, detail="Informe city ou ibge_code")
    # se for so digitos
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) == 7:
        return digits
    # tenta pelo catalogo
    try:
        muni = get_municipality(code)
        return str(muni["ibge_code"])
    except Exception as exc:
        raise HTTPException(404, detail=f"Municipio nao encontrado: {code} ({exc})") from exc


def _run_analysis(ibge: str, horizons: List[int], pressure: float) -> Dict[str, Any]:
    """Usa cache em RAM (preload 1–20) quando possível."""
    try:
        return model_cache.get_or_run_analysis(ibge, horizons=list(horizons), pressure=pressure)
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger("forestguard.api").exception("Falha durante análise municipal")
        raise HTTPException(
            500,
            detail="Falha interna durante a análise. Consulte os logs do servidor.",
        ) from exc


def _artifact_dir(ibge: str) -> Path:
    d = ARTIFACTS_DIR / str(ibge)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_horizons_param(raw: Optional[str], fallback: Optional[List[int]] = None) -> List[int]:
    """Aceita '17' ou '1,5,10' ou None."""
    if raw is None or str(raw).strip() == "":
        return list(fallback) if fallback else [1, 3, 5, 8, 10, 20]
    parts = re.split(r"[;,\s]+", str(raw).strip())
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            h = int(p)
        except ValueError as exc:
            raise HTTPException(400, detail=f"Horizonte inválido: {p!r}") from exc
        if not 1 <= h <= 20:
            raise HTTPException(400, detail=f"Horizonte fora de 1–20: {h}")
        if h not in out:
            out.append(h)
    if not out:
        raise HTTPException(400, detail="Informe ao menos um horizonte entre 1 e 20")
    return out


def _ensure_analysis_artifacts(
    ibge: str,
    horizons: Optional[List[int]] = None,
    pressure: float = 1.0,
    *,
    generate_pdf: bool = True,
    slim_cells: bool = False,
) -> Dict[str, Any]:
    """Usa análise em cache quando possível; grava JSON (+ PDF opcional).

    slim_cells=True remove a lista de células do resultado em memória de resposta
    (mantém horizon_summaries) — evita payload gigante no Swagger.
    """
    horizons = horizons or [1, 3, 5, 8, 10, 20]
    bad = [h for h in horizons if not 1 <= int(h) <= 20]
    if bad:
        raise HTTPException(400, detail=f"Horizontes fora do intervalo 1–20: {bad}")
    result = _run_analysis(ibge, horizons, pressure)
    out = _artifact_dir(ibge)
    name = (result.get("municipality") or {}).get("name") or ibge
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))

    # Persistência com horizontes pedidos (PDF e JSON refletem a mesma análise)
    json_path = out / f"analysis_{ibge}.json"
    meta_path = out / f"analysis_{ibge}_request.json"
    meta_path.write_text(
        json.dumps({"horizons": list(horizons), "pressure": float(pressure)}, ensure_ascii=False),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    pdf_path = out / f"Relatorio_{ibge}_{safe}.pdf"
    pdf_error = None
    if generate_pdf:
        try:
            generate_technical_report_pdf(result, str(pdf_path))
        except Exception as exc:
            pdf_path = None  # type: ignore
            pdf_error = str(exc)
    elif not pdf_path.exists():
        pdf_path = None  # type: ignore

    artifacts = {
        "analysis_json": str(json_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)) if pdf_path and Path(pdf_path).exists() else None,
    }
    boundary = ROOT / "data" / "boundaries" / f"{ibge}.geojson"
    if boundary.exists():
        artifacts["boundary_geojson"] = str(boundary.relative_to(ROOT))

    if slim_cells and isinstance(result, dict):
        result = {k: v for k, v in result.items() if k != "cells"}

    return {
        "result": result,
        "artifacts": artifacts,
        "pdf_error": pdf_error,
    }


def _google_maps_url(lat: float, lon: float, zoom: int = 15) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}&z={zoom}"


def _top_critical_cells(result: Dict[str, Any], years: int, top_n: int = 5) -> Dict[str, Any]:
    """Extrai top-N celulas criticas do horizonte mais proximo de `years`."""
    summaries = result.get("horizon_summaries") or {}
    # chave preferencial
    key = f"{years}y"
    if key not in summaries:
        # pega o horizonte numerico mais proximo
        best = None
        best_diff = 10**9
        for k, s in summaries.items():
            hy = int(s.get("horizon_years") or k.replace("y", "") or 0)
            d = abs(hy - years)
            if d < best_diff:
                best_diff = d
                best = k
        key = best or (next(iter(summaries)) if summaries else None)
    if not key:
        raise HTTPException(404, detail="Nenhum horizonte disponivel no resultado")

    summary = summaries[key]
    points = list(summary.get("top_attention_points") or [])
    # se top_attention_points tiver menos que top_n, completa pelas celulas
    if len(points) < top_n:
        cells = result.get("cells") or []
        scored = []
        for c in cells:
            if not c.get("is_inside_boundary", True):
                continue
            sc = (c.get("scenarios") or {}).get(key) or {}
            scored.append({
                "cell_id": c.get("cell_id"),
                "locality": c.get("locality"),
                "coordinates": c.get("centroid"),
                "probability": sc.get("probability", 0),
                "attention_level": sc.get("attention_level"),
                "expected_suppressed_ha": sc.get("expected_suppressed_ha"),
                "initial_forest_ha": c.get("initial_forest_ha"),
                "dist_road_km": c.get("dist_road_km"),
                "dominant_driver": c.get("dominant_driver"),
            })
        scored.sort(key=lambda x: float(x.get("probability") or 0), reverse=True)
        points = scored

    top = points[:top_n]
    cells_out = []
    for i, p in enumerate(top, start=1):
        coords = p.get("coordinates") or [None, None]
        # centroid no motor e [lat, lon]
        lat = float(coords[0]) if coords[0] is not None else None
        lon = float(coords[1]) if coords[1] is not None else None
        if lat is not None and abs(lat) > 90 and lon is not None and abs(lon) <= 90:
            # se vier [lon, lat]
            lat, lon = lon, lat
        cell = {
            "rank": int(p.get("rank") or i),
            "cell_id": p.get("cell_id"),
            "locality": p.get("locality"),
            "attention_level": p.get("attention_level"),
            "statistics": {
                "probability": float(p.get("probability") or 0),
                "expected_suppressed_ha": float(p.get("expected_suppressed_ha") or 0),
                "initial_forest_ha": float(p.get("initial_forest_ha") or 0) if p.get("initial_forest_ha") is not None else None,
                "dist_road_km": float(p.get("dist_road_km") or 0) if p.get("dist_road_km") is not None else None,
                "lower_sensitivity_bound": (p.get("ci_90") or [None, None])[0] if isinstance(p.get("ci_90"), list) else p.get("lower_sensitivity_bound"),
                "upper_sensitivity_bound": (p.get("ci_90") or [None, None])[1] if isinstance(p.get("ci_90"), list) else p.get("upper_sensitivity_bound"),
            },
            "latitude": lat,
            "longitude": lon,
            "google_maps_url": _google_maps_url(lat, lon) if lat is not None and lon is not None else None,
            "dominant_driver": p.get("dominant_driver"),
            "recommended_action": p.get("recommended_action"),
        }
        cells_out.append(cell)

    muni = result.get("municipality") or {}
    meta = result.get("model_metadata") or {}
    return {
        "ibge_code": muni.get("ibge_code"),
        "municipality": muni.get("name"),
        "uf": muni.get("uf"),
        "requested_years": years,
        "horizon_key": key,
        "horizon_years": summary.get("horizon_years"),
        "target_year": summary.get("target_year"),
        "horizon_summary": {
            "expected_suppressed_area_ha": summary.get("expected_suppressed_area_ha"),
            "percent_of_remnant_lost": summary.get("percent_of_remnant_lost"),
            "mean_probability": summary.get("mean_probability"),
            "max_probability": summary.get("max_probability"),
            "critical_points_count": summary.get("critical_points_count"),
            "high_points_count": summary.get("high_points_count"),
            "moderate_points_count": summary.get("moderate_points_count"),
            "low_points_count": summary.get("low_points_count"),
        },
        "model": {
            "version": meta.get("model_version"),
            "algorithm": meta.get("algorithm"),
            "spatial_model": meta.get("spatial_model"),
            "metrics_validation": meta.get("metrics_validation"),
        },
        "top_critical_cells": cells_out,
        "disclaimer": (
            "Risco relativo estimado (heuristica espacial demonstrativa). "
            "Nao constitui desmatamento consumado nem auto de infracao."
        ),
    }


# ---------------------------------------------------------------------------
# Rotas novas
# ---------------------------------------------------------------------------

# /health movido para routers.py (camada modular do edital)


# REMOVIDO da superfície da API: @app.get("/v1/municipalities")
# @app.get("/v1/municipalities")
def municipalities(only_with_panel: bool = False):
    """Lista municípios do catálogo.

    Query param only_with_panel=true restringe aos que possuem painel
    espacial real (GEE) — útil após remoção de cidades sem dados completos.
    """
    try:
        items = list_municipalities(only_with_panel=only_with_panel)
        return {
            "count": len(items),
            "only_with_panel": only_with_panel,
            "municipalities": items,
        }
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


#@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    """Análise rápida a partir do cache (horizontes 1–20 pré-carregados).

    Não embute a malha completa de células (evita travar o Swagger) e
    **não** gera PDF por padrão — use GET /v1/analyze/{ibge}/pdf?horizons=...
    """
    _check_api_key(request)
    ibge = _resolve_ibge(req.city, req.ibge_code)
    # Gera JSON em disco; PDF sob demanda no endpoint dedicado (muito mais rápido aqui)
    payload = _ensure_analysis_artifacts(
        ibge,
        req.horizons,
        req.pressure,
        generate_pdf=False,
        slim_cells=True,
    )
    result = payload["result"]
    muni = result.get("municipality") or {}
    hs = req.horizons or []
    hs_q = ",".join(str(h) for h in hs) if hs else ""
    return {
        "ibge_code": ibge,
        "municipality": muni.get("name"),
        "uf": muni.get("uf"),
        "pressure": req.pressure,
        "horizons": list(hs),
        "model_version": (result.get("model_metadata") or {}).get("model_version"),
        "horizon_summaries": {
            k: {
                "horizon_years": v.get("horizon_years"),
                "expected_suppressed_area_ha": v.get("expected_suppressed_area_ha"),
                "mean_probability": v.get("mean_probability"),
                "max_probability": v.get("max_probability"),
                "critical_points_count": v.get("critical_points_count"),
                "high_points_count": v.get("high_points_count"),
                "moderate_points_count": v.get("moderate_points_count"),
                "low_points_count": v.get("low_points_count"),
            }
            for k, v in (result.get("horizon_summaries") or {}).items()
        },
        "model_metadata": {
            "spatial_model": (result.get("model_metadata") or {}).get("spatial_model"),
            "forecast_reliability": (result.get("model_metadata") or {}).get("forecast_reliability"),
        },
        "artifacts": payload["artifacts"],
        "links": {
            "pdf": f"/v1/analyze/{ibge}/pdf?pressure={req.pressure}&horizons={hs_q}&regenerate=true",
        },
        "pdf_error": payload.get("pdf_error"),
        "note": (
            "Resposta enxuta (sem cells). "
            "Gere o PDF com o link acima — os horizontes da análise são respeitados."
        ),
    }


@app.get("/v1/analyze/{ibge_code}/pdf")
def analyze_pdf(
    ibge_code: str,
    request: Request,
    pressure: float = Query(1.0, ge=0.5, le=2.0),
    horizons: Optional[str] = Query(
        None,
        description="Horizontes da predição, ex.: 17 ou 1,5,10. Se omitido, usa a última análise POST /v1/analyze.",
    ),
    regenerate: bool = Query(True, description="Regenera o PDF com os horizontes informados"),
):
    """Gera PDF da **predição** com os horizontes pedidos (não reutiliza treino)."""
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    out = _artifact_dir(ibge)

    # Resolver horizontes: query > última análise > padrão
    hs: Optional[List[int]] = None
    if horizons:
        hs = _parse_horizons_param(horizons)
    else:
        req_meta = out / f"analysis_{ibge}_request.json"
        if req_meta.is_file():
            try:
                meta = json.loads(req_meta.read_text(encoding="utf-8"))
                raw_hs = meta.get("horizons")
                if isinstance(raw_hs, list) and raw_hs:
                    hs = [int(x) for x in raw_hs]
                    if abs(float(meta.get("pressure", pressure)) - float(pressure)) > 1e-6:
                        # pressure diferente da última análise: mantém horizons mas aplica pressure nova
                        pass
            except Exception:
                hs = None
    if not hs:
        hs = [1, 3, 5, 8, 10, 20]

    existing = list(out.glob(f"Relatorio_{ibge}_*.pdf"))
    must_regen = regenerate or not existing

    if must_regen:
        payload = _ensure_analysis_artifacts(ibge, hs, pressure, generate_pdf=True, slim_cells=False)
        rel = (payload.get("artifacts") or {}).get("pdf")
        if not rel:
            raise HTTPException(500, detail=f"Falha ao gerar PDF: {payload.get('pdf_error')}")
        path = ROOT / rel
    else:
        path = max(existing, key=lambda p: p.stat().st_mtime)
    if not path.exists():
        raise HTTPException(404, detail="PDF nao encontrado")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# REMOVIDO da superfície da API: @app.get("/v1/analyze/{ibge_code}/artifacts")
# @app.get("/v1/analyze/{ibge_code}/artifacts")
def analyze_artifacts(ibge_code: str, request: Request, regenerate: bool = Query(False)):
    # A rota pode revelar metadados e provocar geração de arquivos; proteger sempre.
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    out = _artifact_dir(ibge)
    files = list(out.glob("*"))
    if regenerate or not files:
        payload = _ensure_analysis_artifacts(ibge)
        arts = payload["artifacts"]
    else:
        arts = {}
        for f in files:
            key = f.stem
            if f.suffix == ".pdf":
                arts["pdf"] = str(f.relative_to(ROOT))
            elif f.suffix == ".json":
                arts["analysis_json"] = str(f.relative_to(ROOT))
        boundary = ROOT / "data" / "boundaries" / f"{ibge}.geojson"
        if boundary.exists():
            arts["boundary_geojson"] = str(boundary.relative_to(ROOT))

    items = []
    for name, rel in arts.items():
        if not rel:
            continue
        p = ROOT / rel
        items.append({
            "name": name,
            "path": rel,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "download_url": f"/v1/artifacts/download?path={rel}",
        })
    return {
        "ibge_code": ibge,
        "artifacts": items,
        "links": {
            "pdf": f"/v1/analyze/{ibge}/pdf",
            "analyze": "/v1/analyze",
            "project": "/v1/project",
        },
    }


# REMOVIDO da superfície da API: @app.get("/v1/artifacts/download")
# @app.get("/v1/artifacts/download")
def download_artifact(request: Request, path: str = Query(..., description="Caminho relativo ao ROOT do projeto")):
    # Downloads de artefatos exigem a mesma autenticação dos endpoints de geração.
    _check_api_key(request)
    # previne path traversal - v16 (parecer M1): is_relative_to em vez de startswith
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise HTTPException(400, detail="Caminho invalido")
    # Não permitir download arbitrário de código-fonte, .env ou outros arquivos do projeto.
    allowed_roots = [(ROOT / "data" / "api_artifacts").resolve(), (ROOT / "data" / "boundaries").resolve()]
    if not any(target.is_relative_to(base) for base in allowed_roots):
        raise HTTPException(403, detail="Tipo de artefato nao permitido para download")
    if target.suffix.lower() not in {".pdf", ".json", ".geojson"}:
        raise HTTPException(403, detail="Extensao de artefato nao permitida")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, detail="Arquivo ausente")
    media = "application/pdf" if target.suffix == ".pdf" else "application/json"
    if target.suffix == ".geojson":
        media = "application/geo+json"
    return FileResponse(target, media_type=media, filename=target.name)


# REMOVIDO da superfície da API: @app.post("/v1/project")
# @app.post("/v1/project")
def project_scenario(req: ProjectRequest, request: Request):
    """
    Recebe cenario de projecao para X anos, le o modelo treinado,
    executa a projecao e retorna os 5 quadrados mais criticos com
    estatisticas, latitude/longitude e URL do Google Maps.
    """
    _check_api_key(request)  # v16 (parecer M1)
    ibge = _resolve_ibge(req.city, req.ibge_code)
    # garante que o horizonte pedido esteja na corrida
    horizons = sorted(set([3, 5, 8, 10, req.years]))
    result = _run_analysis(ibge, horizons, req.pressure)
    payload = _top_critical_cells(result, years=req.years, top_n=req.top_n)
    return payload





def _allocate_annual_loss(requested_ha: float, indices: List[int], remaining: Dict[int, float], weights: Dict[int, float]) -> Dict[int, float]:
    """Distribui perda anual proporcionalmente ao risco, respeitando capacidade restante."""
    allocations = {i: 0.0 for i in indices}
    active = [i for i in indices if remaining.get(i, 0.0) > 1e-9]
    left = min(max(0.0, float(requested_ha)), sum(max(0.0, remaining.get(i, 0.0)) for i in active))
    # Iterações redistribuem o excedente das células que atingem seu teto.
    for _ in range(len(active) + 1):
        if left <= 1e-7 or not active:
            break
        weight_sum = sum(max(0.0, weights.get(i, 0.0)) for i in active)
        if weight_sum <= 0:
            local_weights = {i: 1.0 for i in active}
            weight_sum = float(len(active))
        else:
            local_weights = {i: max(0.0, weights.get(i, 0.0)) for i in active}
        allocated_round = 0.0
        saturated = []
        for i in active:
            share = left * local_weights[i] / weight_sum
            capacity = max(0.0, remaining.get(i, 0.0) - allocations[i])
            take = min(share, capacity)
            allocations[i] += take
            allocated_round += take
            if capacity - take <= 1e-7:
                saturated.append(i)
        left -= allocated_round
        active = [i for i in active if i not in saturated]
        if allocated_round <= 1e-9:
            break
    return allocations


@app.post("/v1/simulate/annual")
def simulate_annual(req: AnnualSimulationRequest, request: Request):
    """Simula a perda anual, atualiza o remanescente de cada célula e aplica
    Feedback Dinâmico de Vizinhança (v23).

    A série anual municipal vem do Ridge autorregressivo existente.
    A alocação espacial usa scores relativos do mapa-base como base_weight.
    Quando dynamic_neighborhood=True (padrão), após cada ano o algoritmo
    recalcula forest_neighbor_frac a partir dos vizinhos k-NN ainda florestados
    e ajusta os pesos de risco (efeito de borda).
    """
    _check_api_key(request)
    ibge = _resolve_ibge(None, req.ibge_code)
    horizons = sorted(set([3, 5, 8, 10, req.years]))
    result = _run_analysis(ibge, horizons, req.pressure)
    projection = result.get("municipal_projection") or {}
    annual_source = projection.get("annual_projections_ha") or {}
    base_year = int(projection.get("base_year") or 0)
    if not base_year or not annual_source:
        raise HTTPException(422, detail="A série anual de projeção não está disponível para este município.")

    cells = result.get("cells") or []
    inside = [c for c in cells if c.get("is_inside_boundary") and float(c.get("initial_forest_ha") or 0) > 0]
    if not inside:
        raise HTTPException(422, detail="Não há células internas com capacidade florestal inicial para simular.")
    indices = list(range(len(inside)))
    remaining = {i: max(0.0, float(inside[i].get("initial_forest_ha") or 0)) for i in indices}
    max_key = f"{req.years}y"
    base_weights: Dict[int, float] = {}
    for i, cell in enumerate(inside):
        scenario = (cell.get("scenarios") or {}).get(max_key) or {}
        try:
            base_weights[i] = max(0.0, float(scenario.get("probability", 0.0)))
        except (TypeError, ValueError):
            base_weights[i] = 0.0
    weights = dict(base_weights)

    # Grafo de vizinhança k-NN a partir dos centroides (uma vez)
    centroids = []
    for cell in inside:
        c = cell.get("centroid") or [0.0, 0.0]
        try:
            centroids.append((float(c[0]), float(c[1])))
        except (TypeError, ValueError, IndexError):
            centroids.append((0.0, 0.0))
    neighbors = _build_knn_neighbors(centroids, k=req.neighborhood_k) if req.dynamic_neighborhood else [[] for _ in indices]

    total_initial = sum(remaining.values())
    annual_rows = []
    cumulative_loss = 0.0
    final_cell_losses = {i: 0.0 for i in indices}
    for step in range(1, req.years + 1):
        year = base_year + step
        try:
            requested_loss = max(0.0, float(annual_source.get(str(year), 0.0)))
        except (TypeError, ValueError):
            requested_loss = 0.0
        # Evita simular perda acima de 95% do estoque inicial no horizonte total.
        remaining_budget = max(0.0, total_initial * 0.95 - cumulative_loss)
        requested_capped = min(requested_loss, remaining_budget)
        alloc = _allocate_annual_loss(requested_capped, indices, remaining, weights)
        allocated = sum(alloc.values())
        for i, value in alloc.items():
            remaining[i] = max(0.0, remaining[i] - value)
            final_cell_losses[i] += value
        cumulative_loss += allocated

        # --- Feedback Dinâmico de Vizinhança (após o desmate do ano) ---
        if req.dynamic_neighborhood:
            weights = _recompute_weights_dynamic(
                base_weights=base_weights,
                remaining=remaining,
                neighbors=neighbors,
                beta=req.edge_effect_beta,
            )

        row = {
            "year": year,
            "requested_loss_ha": round(requested_loss, 2),
            "allocated_loss_ha": round(allocated, 2),
            "unallocated_loss_ha": round(max(0.0, requested_loss - allocated), 2),
            "cumulative_loss_ha": round(cumulative_loss, 2),
            "remaining_forest_ha": round(sum(remaining.values()), 2),
        }
        ranked = sorted(indices, key=lambda i: weights.get(i, 0.0), reverse=True)[:req.top_n]
        row["top_cells"] = [{
            "cell_id": inside[i].get("cell_id"),
            "latitude": (inside[i].get("centroid") or [None, None])[0],
            "longitude": (inside[i].get("centroid") or [None, None])[1],
            "risk_score_relative": round(weights.get(i, 0.0), 6),
            "loss_this_year_ha": round(alloc.get(i, 0.0), 4),
            "remaining_forest_ha": round(remaining[i], 2),
        } for i in ranked]
        if req.include_cell_maps:
            row["cell_map"] = [{
                "cell_id": inside[i].get("cell_id"),
                "latitude": (inside[i].get("centroid") or [None, None])[0],
                "longitude": (inside[i].get("centroid") or [None, None])[1],
                "risk_score_relative": round(weights.get(i, 0.0), 6),
                "loss_this_year_ha": round(alloc.get(i, 0.0), 4),
                "cumulative_loss_ha": round(final_cell_losses[i], 4),
                "remaining_forest_ha": round(remaining[i], 2),
            } for i in indices]
        annual_rows.append(row)

    return {
        "ibge_code": ibge,
        "municipality": (result.get("municipality") or {}).get("name"),
        "base_year": base_year,
        "target_year": base_year + req.years,
        "horizon_years": req.years,
        "method": "annual_ridge_projection_with_sequential_spatial_allocation_and_dynamic_neighborhood_feedback",
        "model_status": "experimental_with_dynamic_neighborhood_v23",
        "scenario": {"pressure_multiplier": req.pressure},
        "spatial_allocation": {
            "weights_source": "base_year_spatial_scores_adjusted_by_dynamic_neighborhood",
            "risk_weights_change_each_year": bool(req.dynamic_neighborhood),
            "dynamic_neighborhood_enabled": bool(req.dynamic_neighborhood),
            "neighborhood_k": req.neighborhood_k if req.dynamic_neighborhood else None,
            "edge_effect_beta": req.edge_effect_beta if req.dynamic_neighborhood else None,
            "forest_capacity_updated_each_year": True,
            "initial_forest_ha_is_uniformly_distributed_over_inside_cells": True,
        },
        "summary": {
            "initial_forest_ha_in_grid": round(total_initial, 2),
            "cumulative_loss_ha": round(cumulative_loss, 2),
            "remaining_forest_ha": round(sum(remaining.values()), 2),
            "percent_initial_forest_lost": round(100.0 * cumulative_loss / max(total_initial, 1.0), 2),
        },
        "annual_predictions": annual_rows,
        "final_cells": [{
            "cell_id": inside[i].get("cell_id"),
            "latitude": (inside[i].get("centroid") or [None, None])[0],
            "longitude": (inside[i].get("centroid") or [None, None])[1],
            "cumulative_loss_ha": round(final_cell_losses[i], 2),
            "remaining_forest_ha": round(remaining[i], 2),
            "risk_score_relative": round(weights.get(i, 0.0), 6),
        } for i in indices],
        "limitations": [
            "A série municipal anual é projetada pelo modelo Ridge autorregressivo treinado nas séries históricas disponíveis; não é um XGBoost anual.",
            "Os pesos espaciais de base vêm do mapa de risco do horizonte solicitado; com dynamic_neighborhood=True eles são ajustados anualmente pelo efeito de borda (forest_neighbor_frac recalculado via k-NN).",
            "O ajuste dinâmico é heurístico (fator linear sobre o score base) e não re-executa o XGBoost completo a cada ano.",
            "O estoque inicial por célula é distribuído uniformemente quando não existe raster observado de capacidade.",
            "Horizontes longos (15–50 anos) permanecem experimentais até validação walk-forward histórica (FoM/IoU).",
        ],
    }


@app.post("/v1/predict")
def predict_with_trained_model(req: TrainedPredictionRequest, request: Request):
    """Executa o XGBoost salvo sobre features observadas do painel mais recente.

    Alvo principal: y_loss_1y (previsão anual).
    """
    _check_api_key(request)
    ibge = _resolve_ibge(None, req.ibge_code)
    years = int(req.years)
    if not 1 <= years <= 20:
        raise HTTPException(400, detail="years deve estar entre 1 e 20.")
    # Padrão: modelo anual y_loss_1y para qualquer horizonte 1–20.
    requested_target = (req.target or "").strip() or "y_loss_1y"
    if requested_target not in {"y_loss_1y", "y_loss_3y"}:
        raise HTTPException(
            400, detail="target deve ser y_loss_1y."
        )
    if requested_target == "y_loss_3y" and years != 3:
        raise HTTPException(
            400,
            detail="target=y_loss_3y (legado) exige years=3.",
        )
    try:
        cached = model_cache.get_or_load_model(ibge)
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger("forestguard.api").exception("Falha ao carregar modelo espacial")
        raise HTTPException(500, detail="Não foi possível carregar o modelo treinado.") from exc

    model = cached.model
    feature_names = cached.feature_names
    meta = cached.meta
    model_path = cached.model_path

    if str(meta.get("ibge_code") or "").strip() not in ("", ibge):
        raise HTTPException(404, detail="O manifesto do modelo não corresponde ao código IBGE solicitado.")
    model_target = str(meta.get("target_col") or "")
    if model_target != requested_target:
        raise HTTPException(
            409,
            detail=(
                f"O modelo disponível está registrado para '{model_target}', "
                f"mas a requisição pediu '{requested_target}'. "
                "Treine ou selecione um modelo compatível."
            ),
        )
    panel_dir = ROOT / "data" / "spatial" / "panels" / ibge
    if not panel_dir.is_dir():
        raise HTTPException(404, detail=f"Painel de features não encontrado para o IBGE {ibge}.")
    horizon = int(meta.get("horizon_years") or (3 if requested_target == "y_loss_3y" else 1))
    import numpy as np
    try:
        prepared = model_cache.get_or_prepare_features(
            ibge,
            feature_names=feature_names,
            horizon_years=horizon,
            panel_dir=panel_dir,
            target_year=None,
            require_forest_t0=True,
        )
    except Exception as exc:
        logging.getLogger("forestguard.api").exception("Falha ao preparar features de inferência")
        raise HTTPException(422, detail=f"Não foi possível preparar features: {exc}") from exc
    X = prepared["X"]
    meta_rows = prepared["meta"]
    skipped = prepared.get("skipped") or []
    latest_year = prepared["panel_year"]
    if X.shape[0] == 0:
        raise HTTPException(422, detail="Nenhuma célula possui todas as features observadas exigidas pelo modelo.")
    try:
        probabilities = _positive_class_probabilities(model, X)
    except Exception as exc:
        logging.getLogger("forestguard.api").exception("Falha na inferência do XGBoost")
        raise HTTPException(500, detail="O modelo treinado falhou ao calcular as probabilidades.") from exc
    prob_key = f"probability_{requested_target}"
    def _finite(v):
        try:
            x = float(v)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    results = []
    for row, probability in zip(meta_rows, probabilities):
        t0 = int(row.get("year_t0") or latest_year)
        lat = row.get("lat")
        lon = row.get("lon")
        gx = _finite(row.get("grid_x_m"))
        gy = _finite(row.get("grid_y_m"))
        lat_f = _finite(lat)
        lon_f = _finite(lon)
        results.append({
            "cell_id": row.get("cell_id") or row.get("source_file"),
            "year_t0": t0,
            "target_year": t0 + horizon,
            "latitude": lat_f,
            "longitude": lon_f,
            "grid_x_m": int(gx) if gx is not None else None,
            "grid_y_m": int(gy) if gy is not None else None,
            "google_maps_url": _google_maps_url(lat_f, lon_f) if lat_f is not None and lon_f is not None else None,
            prob_key: round(float(probability), 6),
            "probability": round(float(probability), 6),
            "predicted_class": int(float(probability) >= 0.5),
        })
    results.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "ibge_code": ibge,
        "model_type": meta.get("model_type"),
        "model_path": str(model_path),
        "target": meta.get("target_col"),
        "target_horizon_years": horizon,
        "panel_year_used": latest_year,
        "target_year": latest_year + horizon,
        "panel_schema": "v2",
        "feature_names": feature_names,
        "cells_scored": len(results),
        "cells_skipped_missing_features": len(skipped),
        "top_predictions": results[:req.top_n],
        "skipped_cells_sample": skipped[:20],
        "metrics_test": meta.get("metrics_test"),
        "walk_forward": meta.get("walk_forward"),
        "feature_importance": meta.get("feature_importance"),
        "warning": (
            f"Probabilidade para o alvo {requested_target} (horizonte {horizon} ano(s)). "
            "Painel schema v2 (cell_id M{{IBGE}}_x_y; built_dist_km/frações opcionais). "
            "Não é previsão de desmatamento confirmado nem probabilidade calibrada operacional."
        ),
    }


# REMOVIDO da superfície da API: @app.get("/v1/metrics/{ibge_code}")
# @app.get("/v1/metrics/{ibge_code}")
def model_metrics(ibge_code: str, request: Request):
    """Retorna em JSON as métricas registradas no manifesto do modelo treinado."""
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    candidates = [
        ROOT / "data" / "spatial" / "models" / f"{ibge}_annual" / "spatial_model_meta.json",
        ROOT / "data" / "spatial" / "models" / ibge / "spatial_model_meta.json",
    ]
    # O artefato na raiz só pode ser usado quando o próprio manifesto identifica
    # explicitamente o mesmo município; nunca cruzar métricas entre municípios.
    candidates.append(ROOT / "data" / "spatial" / "models" / "spatial_model_meta.json")
    meta_path = None
    meta = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            candidate_meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_code = str(candidate_meta.get("ibge_code") or "").strip()
        parent = candidate.parent.name
        parent_ok = parent == ibge or parent == f"{ibge}_annual" or parent.replace("_annual", "") == ibge
        if parent_ok or model_code == ibge:
            meta_path, meta = candidate, candidate_meta
            break
    if meta is None or meta_path is None:
        raise HTTPException(404, detail=f"Não há métricas de modelo treinado registradas para o IBGE {ibge}.")
    return {
        "ibge_code": ibge,
        "model_type": meta.get("model_type"),
        "model_class": meta.get("model_class"),
        "target": meta.get("target_col"),
        "horizon_years": meta.get("horizon_years"),
        "panel_schema": meta.get("panel_schema") or "v2",
        "features": meta.get("feature_names", []),
        "metrics_test": meta.get("metrics_test"),
        "metrics_valid": meta.get("metrics_valid"),
        "walk_forward": meta.get("walk_forward"),
        "feature_importance": meta.get("feature_importance", []),
        "training_note": meta.get("training_note"),
        "calibration_note": meta.get("calibration_note"),
        "project_version": meta.get("project_version"),
        "metadata_file": str(meta_path.relative_to(ROOT)),
        "warning": (
            "Métricas do painel de teste/validação (schema v2). "
            "Não garantem desempenho futuro; células sem features essenciais não entram no score."
        ),
    }





# REMOVIDO da superfície da API: @app.get("/v1/models/loaded")
# @app.get("/v1/models/loaded")
def models_loaded(request: Request):
    """Lista modelos e features atualmente em memória (preload / cache)."""
    _check_api_key(request)
    return model_cache.status()


# REMOVIDO da superfície da API: @app.post("/v1/models/preload/{ibge_code}")
# @app.post("/v1/models/preload/{ibge_code}")
def models_preload(ibge_code: str, request: Request, features: bool = Query(True)):
    """Carrega (ou recarrega) modelo + features do IBGE na memória sem predizer."""
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    try:
        return model_cache.preload_ibge(ibge, also_features=features)
    except Exception as exc:
        logging.getLogger("forestguard.api").exception("Preload sob demanda falhou")
        raise HTTPException(500, detail=f"Falha ao pré-carregar IBGE {ibge}: {exc}") from exc

@app.get("/v1/dashboard/{ibge_code}")
def ml_dashboard_json(
    ibge_code: str,
    request: Request,
    cost_fp: float = Query(1.0, ge=0.0, description="Custo de falso positivo (alarme falso)"),
    cost_fn: float = Query(5.0, ge=0.0, description="Custo de falso negativo (desmate omitido)"),
):
    """Dashboard ML em JSON: desmatamento anual, métricas, importância e detalhes XGBoost.

    Query opcional: cost_fp, cost_fn — recalcula limiar ótimo e custo esperado
    (default FN:FP = 5:1).
    """
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    try:
        payload = build_dashboard_payload(ibge, cost_fp=cost_fp, cost_fn=cost_fn)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if not payload.get("metrics", {}).get("test") and not payload.get("deforestation", {}).get("panel_y_loss_1y_by_year"):
        raise HTTPException(
            404,
            detail=f"Sem painel ou modelo suficiente para montar o dashboard do IBGE {ibge}.",
        )
    return payload


@app.get("/v1/dashboard/{ibge_code}/view")
def ml_dashboard_html(
    ibge_code: str,
    request: Request,
    cost_fp: float = Query(1.0, ge=0.0, description="Custo de falso positivo (alarme falso)"),
    cost_fn: float = Query(5.0, ge=0.0, description="Custo de falso negativo (desmate omitido)"),
):
    """Dashboard ML interativo (HTML + Chart.js). Aceita cost_fp e cost_fn na query."""
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    try:
        payload = build_dashboard_payload(ibge, cost_fp=cost_fp, cost_fn=cost_fn)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if not payload.get("metrics", {}).get("test") and not payload.get("deforestation", {}).get("panel_y_loss_1y_by_year"):
        raise HTTPException(
            404,
            detail=f"Sem painel ou modelo suficiente para montar o dashboard do IBGE {ibge}.",
        )
    return HTMLResponse(content=render_dashboard_html(payload))


# REMOVIDO da superfície da API: @app.get("/v1/report/{ibge_code}")
# @app.get("/v1/report/{ibge_code}")
def technical_report(ibge_code: str, request: Request,
                     years: int = Query(10, ge=1, le=50),
                     pressure: float = Query(1.0, ge=0.5, le=2.0)):
    """Gera o mesmo tipo de informe PDF usado pela interface desktop."""
    _check_api_key(request)
    ibge = _resolve_ibge(None, ibge_code)
    horizons = sorted(set([3, 5, 8, 10, 20, years]))
    payload = _ensure_analysis_artifacts(ibge, horizons, pressure)
    rel = (payload.get("artifacts") or {}).get("pdf")
    if not rel:
        raise HTTPException(500, detail=f"Falha ao gerar informe técnico: {payload.get('pdf_error') or 'erro não especificado'}")
    path = (ROOT / rel).resolve()
    if not path.is_relative_to((ROOT / "data" / "api_artifacts").resolve()) or not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(500, detail="O caminho do informe gerado não passou na validação de segurança.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)

def _validated_job_artifact_path(rel: str, allowed_suffixes: set[str]) -> Path:
    """Resolve um artefato de job apenas dentro das pastas de dados autorizadas."""
    candidate = (ROOT / rel).resolve()
    allowed = [(ROOT / "data" / "jobs").resolve(), (ROOT / "data" / "api_artifacts").resolve()]
    if not any(candidate.is_relative_to(base) for base in allowed):
        raise HTTPException(403, detail="Caminho de artefato nao permitido")
    if candidate.suffix.lower() not in allowed_suffixes:
        raise HTTPException(403, detail="Extensao de artefato nao permitida")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(404, detail="Arquivo ausente")
    return candidate


# ---------------------------------------------------------------------------
# Jobs legados (compatibilidade)
# ---------------------------------------------------------------------------

if HAS_JOBS:
    from concurrent.futures import ThreadPoolExecutor

    # v16 (parecer M8): executor single-worker — jobs rodam fora do ciclo
    # request/response; a API continua responsiva durante análises longas.
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fg-job")

    def _run_job_safe(job_id: str) -> None:
        try:
            run_job(job_id)
        except Exception:
            logging.getLogger("forestguard.jobs").exception("Job %s falhou com excecao nao tratada", job_id)

    class JobRequest(BaseModel):
        city: Optional[str] = None
        ibge_code: Optional[str] = None
        municipality: Optional[str] = None
        pressure: float = 1.0
        horizons: List[int] = Field(default_factory=lambda: [3, 5, 8, 10, 20])
        grid_rows: Optional[int] = None
        grid_cols: Optional[int] = None
        auto_add_city: bool = True
        prodes_json: Optional[Union[Dict[str, float], str]] = None
        forest_remnant_ha: Optional[float] = None
        spatial_panel_path: Optional[str] = None

    @app.post("/v1/jobs")
    def jobs_create(req: JobRequest, request: Request):
        _check_api_key(request)  # v16 (parecer M1): mutante exige API-key quando configurada
        rec = create_job(req.model_dump())
        # v16 (parecer M8): execucao em background - responde imediatamente com status queued.
        # Status real disponivel em GET /v1/jobs/{job_id}.
        _executor.submit(_run_job_safe, rec.job_id)
        return _job_response(rec)

    @app.get("/v1/jobs")
    def jobs_list(request: Request):
        _check_api_key(request)
        return {"jobs": [_job_response(r) for r in list_jobs()]}

    @app.get("/v1/jobs/{job_id}")
    def jobs_get(job_id: str, request: Request):
        _check_api_key(request)
        rec = load_job(job_id)
        if not rec:
            raise HTTPException(404, detail="Job nao encontrado")
        return _job_response(rec)

    @app.get("/v1/jobs/{job_id}/result")
    def jobs_result(job_id: str, request: Request):
        _check_api_key(request)
        rec = load_job(job_id)
        if not rec:
            raise HTTPException(404, detail="Job nao encontrado")
        if rec.status != "succeeded":
            raise HTTPException(409, detail=f"Job em status {rec.status}")
        rel = rec.artifacts.get("analysis_json")
        if not rel:
            raise HTTPException(404, detail="Artefato analysis.json ausente")
        path = _validated_job_artifact_path(rel, {".json"})
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/v1/jobs/{job_id}/pdf")
    def jobs_pdf(job_id: str, request: Request):
        _check_api_key(request)
        rec = load_job(job_id)
        if not rec:
            raise HTTPException(404, detail="Job nao encontrado")
        if rec.status != "succeeded":
            raise HTTPException(409, detail=f"Job em status {rec.status}")
        rel = rec.artifacts.get("pdf")
        if not rel:
            raise HTTPException(404, detail="PDF ausente")
        path = _validated_job_artifact_path(rel, {".pdf"})
        if not path.exists():
            raise HTTPException(404, detail="Arquivo PDF nao encontrado")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    @app.get("/v1/jobs/{job_id}/artifacts/{name}")
    def jobs_artifact(job_id: str, name: str, request: Request):
        _check_api_key(request)
        rec = load_job(job_id)
        if not rec:
            raise HTTPException(404, detail="Job nao encontrado")
        rel = rec.artifacts.get(name)
        if not rel:
            raise HTTPException(404, detail=f"Artefato '{name}' nao existe")
        path = _validated_job_artifact_path(rel, {".pdf", ".json", ".geojson"})
        if not path.exists():
            raise HTTPException(404, detail="Arquivo ausente")
        media = "application/pdf" if path.suffix == ".pdf" else "application/json"
        if path.suffix == ".geojson":
            media = "application/geo+json"
        return FileResponse(path, media_type=media, filename=path.name)

    def _job_response(rec) -> Dict[str, Any]:
        base = {
            "job_id": rec.job_id,
            "status": rec.status,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "ibge_code": rec.ibge_code,
            "municipality_name": rec.municipality_name,
            "request": rec.request,
            "artifacts": rec.artifacts,
            "summary": rec.result_summary,
            "error": rec.error,
            "links": {},
        }
        if rec.status == "succeeded" and rec.job_id:
            jid = rec.job_id
            base["links"] = {
                "self": f"/v1/jobs/{jid}",
                "result": f"/v1/jobs/{jid}/result",
                "pdf": f"/v1/jobs/{jid}/pdf",
                "artifacts": {k: f"/v1/jobs/{jid}/artifacts/{k}" for k in rec.artifacts},
            }
        return base
