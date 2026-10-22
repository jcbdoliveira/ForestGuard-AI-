"""
schemas.py — contratos Pydantic de entrada e saída (Fase 4 do edital SCTEC).

Validação estrita com Field (ge/le/pattern/descrições).
Payloads inválidos geram HTTP 422 automaticamente pelo FastAPI/Pydantic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    service: str
    version: str
    model_in_memory: bool = Field(..., description="True se o artefato ML está em RAM")
    models_loaded: List[Dict[str, Any]] = Field(default_factory=list)
    analyses_cached: List[Dict[str, Any]] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Corpo de POST /v1/predict e POST /predict."""

    ibge_code: str = Field(
        ...,
        min_length=7,
        max_length=7,
        pattern=r"^\d{7}$",
        description="Código IBGE de 7 dígitos (ex.: 4210100 = Mafra/SC)",
        examples=["4210100"],
    )
    years: int = Field(
        1,
        ge=1,
        le=20,
        description="Horizonte em anos (1–20). Modelo principal: y_loss_1y",
    )
    top_n: int = Field(
        20,
        ge=1,
        le=500,
        description="Quantidade de células retornadas ordenadas por probabilidade",
    )
    target: Optional[str] = Field(
        None,
        description="Alvo explícito: y_loss_1y",
    )

    @field_validator("target")
    @classmethod
    def _check_target(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.strip()
        if v not in {"y_loss_1y", "y_loss_3y"}:
            raise ValueError("target deve ser y_loss_1y")
        return v


class CellPrediction(BaseModel):
    rank: int
    cell_id: Optional[str] = None
    probability: float = Field(..., ge=0.0, le=1.0)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PredictResponse(BaseModel):
    ibge_code: str
    target: str
    years: int
    model_path: Optional[str] = None
    model_in_memory: bool = True
    n_cells_scored: int = 0
    top_cells: List[Dict[str, Any]] = Field(default_factory=list)
    metrics_test: Optional[Dict[str, Any]] = None
    disclaimer: str = (
        "Risco relativo estimado. Não constitui desmatamento consumado nem auto de infração."
    )


class AnalyzeRequest(BaseModel):
    city: Optional[str] = Field(None, description="NOME-UF ou código IBGE")
    ibge_code: Optional[str] = Field(None, description="Código IBGE 7 dígitos")
    pressure: float = Field(1.0, ge=0.5, le=2.0)
    horizons: List[int] = Field(
        default_factory=lambda: [1, 3, 5, 8, 10, 20],
        description="Horizontes em anos (cada um entre 1 e 20)",
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
        return list(dict.fromkeys(int(x) for x in v))
