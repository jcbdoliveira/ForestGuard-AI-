"""
main.py — ponto de entrada da API ForestGuard (Fase 2 do edital SCTEC).

Arquitetura em camadas:
  schemas.py  — contratos Pydantic (entrada/saída)
  services.py — carga do modelo em RAM + inferência
  routers.py  — APIRouter (/health, /predict)
  model_cache.py — cache de painel/modelo/análise (lifespan)
  app.py      — rotas estendidas (analyze, dashboard, pdf, …)
  main.py     — este arquivo: exporta `app` para o Uvicorn

Uso:
  uvicorn supressao_vegetacao.api.main:app --host 0.0.0.0 --port 8080
  python scripts/run_api.py
"""
from __future__ import annotations

# Reexporta a aplicação FastAPI já configurada (lifespan, CORS, allowlist, rotas).
from supressao_vegetacao.api.app import app  # noqa: F401

__all__ = ["app"]
