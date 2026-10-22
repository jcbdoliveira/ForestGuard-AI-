#!/usr/bin/env python3
"""
Sobe a API ForestGuard.

  python3 scripts/run_api.py
  python3 scripts/run_api.py --host 0.0.0.0 --port 8080
  python3 scripts/run_api.py --preload-ibge 4210100
  python3 scripts/run_api.py --preload-ibge 4210100 --preload-ibge 4217808

Documentação Swagger: http://localhost:8080/docs

O preload (padrão 4210100 se houver painel) carrega no startup:
  - CSVs do painel + modelo XGBoost
  - features para /v1/predict
  - análise completa horizontes 1–20 para /v1/analyze
Horizontes acima de 20 anos são rejeitados.
  export FORESTGUARD_PRELOAD_IBGE=4210100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="ForestGuard HTTP API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--preload-ibge",
        action="append",
        default=[],
        metavar="IBGE",
        help="Código IBGE do município cujo modelo deve ser carregado na memória no startup. Pode repetir.",
    )
    args = parser.parse_args()

    existing = os.environ.get("FORESTGUARD_PRELOAD_IBGE", "").strip()
    codes: list[str] = []
    if existing:
        for chunk in existing.replace(";", ",").split(","):
            c = "".join(ch for ch in chunk.strip() if ch.isdigit())
            if c:
                codes.append(c)
    for raw in args.preload_ibge:
        c = "".join(ch for ch in str(raw).strip() if ch.isdigit())
        if c:
            codes.append(c)
    seen: set[str] = set()
    unique: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if unique:
        os.environ["FORESTGUARD_PRELOAD_IBGE"] = ",".join(unique)
        print(f"[API] Preload IBGE: {', '.join(unique)}")

    try:
        import uvicorn
    except ImportError:
        print("Instale: pip install fastapi uvicorn", file=sys.stderr)
        return 1

    uvicorn.run(
        "supressao_vegetacao.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(SRC),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
