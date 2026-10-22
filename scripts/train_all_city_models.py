#!/usr/bin/env python3
"""
Treina um modelo espacial XGBoost **por município**.

Por padrão treina apenas municípios que possuem painel GEE/MapBiomas real
em data/spatial/panels/{IBGE}. Isso evita erros em massa após a remoção
de cidades sem dados completos do pacote.

Uso:
  python3 scripts/train_all_city_models.py
  python3 scripts/train_all_city_models.py --ibge 4210100
  python3 scripts/train_all_city_models.py --all-catalog   # tenta todo o catálogo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from supressao_vegetacao.sources.municipality import list_municipalities  # noqa: E402
from supressao_vegetacao.sources.mapbiomas_catalog import mapbiomas_series_for_prodes  # noqa: E402
from supressao_vegetacao.sources.real_prodes_data import REAL_HISTORICAL_DATA  # noqa: E402
from supressao_vegetacao.spatial.train_city_model import train_city_spatial_model  # noqa: E402


def mean_rate_for(ibge: str) -> float:
    mb = mapbiomas_series_for_prodes(ibge)
    if mb and mb.get("mean_annual_deforestation_rate_pct") is not None:
        return float(mb["mean_annual_deforestation_rate_pct"]) / 100.0
    rec = REAL_HISTORICAL_DATA.get(str(ibge)) or {}
    if rec.get("mean_annual_deforestation_rate_pct") is not None:
        return float(rec["mean_annual_deforestation_rate_pct"]) / 100.0
    hist = rec.get("historic_prodes_deforestation_ha") or {}
    remnant = float(rec.get("official_forest_remnant_ha") or 0)
    if hist and remnant > 0:
        vals = [float(v) for v in hist.values()]
        return (sum(vals) / len(vals)) / remnant
    return 0.005


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Treina modelos espaciais XGBoost. Por padrão só municípios com painel GEE real."
    )
    ap.add_argument("--ibge", default=None, help="Treinar só este código IBGE")
    ap.add_argument(
        "--all-catalog",
        action="store_true",
        help="Inclui cidades do catálogo sem painel real (gerará erro ou exigirá allow_synthetic_dev)",
    )
    args = ap.parse_args()
    if args.ibge:
        codes = [str(args.ibge).strip()]
    else:
        # Default: apenas municípios com painel real — evita erros em massa
        # depois que outras cidades foram removidas do pacote de dados.
        codes = [
            str(m["ibge_code"])
            for m in list_municipalities(only_with_panel=not args.all_catalog)
        ]
    if not codes:
        print("Nenhum município com painel real encontrado. Use --ibge CODE ou --all-catalog.")
        return 1
    print(f"Treinando {len(codes)} modelo(s) espacial(is) por cidade...")
    ok, fail = 0, 0
    for code in codes:
        rate = mean_rate_for(code)
        try:
            tr = train_city_spatial_model(code, mean_loss_rate=rate)
            auc = (tr.get("metrics_test") or {}).get("roc_auc")
            print(f"  ✓ {code} rate={rate:.4f} AUC={auc} → {tr['model_dir']}")
            ok += 1
        except Exception as exc:
            print(f"  ✗ {code}: {exc}")
            fail += 1
    print(f"\nConcluído: {ok} ok, {fail} falha(s).")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
