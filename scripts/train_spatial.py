#!/usr/bin/env python3
"""
Treina modelo espacial (XGBoost) com split temporal e reporta AUC / PR-AUC reais.

Uso:
  # Após exportar CSVs do Google Earth Engine (scripts/gee/export_municipality_panel.js)
  python3 scripts/train_spatial.py --panel data/spatial/panels/4210100 --target y_loss_1y

  python3 scripts/train_spatial.py --panel data/spatial/panels/4210100 --target y_loss_3y  # legado

O painel DEVE conter rótulos observados (y_loss_*) derivados de MapBiomas/PRODES.
Não inventamos desmatamento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supressao_vegetacao.spatial.panel import load_panel_csv
from supressao_vegetacao.spatial.train_spatial import train_spatial_model


def main() -> int:
    p = argparse.ArgumentParser(description="Treino espacial XGBoost com métricas reais (AUC/PR-AUC)")
    p.add_argument("--panel", required=True, help="CSV único ou diretório com CSVs do painel célula×ano")
    p.add_argument("--target", default="y_loss_1y", help="Coluna alvo (y_loss_1y — principal; y_loss_3y legado)")
    p.add_argument("--test-year", type=int, default=None, help="Ano de teste (default: último do painel)")
    p.add_argument("--valid-year", type=int, default=None, help="Ano de validação (default: penúltimo)")
    p.add_argument("--model-dir", default=str(ROOT / "data" / "spatial" / "models"), help="Onde salvar modelo")
    p.add_argument("--no-xgboost", action="store_true", help="Força sklearn GradientBoosting (não suporta estradas ausentes sem imputação)")
    p.add_argument("--no-walk-forward", action="store_true", help="Desativa walk-forward temporal completo")
    args = p.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        print(
            f"ERRO: painel não encontrado: {panel_path}\n"
            "Gere os CSVs com scripts/gee/export_municipality_panel.js no Google Earth Engine\n"
            "ou forneça um CSV com colunas year_t0, y_loss_*, slope_deg, elev_m, forest_neighbor_frac, ...",
            file=sys.stderr,
        )
        return 1

    print(f"→ Carregando painel: {panel_path}")
    panel = load_panel_csv(panel_path, target_col=args.target)
    print(f"✓ Amostras: {len(panel.y)} | Features: {panel.feature_names}")
    print(f"  Anos: {sorted(set(int(y) for y in panel.years))}")
    print(f"  Taxa positiva (global): {panel.y.mean():.4f}")

    model, result = train_spatial_model(
        panel,
        target_col=args.target,
        test_year=args.test_year,
        valid_year=args.valid_year,
        model_dir=args.model_dir,
        prefer_xgboost=not args.no_xgboost,
        run_walk_forward=not args.no_walk_forward,
    )

    print("\n=== RESULTADO DO TREINO ESPACIAL ===")
    print(f"Modelo: {result.model_type}")
    print(f"Treino anos: {result.train_years} | corte treino={result.train_cutoff_year} | valid={result.valid_year} | test={result.test_year} | horizonte={result.horizon_years}y")
    print(f"n_train={result.n_train} n_valid={result.n_valid} n_test={result.n_test}")
    print("\nMétricas no TESTE (split temporal — não inventadas):")
    for k, v in result.metrics_test.items():
        print(f"  {k}: {v}")
    if result.metrics_valid:
        print("\nMétricas na VALIDAÇÃO:")
        for k, v in result.metrics_valid.items():
            print(f"  {k}: {v}")
    print("\nImportância das features:")
    print(f"Features usadas no treino: {result.feature_names}")
    for item in result.feature_importance:
        print(f"  {item['feature']}: {item['importance']}")
    if result.walk_forward:
        print("\n=== WALK-FORWARD TEMPORAL (EMBARGO DE HORIZONTE) ===")
        print(f"Cortes: {result.walk_forward.get('n_cutoffs')} | mediana ROC-AUC: {result.walk_forward.get('median_roc_auc')}")
        for cutoff in result.walk_forward.get("by_cutoff", []):
            print(f"  teste={cutoff['test_year']} treino_ate={cutoff['train_cutoff_year']} ROC-AUC={cutoff['metrics_model'].get('roc_auc')} PR-AUC={cutoff['metrics_model'].get('pr_auc')}")
    if result.model_path:
        print(f"\n✓ Modelo/metadata: {result.model_path}")
    wf_summary = {k: result.walk_forward.get(k) for k in ("n_cutoffs", "cutoffs", "median_roc_auc", "horizon_years", "label_embargo")} if result.walk_forward else {}
    print(json.dumps({"metrics_test": result.metrics_test, "walk_forward_summary": wf_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
