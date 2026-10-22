#!/usr/bin/env python3
"""
CLI Principal em Python: Análise de Pontos de Maior Atenção e Projeção ML (3, 5, 8, 10 e 20 Anos).

Uso via terminal:
  python3 scripts/run_inference.py --ibge-code 4210100 --horizons 3 5 8 10 20
  python3 scripts/run_inference.py --list-cities
  python3 scripts/run_inference.py --ibge-code 4210100 --json
"""

import argparse
import json
import sys
from pathlib import Path

# Garante que o diretório src/ esteja no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supressao_vegetacao.sources.municipality import list_municipalities
from supressao_vegetacao.inference.predict import run_city_deforestation_analysis
from supressao_vegetacao.settings import SUPPORTED_HORIZONS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Análise espacial e projeção ML (horizontes 1–50 anos; recomendados 3 5 8 10 20)."
    )
    parser.add_argument("--ibge-code", type=str, default="4210100", help="Código IBGE do município (ex: 4210100 para Mafra/SC)")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(SUPPORTED_HORIZONS),
        help="Horizontes em anos para projeção ML (padrão: 3 5 8 10 20 20)"
    )
    parser.add_argument("--pressure", type=float, default=1.0, help="Multiplicador de pressão antrópica do cenário (0.5 a 2.0)")
    parser.add_argument("--list-cities", action="store_true", help="Lista os municípios disponíveis no catálogo")
    parser.add_argument("--json", action="store_true", help="Imprime a saída completa em JSON na stdout")
    parser.add_argument("--output", type=str, default=None, help="Caminho opcional para salvar o JSON resultante")

    args = parser.parse_args()

    if args.list_cities:
        cities = list_municipalities()
        if args.json:
            print(json.dumps(cities, ensure_ascii=False, indent=2))
        else:
            print("=== MUNICÍPIOS HOMOLOGADOS NO CATÁLOGO ===")
            for c in cities:
                print(f"• {c['ibge_code']} — {c['name']}/{c['uf']} ({c['biome']}) | Remanescente: {c['forest_remnant_ha']} ha")
        return 0

    result = run_city_deforestation_analysis(
        ibge_code=args.ibge_code,
        horizons=args.horizons,
        custom_pressure_multiplier=args.pressure,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0

    muni = result["municipality"]
    print("=" * 84)
    print(f"DIAGNÓSTICO DE DESMATAMENTO & PROJEÇÃO ML — {muni['name']}/{muni['uf']} (IBGE {muni['ibge_code']})")
    print(f"Bioma: {muni['biome']} | Área Total: {muni['total_area_ha']} ha | Remanescente: {muni['forest_remnant_ha']} ha")
    print("=" * 84)

    for h in args.horizons:
        h_key = f"{h}y"
        summary = result["horizon_summaries"][h_key]
        print(
            f"\n[CENÁRIO {h} ANOS -> {summary['target_year']}] "
            f"Prob. Média: {summary['mean_probability']*100:.1f}% | "
            f"Pontos Críticos: {summary['critical_points_count']} | "
            f"Área Suprimida Projetada: {summary['expected_suppressed_area_ha']:.1f} ha "
            f"({summary['percent_of_remnant_lost']:.2f}% do remanescente)"
        )
        print("  Top 3 Pontos de Maior Atenção (Hotspots):")
        for pt in summary["top_attention_points"][:3]:
            print(
                f"    #{pt['rank']} {pt['cell_id']} ({pt['locality']}) -> "
                f"Risco: {pt['probability']*100:.1f}% [{pt['attention_level']}] | "
                f"Supressão est.: {pt['expected_suppressed_ha']} ha | Fator: {pt['dominant_driver']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
