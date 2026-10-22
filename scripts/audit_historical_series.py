"""Audita zeros, negativos, lacunas e coerência básica das séries MapBiomas.

Uso: python scripts/audit_historical_series.py [--output data/audit_historical_series.json]
Valores esperados no painel: hectares (ha), não km² nem proporções.
O relatório sinaliza pontos para revisão; não transforma zero em dado ausente.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "mapbiomas" / "forest_by_municipality.json"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/audit_historical_series.json")
    args = parser.parse_args()
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for code, rec in data.items():
        forest = rec.get("forest_ha_by_year") or {}
        loss = rec.get("approx_annual_forest_loss_ha") or {}
        vals = [("forest_ha_by_year", y, v) for y, v in forest.items()] + [("approx_annual_forest_loss_ha", y, v) for y, v in loss.items()]
        zeros = [(field, year) for field, year, v in vals if isinstance(v, (int, float)) and v == 0]
        negatives = [(field, year, v) for field, year, v in vals if isinstance(v, (int, float)) and v < 0]
        invalid = [(field, year, repr(v)) for field, year, v in vals if not isinstance(v, (int, float)) or not math.isfinite(float(v))]
        years_f, years_l = set(forest), set(loss)
        issues = []
        if len(forest) < 5 or len(loss) < 4: issues.append("serie_curta")
        if zeros: issues.append("valores_zero_revisar_sem_imputacao_automatica")
        if negatives: issues.append("valores_negativos")
        if invalid: issues.append("valores_nao_finitos_ou_nao_numericos")
        if rec.get("source") is None: issues.append("fonte_nao_declarada")
        rows.append({"ibge_code": code, "municipality": rec.get("name"), "state": rec.get("state"), "unit_expected": "ha", "forest_years": sorted(years_f), "loss_years": sorted(years_l), "forest_zero_count": sum(1 for v in forest.values() if isinstance(v, (int,float)) and v == 0), "loss_zero_count": sum(1 for v in loss.values() if isinstance(v, (int,float)) and v == 0), "negative_count": len(negatives), "invalid_count": len(invalid), "source": rec.get("source"), "issues": issues})
    report = {"source_file": str(SOURCE.relative_to(ROOT)), "unit_policy": "hectares (ha); zeros preserved and flagged, never assumed missing", "municipality_count": len(rows), "municipalities_with_issues": sum(bool(r["issues"]) for r in rows), "records": rows}
    target = (ROOT / args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Auditados {len(rows)} municípios; {report['municipalities_with_issues']} requerem revisão. Relatório: {target.relative_to(ROOT)}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
