#!/usr/bin/env python3
"""Verificação rápida da instalação e do núcleo anual (sem treinar)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    errors = []
    print("ForestGuard v25 — verify_annual")
    print("root:", ROOT)

    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        print("✓ numpy / sklearn")
    except ImportError as exc:
        errors.append(f"deps ML: {exc}")
        print("✗ numpy/sklearn:", exc)

    try:
        import xgboost  # noqa: F401
        print("✓ xgboost")
        has_xgb = True
    except ImportError:
        has_xgb = False
        print("! xgboost ausente (treino/inferência XGB indisponível)")

    from supressao_vegetacao.spatial.panel import (
        load_panel_csv,
        prepare_inference_features,
        _neighbor_feature_name,
    )
    from supressao_vegetacao.spatial.infer import find_spatial_model, load_spatial_model

    panel_dir = ROOT / "data" / "spatial" / "panels" / "4210100"
    if not panel_dir.is_dir():
        errors.append("painel 4210100 ausente")
        print("✗ painel 4210100")
    else:
        panel = load_panel_csv(panel_dir, target_col="y_loss_1y")
        print(f"✓ painel y_loss_1y: n={len(panel.y)} pos={panel.y.mean():.4f}")

    model_path = find_spatial_model(ibge_code="4210100")
    if model_path is None:
        errors.append("modelo espacial 4210100 não encontrado")
        print("✗ modelo")
    else:
        if has_xgb:
            model, names, meta = load_spatial_model(model_path)
            print(f"✓ modelo: {model_path.parent.name} target={meta.get('target_col')} H={meta.get('horizon_years')}")
            print(f"  features: {names}")
            if meta.get("target_col") != "y_loss_1y":
                errors.append(f"target inesperado: {meta.get('target_col')}")
            prepared = prepare_inference_features(
                panel_dir, feature_names=names, horizon_years=1
            )
            print(f"✓ inferência features: n={prepared['X'].shape[0]} year={prepared['panel_year']}")
            if prepared["X"].shape[0] == 0:
                errors.append("matriz de inferência vazia")
        else:
            meta = json.loads((model_path.parent / "spatial_model_meta.json").read_text())
            print(f"✓ meta (sem load XGB): target={meta.get('target_col')}")

    print("vizinho:", _neighbor_feature_name(1))
    if errors:
        print("FALHAS:", errors)
        return 1
    print("OK — núcleo anual verificável")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
