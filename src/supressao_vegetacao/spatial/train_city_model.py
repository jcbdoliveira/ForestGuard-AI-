"""
Treino de modelo espacial **por município**.

Prioridade:
  1. Painel GEE real em data/spatial/panels/{ibge}/*.csv
  2. Painel explícito passado pelo chamador
  3. Painel sintético calibrado (seed=hash IBGE + taxa MapBiomas)

Artefatos:
  data/spatial/models/{ibge_code}/spatial_model.json
  data/spatial/models/{ibge_code}/spatial_model_meta.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .panel import Panel, load_panel_csv
from .train_spatial import train_spatial_model

ROOT = Path(__file__).resolve().parents[3]
PANELS_DIR = ROOT / "data" / "spatial" / "panels"


def _seed_from_ibge(ibge_code: str) -> int:
    h = hashlib.md5(str(ibge_code).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def find_city_panel(ibge_code: str) -> Optional[Path]:
    """Retorna diretório ou CSV do painel GEE da cidade, se existir."""
    code = str(ibge_code).strip()
    d = PANELS_DIR / code
    if d.is_dir() and any(d.glob("*.csv")):
        return d
    # CSV solto
    singles = list(PANELS_DIR.glob(f"panel_{code}*.csv")) + list(PANELS_DIR.glob(f"*{code}*.csv"))
    if singles:
        return singles[0]
    return None


def build_city_synthetic_panel(
    ibge_code: str,
    mean_loss_rate: float = 0.005,
    n_per_year: int = 100,
    years: Optional[list] = None,
) -> Panel:
    """Painel multi-feature sintético, distinto por município (fallback)."""
    rng = np.random.default_rng(_seed_from_ibge(ibge_code))
    years = years or [2018, 2019, 2020, 2021, 2022]
    pressure = float(np.clip(mean_loss_rate * 80.0, 0.3, 3.0))

    rows_X, rows_y, rows_year = [], [], []
    for year in years:
        for _ in range(n_per_year):
            slope = float(rng.uniform(0, 40))
            elev = float(rng.uniform(50, 1200))
            forest_n = float(rng.beta(2, 2))
            built = float(rng.exponential(40))
            z = (
                -1.2
                + pressure * 0.65
                + 0.4 * (1.0 - forest_n)
                + 0.15 * (built / 100.0)
                - 0.01 * slope
            )
            p = 1.0 / (1.0 + float(np.exp(-z)))
            y = 1.0 if rng.random() < p else 0.0
            rows_X.append([slope, elev, forest_n, built])
            rows_y.append(y)
            rows_year.append(year)

    return Panel(
        X=np.asarray(rows_X, dtype=float),
        y=np.asarray(rows_y, dtype=float),
        years=np.asarray(rows_year, dtype=np.int32),
        feature_names=["slope_deg", "elev_m", "forest_neighbor_frac", "built_surface", "forest_fraction_t0"],
        meta=[{"ibge_code": str(ibge_code), "synthetic": True}] * len(rows_y),
    )


def train_city_spatial_model(
    ibge_code: str,
    mean_loss_rate: float = 0.005,
    model_dir: Optional[str | Path] = None,
    panel: Optional[Panel] = None,
    panel_path: Optional[str | Path] = None,
    prefer_gee: bool = True,
    allow_synthetic_dev: bool = False,
) -> Dict[str, Any]:
    """
    Treina e persiste modelo espacial específico do município.
    Se prefer_gee=True e existir painel em data/spatial/panels/{ibge}/, usa GEE real.
    """
    code = str(ibge_code).strip()
    dest = Path(model_dir) if model_dir else ROOT / "data" / "spatial" / "models" / code

    data_source = "synthetic_city_calibrated"
    resolved_panel_path = None

    if panel is None:
        path = Path(panel_path) if panel_path else (find_city_panel(code) if prefer_gee else None)
        if path and Path(path).exists():
            panel = load_panel_csv(path, target_col="y_loss_1y", augment_osm_features=False)
            data_source = "gee_mapbiomas_panel"
            resolved_panel_path = str(path)
        elif allow_synthetic_dev:
            panel = build_city_synthetic_panel(code, mean_loss_rate=mean_loss_rate)
            dest = ROOT / "data" / "spatial" / "models" / "_development_only" / code
            dest.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Painel GEE/MapBiomas real não encontrado para {code}. "
                "Treino sintético bloqueado; para desenvolvimento use allow_synthetic_dev=True."
            )

    dest.mkdir(parents=True, exist_ok=True)
    _, result = train_spatial_model(panel, model_dir=str(dest), prefer_xgboost=True)

    meta_path = dest / "spatial_model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["ibge_code"] = code
        meta["model_class"] = (
            "city_specific_spatial_gee" if data_source == "gee_mapbiomas_panel"
            else "city_specific_spatial"
        )
        meta["data_source"] = data_source
        meta["panel_path"] = resolved_panel_path
        meta["mean_loss_rate_used"] = mean_loss_rate
        if data_source == "gee_mapbiomas_panel":
            meta["training_note"] = (
                f"Modelo espacial do município {code} treinado em painel GEE/MapBiomas real "
                f"({resolved_panel_path}). Split temporal; features em t0 apenas."
            )
        else:
            meta["training_note"] = (
                f"Modelo espacial do município {code}. Sem painel GEE em "
                f"data/spatial/panels/{code}/ — usou painel sintético seed=hash(IBGE) "
                "calibrado pela taxa MapBiomas municipal. Exporte o painel real via "
                "scripts/gee/export_municipality_panel.js para substituir."
            )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ibge_code": code,
        "model_dir": str(dest),
        "model_path": result.model_path,
        "model_type": result.model_type,
        "metrics_test": result.metrics_test,
        "feature_names": result.feature_names,
        "feature_importance": result.feature_importance,
        "n_train": result.n_train,
        "city_specific": True,
        "data_source": data_source,
        "panel_path": resolved_panel_path,
    }
