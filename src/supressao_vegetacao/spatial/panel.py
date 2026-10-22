"""
Carrega e prepara painel célula × ano para modelagem espacial.

Schema CSV v2 (export GEE / painéis atuais):
  cell_id, ibge_code, lon, lat, year_t0, forest_t0,
  y_loss_1y, y_loss_3y (pode ser vazio se t0+3 indisponível),
  slope_deg, elev_m, forest_neighbor_frac, years_stable_forest,
  forest_fraction_t0, valid_fraction_t0, grid_x_m, grid_y_m,
  built_surface, built_dist_km

cell_id típico: M{IBGE}_{grid_x_m}_{grid_y_m}

Opcionais (NaN permitido): built_*, forest_fraction_t0, valid_fraction_t0,
years_stable_forest, road_dist_* (OSM se augment_osm_features=True).

y_loss_Hy = 1 se floresta em t0 e não-floresta em t0+H (observado).
Features conhecidas em t0 (sem leakage). Treino usa só forest_t0≈1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_FEATURE_COLS = [
    "slope_deg",
    "elev_m",
    "forest_neighbor_frac",
    "built_surface",
    "built_dist_km",
    "years_stable_forest",
    "forest_fraction_t0",
    "valid_fraction_t0",
    "road_dist_km",
    "road_dist_major_km",
    "road_dist_secondary_km",
    "road_dist_local_km",
    "road_dist_track_km",
]

# Features que podem faltar em alguns anos/exportações sem invalidar a célula.
OPTIONAL_FEATURE_COLS = {
    "built_surface",
    "built_dist_km",
    "forest_fraction_t0",
    "valid_fraction_t0",
    "years_stable_forest",
    "road_dist_km",
    "road_dist_major_km",
    "road_dist_secondary_km",
    "road_dist_local_km",
    "road_dist_track_km",
}


@dataclass
class Panel:
    X: np.ndarray
    y: np.ndarray
    years: np.ndarray
    feature_names: List[str]
    meta: List[Dict]


def _neighbor_feature_name(horizon_years: int) -> str:
    """Nome canônico da feature de vizinhança alinhado ao horizonte do alvo."""
    h = max(1, int(horizon_years))
    return f"loss_neighbors_history_{h}y"


def augment_with_spatial_neighbors(panel: Panel, k: int = 3, horizon_years: int = 1) -> Panel:
    """
    Adiciona histórico de perda em células vizinhas usando somente rótulos
    observáveis antes do ano previsto.

    Para uma célula em T, candidatos precisam satisfazer:
        year_t0 <= T - horizon_years - 1

    Importante: o painel contém várias observações anuais da mesma célula.
    Cada coordenada vizinha contribui UMA vez, usando o rótulo elegível mais
    recente daquela célula. Isso evita que os k vizinhos sejam, na prática,
    várias linhas históricas repetidas de um único vizinho.

    A própria coordenada é excluída. Se não houver vizinhos elegíveis, o valor
    permanece NaN; nunca é substituído pelo rótulo da própria célula.
    """
    if panel.X is None or len(panel.y) == 0 or not panel.meta:
        return panel

    H = max(1, int(horizon_years))
    feat_name = _neighbor_feature_name(H)
    legacy_names = {
        "loss_neighbors_persistence_3y",
        "loss_neighbors_history_1y",
        "loss_neighbors_history_3y",
        feat_name,
    }
    drop_idx = [i for i, n in enumerate(panel.feature_names) if n in legacy_names]
    names = [n for n in panel.feature_names if n not in legacy_names]
    base_X = np.delete(panel.X, drop_idx, axis=1) if drop_idx else panel.X

    n = len(panel.y)
    lons = np.array([_to_float(m.get("lon")) for m in panel.meta], dtype=np.float64)
    lats = np.array([_to_float(m.get("lat")) for m in panel.meta], dtype=np.float64)
    years = np.array([int(_to_float(m.get("year_t0"), 0)) for m in panel.meta], dtype=np.int32)
    grid_x = np.array([_to_float(m.get("grid_x_m")) for m in panel.meta], dtype=np.float64)
    grid_y = np.array([_to_float(m.get("grid_y_m")) for m in panel.meta], dtype=np.float64)
    use_projected = np.isfinite(grid_x).all() and np.isfinite(grid_y).all()
    if use_projected:
        px, py = grid_x, grid_y
        # Grid centers in the exported panel are metric; group by rounded meter.
        coord_keys = [(round(float(x), 1), round(float(y), 1)) for x, y in zip(grid_x, grid_y)]
    else:
        px, py = lons, lats
        coord_keys = [
            (round(float(lon), 5), round(float(lat), 5))
            if np.isfinite(lon) and np.isfinite(lat) else (None, i)
            for i, (lon, lat) in enumerate(zip(lons, lats))
        ]

    extra = np.full(n, np.nan, dtype=np.float64)
    y = np.asarray(panel.y, dtype=np.float64)
    coordinate_to_rows: Dict[Tuple[Any, Any], List[int]] = {}
    for idx, key in enumerate(coord_keys):
        coordinate_to_rows.setdefault(key, []).append(idx)

    for year in sorted(set(int(v) for v in years.tolist())):
        target = np.flatnonzero(years == year)
        cutoff = year - H - 1
        if len(target) == 0:
            continue

        # Latest eligible label per unique neighboring cell, not per CSV row.
        candidate_rows: List[int] = []
        for rows in coordinate_to_rows.values():
            eligible = [i for i in rows if years[i] <= cutoff and np.isfinite(y[i])]
            if eligible:
                latest_year = max(int(years[i]) for i in eligible)
                latest_rows = [i for i in eligible if int(years[i]) == latest_year]
                candidate_rows.append(latest_rows[0])

        if not candidate_rows:
            continue
        cand = np.asarray(candidate_rows, dtype=int)
        dist = np.hypot(
            px[cand][None, :] - px[target][:, None],
            py[cand][None, :] - py[target][:, None],
        )
        same = np.array([
            [coord_keys[cand_j] == coord_keys[target_i] for cand_j in cand]
            for target_i in target
        ], dtype=bool)
        dist = np.where(same | ~np.isfinite(dist), np.inf, dist)

        kk = min(max(1, int(k)), dist.shape[1])
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(target))[:, None]
        chosen_dist = dist[rows, part]
        chosen_y = y[cand][part]
        weights = np.where(np.isfinite(chosen_dist), 1.0 / np.maximum(chosen_dist, 1e-9), 0.0)
        weight_sum = weights.sum(axis=1)
        valid = weight_sum > 0
        if np.any(valid):
            extra[target[valid]] = (
                weights[valid] * chosen_y[valid]
            ).sum(axis=1) / weight_sum[valid]

    new_X = np.hstack([base_X, extra.reshape(-1, 1)])
    return Panel(
        X=new_X,
        y=panel.y,
        years=panel.years,
        feature_names=list(names) + [feat_name],
        meta=list(panel.meta),
    )


def _to_float(v, default=np.nan) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_panel_csv(
    path: str | Path,
    target_col: str = "y_loss_1y",
    feature_cols: Optional[Sequence[str]] = None,
    require_forest_t0: bool = True,
    augment_osm_features: bool = False,
) -> Panel:
    """
    Lê um ou mais CSVs (caminho único ou diretório com vários panel_*.csv).
    """
    path = Path(path)
    files: List[Path]
    if path.is_dir():
        # Convenção preferencial: painéis exportados usam panel_*.csv. Evita
        # misturar planilhas de saída (ex. resultado_projecao_ia.csv) no treino.
        files = sorted(path.glob("panel_*.csv"))
        if not files:
            ignored_tokens = ("resultado", "resultados", "output", "predicao", "projecao")
            files = [
                fp for fp in sorted(path.glob("*.csv"))
                if not any(token in fp.stem.lower() for token in ignored_tokens)
            ]
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"Nenhum CSV em {path}")

    feature_cols = list(feature_cols) if feature_cols else None
    rows_X: List[List[float]] = []
    rows_y: List[float] = []
    rows_year: List[int] = []
    meta: List[Dict] = []
    resolved_features: Optional[List[str]] = None

    for fp in files:
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fields = list(reader.fieldnames)
            if target_col not in fields:
                raise ValueError(f"{fp.name}: coluna alvo '{target_col}' ausente. Colunas: {fields}")
            if feature_cols is None:
                cand = [c for c in DEFAULT_FEATURE_COLS if c in fields]
                if not cand:
                    raise ValueError(
                        f"{fp.name}: nenhuma feature conhecida encontrada. "
                        f"Esperado subconjunto de {DEFAULT_FEATURE_COLS}. Colunas: {fields}"
                    )
                if resolved_features is None:
                    resolved_features = cand
            else:
                missing = [c for c in feature_cols if c not in fields]
                if missing:
                    raise ValueError(f"{fp.name}: features ausentes: {missing}")
                resolved_features = list(feature_cols)

            assert resolved_features is not None
            for row in reader:
                if require_forest_t0 and "forest_t0" in fields:
                    if _to_float(row.get("forest_t0"), 0) < 0.5:
                        continue
                y = _to_float(row.get(target_col))
                if np.isnan(y):
                    continue
                feat = [_to_float(row.get(c)) for c in resolved_features]
                # XGBoost lida com NaN nativamente. Features opcionais (estradas,
                # built_*, frações auxiliares) podem faltar; essenciais não.
                if any(
                    np.isnan(v)
                    for c, v in zip(resolved_features, feat)
                    if c not in OPTIONAL_FEATURE_COLS
                ):
                    continue
                year = int(_to_float(row.get("year_t0"), 0))
                if year < 2000:
                    continue
                rows_X.append(feat)
                rows_y.append(1.0 if y >= 0.5 else 0.0)
                rows_year.append(year)
                meta.append({
                    "source_file": fp.name,
                    "cell_id": row.get("cell_id"),
                    "lon": _to_float(row.get("lon")),
                    "lat": _to_float(row.get("lat")),
                    "ibge_code": row.get("ibge_code") or row.get("CD_MUN"),
                    "year_t0": year,
                    "grid_x_m": _to_float(row.get("grid_x_m")),
                    "grid_y_m": _to_float(row.get("grid_y_m")),
                })

    if not rows_X:
        raise RuntimeError(
            f"Painel vazio após filtros (path={path}, target={target_col}). "
            "Verifique CSVs e se há células com floresta em t0."
        )

    X = np.asarray(rows_X, dtype=np.float64)
    feature_names = list(resolved_features or [])
    # OSM é opt-in: painéis genéricos preservam exatamente seu schema.
    # Quando solicitado, usa exclusivamente a rede do município.
    if augment_osm_features:
        try:
            from .road_distance import road_distance_features, ROAD_CLASS_FEATURES, roads_path_for_municipality
            names = list(feature_names)
            additions = [name for name in ROAD_CLASS_FEATURES.values() if name not in names]
            # Se o painel tem as quatro classes, a distância geral é redundante e sai.
            for i, name in enumerate(list(names)):
                if name == "road_dist_km" and all(n in names or n in additions for n in ROAD_CLASS_FEATURES.values()):
                    X = np.delete(X, i, axis=1)
                    names.pop(i)
                    break
            additions = [name for name in ROAD_CLASS_FEATURES.values() if name not in names]
            old_n = X.shape[1]
            for name in additions:
                X = np.column_stack([X, np.full(len(meta), np.nan, dtype=np.float64)])
                names.append(name)
            # Distâncias calculadas exclusivamente com o GeoJSON do código IBGE da linha.
            for i, item in enumerate(meta):
                road_path = roads_path_for_municipality(item.get("ibge_code"))
                if road_path is None:
                    road_values = {name: None for name in ["road_dist_km", *ROAD_CLASS_FEATURES.values()]}
                else:
                    road_values = road_distance_features(item.get("lon"), item.get("lat"), roads_path=road_path)
                for name, value in road_values.items():
                    if name in names:
                        X[i, names.index(name)] = float(value) if value is not None else np.nan
            # Distância geral removida sempre que as quatro categorias estão presentes.
            if all(n in names for n in ROAD_CLASS_FEATURES.values()) and "road_dist_km" in names:
                j = names.index("road_dist_km")
                X = np.delete(X, j, axis=1)
                names.pop(j)
            feature_names = names
            import logging
            logging.getLogger("forestguard.spatial.panel").info(
                "Features OSM por município: %s; amostras=%d (colunas base=%d); distância geral removida se redundante",
                list(ROAD_CLASS_FEATURES.values()), len(meta), old_n
            )
        except Exception as exc:
            import logging
            logging.getLogger("forestguard.spatial.panel").warning(
                "Não foi possível aplicar distâncias rodoviárias OSM por classe: %s", exc
            )

    return Panel(
        X=X,
        y=np.asarray(rows_y, dtype=np.float64),
        years=np.asarray(rows_year, dtype=np.int32),
        feature_names=feature_names,
        meta=meta,
    )


def temporal_split(
    panel: Panel,
    test_year: Optional[int] = None,
    valid_year: Optional[int] = None,
    horizon_years: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split temporal com embargo para que rótulos de horizonte não cruzem o teste.

    Para horizonte H, o último t0 de treino é test_year - H - 1.
    Ex.: teste 2020 / horizonte 3 => treino no máximo 2016.
    Uma validação explícita também precisa terminar seu rótulo antes do teste.
    """
    years_sorted = sorted(set(int(y) for y in panel.years))
    if not years_sorted:
        raise RuntimeError("Painel sem anos.")
    if test_year is None:
        test_year = years_sorted[-1]
    horizon_years = max(0, int(horizon_years))
    train_cutoff = int(test_year) - horizon_years - 1
    # Compatibilidade para split genérico sem horizonte: mantém validação penúltima.
    if valid_year is None and horizon_years == 0 and len(years_sorted) >= 2:
        valid_year = years_sorted[-2]

    if valid_year is not None:
        if int(valid_year) > train_cutoff:
            raise ValueError(
                f"Validação {valid_year} inválida para teste {test_year} com horizonte {horizon_years}y; "
                f"o rótulo da validação pode cruzar o teste. Máximo seguro: {train_cutoff}."
            )
        valid_cutoff = int(valid_year) - horizon_years - 1
        train_m = panel.years <= valid_cutoff
        valid_m = panel.years == int(valid_year)
    else:
        train_m = panel.years <= train_cutoff
        valid_m = np.zeros_like(panel.years, dtype=bool)

    test_m = panel.years == int(test_year)
    if train_m.sum() == 0 or test_m.sum() == 0:
        raise RuntimeError(
            f"Split inválido: train={train_m.sum()} valid={valid_m.sum()} test={test_m.sum()} "
            f"(anos={years_sorted}, horizonte={horizon_years}, corte treino={train_cutoff}, "
            f"valid={valid_year}, test={test_year})"
        )

    out = {
        "train": (panel.X[train_m], panel.y[train_m]),
        "test": (panel.X[test_m], panel.y[test_m]),
        "years": (years_sorted, int(valid_year) if valid_year is not None else None, int(test_year)),
        "train_years": sorted(set(int(y) for y in panel.years[train_m])),
        "train_cutoff_year": int(np.max(panel.years[train_m])),
        "horizon_years": horizon_years,
    }
    if valid_m.sum() > 0:
        out["valid"] = (panel.X[valid_m], panel.y[valid_m])
    return out


def prepare_inference_features(
    panel_dir: str | Path,
    feature_names: Sequence[str],
    target_year: Optional[int] = None,
    horizon_years: int = 1,
    require_forest_t0: bool = True,
    panel: Optional["Panel"] = None,
) -> Dict[str, Any]:
    """
    Constrói a matriz de features para inferência com a mesma semântica do treino.

    - Carrega todos os panel_*.csv do município (ou usa `panel` já em memória).
    - Seleciona o ano de previsão (último disponível ou target_year).
    - Calcula loss_neighbors_history_{H}y usando apenas histórico elegível
      (year_t0 <= T - H - 1), excluindo a própria célula.
    - Retorna apenas células com floresta em t0 e features essenciais presentes.
    - Features opcionais (estradas, vizinhos) podem ser NaN.

    Retorno:
      {
        "X": np.ndarray (n, n_features),
        "meta": list[dict],
        "feature_names": list[str],
        "panel_year": int,
        "target_year": int,
        "n_skipped_missing": int,
        "skipped": list[dict],
      }
    """
    panel_dir = Path(panel_dir)

    H = max(1, int(horizon_years))
    neighbor_name = _neighbor_feature_name(H)
    needs_neighbor = neighbor_name in feature_names or any(
        n.startswith("loss_neighbors_") for n in feature_names
    )

    # Painel completo em memória (startup) ou leitura dos CSVs.
    # y é necessário apenas para calcular a feature de vizinhança histórica.
    if panel is not None:
        full = panel
    else:
        if not panel_dir.is_dir():
            raise FileNotFoundError(f"Diretório de painel inexistente: {panel_dir}")
        try:
            full = load_panel_csv(
                panel_dir,
                target_col="y_loss_1y",
                require_forest_t0=require_forest_t0,
                augment_osm_features=False,
            )
        except (ValueError, RuntimeError):
            full = load_panel_csv(
                panel_dir,
                target_col="y_loss_3y",
                require_forest_t0=require_forest_t0,
                augment_osm_features=False,
            )

    years_available = sorted(set(int(y) for y in full.years))
    if not years_available:
        raise RuntimeError("Painel sem anos após filtros.")
    if target_year is None:
        target_year = years_available[-1]
    if int(target_year) not in years_available:
        raise RuntimeError(
            f"Ano de previsão {target_year} ausente no painel. Disponíveis: {years_available}"
        )

    # Calcula feature de vizinhança no painel completo (usa y histórico)
    if needs_neighbor:
        full = augment_with_spatial_neighbors(full, k=3, horizon_years=H)

    # Mapa nome -> coluna no painel completo
    name_to_idx = {n: i for i, n in enumerate(full.feature_names)}

    nan_allowed = set(OPTIONAL_FEATURE_COLS) | {
        "loss_neighbors_history_1y", "loss_neighbors_history_3y",
        "loss_neighbors_persistence_3y", neighbor_name,
    }

    pred_mask = full.years == int(target_year)
    X_rows: List[List[float]] = []
    meta_out: List[Dict] = []
    skipped: List[Dict] = []

    for i in np.flatnonzero(pred_mask):
        row_meta = dict(full.meta[i])
        values: List[float] = []
        missing_essential: List[str] = []
        for fname in feature_names:
            if fname in name_to_idx:
                v = float(full.X[i, name_to_idx[fname]])
            else:
                v = float("nan")
            if not np.isfinite(v) and fname not in nan_allowed:
                missing_essential.append(fname)
            values.append(v)
        if missing_essential:
            skipped.append({
                "cell_id": row_meta.get("cell_id") or row_meta.get("source_file"),
                "missing_features": missing_essential,
            })
            continue
        X_rows.append(values)
        meta_out.append(row_meta)

    if not X_rows:
        return {
            "X": np.zeros((0, len(feature_names)), dtype=np.float64),
            "meta": [],
            "feature_names": list(feature_names),
            "panel_year": int(target_year),
            "target_year": int(target_year) + H,
            "n_skipped_missing": len(skipped),
            "skipped": skipped,
        }

    return {
        "X": np.asarray(X_rows, dtype=np.float64),
        "meta": meta_out,
        "feature_names": list(feature_names),
        "panel_year": int(target_year),
        "target_year": int(target_year) + H,
        "n_skipped_missing": len(skipped),
        "skipped": skipped,
    }
