"""Distância métrica até à via OSM mais próxima no GeoJSON local.

O GeoJSON pode conter mais de uma região (por exemplo Mafra/SC e Mafra/PT).
As geometrias são agrupadas por zona UTM para nunca medir distâncias entre regiões
em graus nem projetar um continente no sistema métrico do outro.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROADS_PATH = ROOT / "data" / "roads" / "export.geojson"  # apenas para utilitários genéricos/testes
ROADS_DIR = ROOT / "data" / "roads"
MAX_COVERAGE_KM = 50.0


def roads_path_for_municipality(ibge_code: object) -> Optional[Path]:
    """Rede OSM isolada por município; nunca usa a rede global como fallback."""
    code = "".join(ch for ch in str(ibge_code or "") if ch.isdigit())
    if len(code) != 7:
        return None
    path = ROADS_DIR / f"{code}.geojson"
    return path if path.is_file() else None


# Categorias de influência potencial. São grupos de tags OSM, não pesos causais:
# o modelo aprende a contribuição de cada distância no painel temporal.
ROAD_CLASS_TYPES = {
    "major": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"},
    "secondary": {"secondary", "secondary_link", "tertiary", "tertiary_link"},
    "local": {"unclassified", "residential", "living_street", "service", "services", "road"},
    "track": {"track"},
}
ROAD_CLASS_FEATURES = {key: f"road_dist_{key}_km" for key in ROAD_CLASS_TYPES}


def _matches_class(highway: str, road_class: str) -> bool:
    if road_class == "all":
        return True
    return highway in ROAD_CLASS_TYPES.get(road_class, set())


def _zone(lon: float, lat: float) -> tuple[int, bool]:
    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return zone, lat < 0


@lru_cache(maxsize=32)
def _road_layers(path_string: str, road_class: str = "all"):
    path = Path(path_string)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        # Evita confundir trilhas exclusivas de pedestres/bicicletas com vias
        # de acesso motorizado à floresta. Estradas de terra (track) são mantidas.
        highway = props.get("highway")
        if not highway or highway in {"footway", "path", "steps", "pedestrian", "bridleway", "cycleway"}:
            continue
        if not _matches_class(str(highway), road_class):
            continue
        geom_data = feature.get("geometry")
        if not geom_data:
            continue
        try:
            geom = shape(geom_data)
            if geom.is_empty:
                continue
            # Each road feature is assigned to the UTM zone of its centroid.
            lon, lat = geom.centroid.x, geom.centroid.y
            key = _zone(lon, lat)
            grouped.setdefault(key, []).append(geom)
        except Exception:
            continue

    layers = {}
    for (zone, south), geoms in grouped.items():
        epsg = (32700 + zone) if south else (32600 + zone)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        projected = [transform_geometry(transformer.transform, g) for g in geoms]
        projected = [g for g in projected if not g.is_empty]
        if projected:
            layers[(zone, south)] = (STRtree(projected), projected, transformer)
    return layers


@lru_cache(maxsize=200000)
def _distance_cached(lon_rounded: float, lat_rounded: float, path_string: str, road_class: str = "all") -> Optional[float]:
    if not (math.isfinite(lon_rounded) and math.isfinite(lat_rounded)):
        return None
    zone_key = _zone(lon_rounded, lat_rounded)
    layer = _road_layers(path_string, road_class).get(zone_key)
    if layer is None:
        return None
    tree, geometries, transformer = layer
    x, y = transformer.transform(lon_rounded, lat_rounded)
    point = Point(x, y)
    try:
        nearest = tree.nearest(point)
        if nearest is None:
            return None
        # Shapely 2 STRtree.nearest returns an index, not the geometry.
        nearest_geom = geometries[int(nearest)] if isinstance(nearest, (int,)) or getattr(nearest, "dtype", None) is not None else nearest
        km = float(nearest_geom.distance(point)) / 1000.0
    except Exception:
        return None
    # Avoid assigning the road network of an unrelated region to a cell.
    if not math.isfinite(km) or km > MAX_COVERAGE_KM:
        return None
    return round(max(0.0, km), 5)


def road_distance_km(lon: object, lat: object, roads_path: str | Path | None = None) -> Optional[float]:
    """Returns nearest OSM road distance in km, or None outside road coverage."""
    try:
        x, y = float(lon), float(lat)
    except (TypeError, ValueError):
        return None
    path = Path(roads_path) if roads_path is not None else DEFAULT_ROADS_PATH
    return _distance_cached(round(x, 5), round(y, 5), str(path.resolve()), "all")


def road_distance_features(lon: object, lat: object, roads_path: str | Path | None = None) -> dict[str, Optional[float]]:
    """Distâncias métricas por categoria OSM, mais a distância geral.

    Classes: vias principais (motorway/trunk/primary), secundárias/terciárias,
    locais (residential/service/unclassified) e estradas de terra/florestais
    (track). Valores None significam que não há geometria dessa classe a até
    MAX_COVERAGE_KM; o chamador deve tratá-los como ausentes, não como zero.
    """
    try:
        x, y = float(lon), float(lat)
    except (TypeError, ValueError):
        return {"road_dist_km": None, **{name: None for name in ROAD_CLASS_FEATURES.values()}}
    path = Path(roads_path) if roads_path is not None else DEFAULT_ROADS_PATH
    path_string = str(path.resolve())
    x, y = round(x, 5), round(y, 5)
    out = {"road_dist_km": _distance_cached(x, y, path_string, "all")}
    for key, feature_name in ROAD_CLASS_FEATURES.items():
        out[feature_name] = _distance_cached(x, y, path_string, key)
    return out
