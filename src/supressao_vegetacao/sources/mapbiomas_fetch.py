"""
Série anual de desmate municipal a partir de fontes oficiais.

1) TerraBrasilis / INPE (WFS PRODES por bioma) — polígonos anuais intersectados
   com a malha IBGE do município (ha reais).
2) Cache local em data/mapbiomas/cache/{ibge}_prodes.json

Não inventa hectares: só agrega geometrias oficiais.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "mapbiomas" / "cache"
BOUNDARIES_DIR = ROOT / "data" / "boundaries"

# Workspaces WFS TerraBrasilis (INPE) por bioma/recortes
WFS_BASE = "https://terrabrasilis.dpi.inpe.br/geoserver"
WFS_LAYERS = [
    # Mata Atlântica (Curitiba, SC, PR, SP, etc.)
    ("prodes-mata-atlantica-nb", "prodes-mata-atlantica-nb:yearly_deforestation"),
    # Amazônia Legal
    ("prodes-legal-amazon", "prodes-legal-amazon:yearly_deforestation"),
    # Cerrado
    ("prodes-cerrado-nb", "prodes-cerrado-nb:yearly_deforestation"),
    # Pantanal / outros quando existirem
    ("prodes-pantanal-nb", "prodes-pantanal-nb:yearly_deforestation"),
]


WFS_RESIDUAL_LAYERS = [
    ("prodes-mata-atlantica-nb", "prodes-mata-atlantica-nb:residual"),
    ("prodes-legal-amazon", "prodes-legal-amazon:residual"),
    ("prodes-cerrado-nb", "prodes-cerrado-nb:residual"),
]


def estimate_forest_remnant_ha(ibge: str) -> Optional[float]:
    """
    Estima remanescente florestal (ha) somando polígonos `residual` do PRODES
    (TerraBrasilis/INPE) cujo centroide cai dentro da malha IBGE.
    Retorna None se não houver features.
    """
    geom = load_municipality_boundary(ibge)
    if geom is None:
        return None
    bbox = _bbox_of_geom(geom)
    if bbox is None:
        return None
    total_ha = 0.0
    n = 0
    for _ws, layer in WFS_RESIDUAL_LAYERS:
        start = 0
        page = 5000
        while True:
            feats = _wfs_yearly_features(layer, bbox, start_index=start, count=page)
            if not feats:
                break
            for ft in feats:
                g = ft.get("geometry") or {}
                props = ft.get("properties") or {}
                cen = _centroid_of_geom(g)
                if cen is None:
                    continue
                if not _point_in_polygon(cen[0], cen[1], geom):
                    continue
                try:
                    ha = float(props.get("area_km") or 0) * 100.0
                except Exception:
                    continue
                if ha > 0:
                    total_ha += ha
                    n += 1
            if len(feats) < page:
                break
            start += page
            if start > 50000:
                break
        if n > 0:
            break
    if n == 0:
        return None
    return round(total_ha, 1)




def _http_get_json(url: str, timeout: int = 120) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ForestGuard/7.0 (research; +local)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    """Ray casting; ring = [[lon,lat], ...]"""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, geojson_geom: Dict[str, Any]) -> bool:
    gtype = geojson_geom.get("type")
    coords = geojson_geom.get("coordinates")
    if not coords:
        return False
    if gtype == "Polygon":
        # exterior ring only
        return _point_in_ring(lon, lat, coords[0])
    if gtype == "MultiPolygon":
        for poly in coords:
            if poly and _point_in_ring(lon, lat, poly[0]):
                return True
    return False


def _centroid_of_geom(geom: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Centroid aproximado (média dos vértices do anel exterior)."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    ring = None
    if gtype == "Polygon":
        ring = coords[0]
    elif gtype == "MultiPolygon" and coords and coords[0]:
        ring = coords[0][0]
    elif gtype == "Point":
        return float(coords[0]), float(coords[1])
    if not ring:
        return None
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def load_municipality_boundary(ibge: str) -> Optional[Dict[str, Any]]:
    path = BOUNDARIES_DIR / f"{ibge}.geojson"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features") or []
    if not feats:
        # pode ser GeometryCollection direto
        if data.get("type") in ("Polygon", "MultiPolygon"):
            return data
        return None
    return feats[0].get("geometry")


def _bbox_of_geom(geom: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    pts: List[Tuple[float, float]] = []

    def walk(c):
        if c is None:
            return
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            pts.append((float(c[0]), float(c[1])))
            return
        if isinstance(c, (list, tuple)):
            for x in c:
                walk(x)

    walk(coords)
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return min(lons), min(lats), max(lons), max(lats)


def _wfs_yearly_features(workspace_layer: str, bbox: Tuple[float, float, float, float], start_index: int = 0, count: int = 5000) -> List[Dict[str, Any]]:
    minx, miny, maxx, maxy = bbox
    # BBOX filter in lon/lat
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": workspace_layer,
        "outputFormat": "application/json",
        "maxFeatures": str(count),
        "startIndex": str(start_index),
        "srsName": "EPSG:4326",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
    }
    url = f"{WFS_BASE}/{workspace_layer.split(':')[0]}/ows?{urllib.parse.urlencode(params)}"
    # workspace in path: geoserver/{workspace}/ows
    ws = workspace_layer.split(":")[0]
    url = f"{WFS_BASE}/{ws}/ows?{urllib.parse.urlencode(params)}"
    try:
        data = _http_get_json(url, timeout=180)
    except Exception:
        return []
    return data.get("features") or []


def aggregate_deforestation_ha(
    ibge: str,
    years: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Agrega área de desmate (ha) por ano dentro do polígono IBGE do município.

    Retorna:
      {
        "ibge_code": "...",
        "source": "TerraBrasilis PRODES (INPE) WFS yearly_deforestation",
        "historic_prodes_deforestation_ha": {"2018": 12.3, ...},
        "forest_remnant_note": "...",
        "n_polygons": N,
        "layers_tried": [...]
      }
    """
    geom = load_municipality_boundary(ibge)
    if geom is None:
        raise FileNotFoundError(
            f"Malha IBGE ausente: data/boundaries/{ibge}.geojson. "
            "Rode add_city antes (baixa o polígono oficial)."
        )
    bbox = _bbox_of_geom(geom)
    if bbox is None:
        raise ValueError("BBOX inválido para o polígono municipal")

    year_set = set(int(y) for y in (years or list(range(2015, 2025))))
    ha_by_year: Dict[str, float] = {str(y): 0.0 for y in sorted(year_set)}
    n_poly = 0
    layers_ok: List[str] = []

    for _ws, layer in WFS_LAYERS:
        # paginar para cobrir o bbox
        start = 0
        page = 5000
        layer_hits = 0
        while True:
            feats = _wfs_yearly_features(layer, bbox, start_index=start, count=page)
            if not feats:
                break
            if layer not in layers_ok:
                layers_ok.append(layer)
            for ft in feats:
                props = ft.get("properties") or {}
                g = ft.get("geometry") or {}
                y_raw = props.get("year") or props.get("image_date") or ""
                y_str = str(y_raw)[:4]
                if not re.fullmatch(r"20\d{2}", y_str):
                    continue
                yi = int(y_str)
                if yi not in year_set:
                    continue
                cen = _centroid_of_geom(g)
                if cen is None:
                    continue
                lon, lat = cen
                if not _point_in_polygon(lon, lat, geom):
                    continue
                area_km = props.get("area_km")
                try:
                    ha = float(area_km) * 100.0  # km² → ha
                except Exception:
                    continue
                if ha < 0:
                    continue
                ha_by_year[y_str] = ha_by_year.get(y_str, 0.0) + ha
                n_poly += 1
                layer_hits += 1
            if len(feats) < page:
                break
            start += page
            if start > 50000:
                break

    # arredonda
    series = {y: round(v, 2) for y, v in sorted(ha_by_year.items(), key=lambda kv: kv[0]) if v > 0 or y in {str(x) for x in (years or [])}}
    # garante anos pedidos mesmo se zero (desmate zero é dado real)
    if years:
        for y in years:
            series.setdefault(str(y), 0.0)
        series = {str(y): series.get(str(y), 0.0) for y in sorted(years)}

    return {
        "ibge_code": ibge,
        "source": "TerraBrasilis/INPE PRODES WFS yearly_deforestation (centroid ∈ polígono IBGE)",
        "historic_prodes_deforestation_ha": series,
        "n_polygons": n_poly,
        "layers_used": layers_ok,
        "note": (
            "Áreas oficiais de desmate anual (PRODES biomas via TerraBrasilis). "
            "Atribuição municipal por centroide do polígono de desmate dentro da malha IBGE."
        ),
    }


def fetch_and_write_prodes_json(
    ibge: str,
    out_path: Path,
    city_label: str = "",
    years: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Busca série oficial e grava JSON no formato do add_city.
    """
    years = years or list(range(2018, 2025))
    result = aggregate_deforestation_ha(ibge, years=years)
    series = result["historic_prodes_deforestation_ha"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_source": result["source"],
        "_ibge_code": ibge,
        "_municipio": city_label,
        "_n_polygons": result["n_polygons"],
        "_layers": result.get("layers_used"),
        "_note": result.get("note"),
        **{y: series.get(str(y), 0.0) for y in [str(x) for x in years]},
    }
    # remove meta keys for clean series file? keep meta for audit; load_prodes skips _keys
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{ibge}_prodes.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {y: float(series.get(str(y), 0.0)) for y in [str(x) for x in years]}
