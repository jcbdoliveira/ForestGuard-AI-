"""
Motor de análise espacial e projeção multi-horizonte (3, 5, 8 e 10 anos).

Princípios:
1. Projeção municipal: modelo Ridge treinado de fato nas séries PRODES históricas
   (ver train.py). Métricas vêm de backtesting temporal — não são inventadas.
2. Distribuição espacial do risco: apenas features geométricas computáveis
   (distância a corredores rodoviários definidos, contido no polígono IBGE).
   NÃO inventa NDVI, fragmentação, slope SRTM nem taxa histórica por célula.
3. Área de floresta por célula: alocação uniforme do remanescente municipal
   entre células com centroide dentro do polígono oficial (sem mapa de cobertura
   espacial disponível no repositório).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..settings import settings, SUPPORTED_HORIZONS

# v16 (parecer M2): logging estruturado substitui o silenciamento de exceções
logger = logging.getLogger("forestguard.predict")
from ..validation import validate_horizons, validate_ibge_code
from ..sources.municipality import get_municipality, get_real_boundary_geojson
from .train import get_trained_model, project_municipality_series
from ..spatial.infer import score_cells as spatial_score_cells
from ..spatial.allocation import allocate_weighted_loss


def _point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    """Ray-casting para um anel GeoJSON (coordenadas longitude/latitude)."""
    if len(ring) < 3:
        return False
    inside = False
    p1x, p1y = ring[0]
    for i in range(1, len(ring) + 1):
        p2x, p2y = ring[i % len(ring)]
        if (p1y > y) != (p2y > y):
            xinters = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
            if x < xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _point_in_polygon(x: float, y: float, polygons: List[List[List[List[float]]]]) -> bool:
    """Testa Polygon/MultiPolygon GeoJSON, respeitando anéis internos (buracos).

    ``polygons`` é uma lista de polígonos; cada polígono contém o anel externo
    seguido de zero ou mais anéis internos. O ponto pertence se estiver dentro
    de pelo menos um anel externo e fora de todos os buracos desse polígono.
    """
    for rings in polygons:
        if not rings or not _point_in_ring(x, y, rings[0]):
            continue
        if not any(_point_in_ring(x, y, hole) for hole in rings[1:]):
            return True
    return False


def _min_dist_to_corridors(lon: float, lat: float, corridors: List[Dict[str, Any]]) -> float:
    """Distância aproximada em km a segmentos de corredor.

    v16 (parecer M3): fator de conversão de longitude corrigido por cos(latitude).
    Antes usava fator fixo 100 — erro sistemático de ~11% a 26°S (Mafra).
    """
    k_lon = math.cos(math.radians(lat)) * 111.0  # graus de lon → km na latitude dada
    min_d = 999.0
    for corr in corridors:
        pts = corr.get("coords") or []
        for i in range(len(pts) - 1):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                d = math.hypot((lon - x1) * k_lon, (lat - y1) * 111.0)
            else:
                t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / (dx * dx + dy * dy)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                d = math.hypot((lon - proj_x) * k_lon, (lat - proj_y) * 111.0)
            if d < min_d:
                min_d = d
    return round(min_d, 3)



def _low_variability_zero_series(hist: Dict[str, float], threshold: float = 0.80) -> bool:
    """True quando >80% dos valores anuais são zero; não infere ausência de dados."""
    values = [float(v) for v in hist.values()]
    if len(values) < 5:
        return False
    if any((not math.isfinite(v)) or v < 0 for v in values):
        return False
    return sum(v == 0.0 for v in values) / len(values) > threshold


def _apply_historical_mean_fallback(projection: Dict[str, Any], hist: Dict[str, float],
                                    remnant_ha: float, horizons: List[int]) -> None:
    """Substitui Ridge pela média histórica em séries extremamente esparsas."""
    vals = [max(0.0, float(v)) for v in hist.values()]
    mean_ha = float(sum(vals) / len(vals)) if vals else 0.0
    base_year = int(projection["base_year"])
    max_h = max(horizons) if horizons else 0
    annual = {str(base_year + i): round(mean_ha, 2) for i in range(1, max_h + 1)}
    projection["annual_projections_ha"] = annual
    projection["annual_cap_ha"] = round(mean_ha, 2)
    cumulative = {}
    for h in horizons:
        total = min(mean_ha * h, max(0.0, float(remnant_ha)) * 0.95)
        cumulative[f"{h}y"] = {
            "horizon_years": h, "target_year": base_year + h,
            "expected_suppressed_ha": round(total, 2),
            "percent_of_remnant": round(100.0 * total / max(float(remnant_ha), 1.0), 2),
        }
    projection["horizon_cumulative_ha"] = cumulative
    projection["projection_method"] = "historical_mean_fallback_low_variability"
    projection["warnings"] = ["Série Histórica de Baixa Variabilidade: mais de 80% dos valores anuais são zero; Ridge desativado e média histórica usada como referência conservadora."]

def classify_attention_level(probability: float) -> str:
    """v4: Baixo / Médio / Alto / Maior (CRITICO mantido como alias de Maior)."""
    if probability >= settings.major_threshold:
        return "CRITICO"  # Maior
    if probability >= settings.high_threshold:
        return "ALTO"
    if probability >= settings.moderate_threshold:
        return "MODERADO"  # Médio
    return "BAIXO"


def _recommend_action(attention_level: str, dominant_driver: str) -> str:
    if attention_level == "NAO_CONFIAVEL":
        return "Projeção sem skill contra a persistência neste município. Não usar como prioridade de fiscalização."
    if attention_level == "CRITICO":
        if "Rodovi" in dominant_driver or "Eixo" in dominant_driver:
            return "Fiscalização prioritária em acessos vicinais e abertura de clareiras."
        return "Monitoramento e vistoria de campo prioritária na área de maior pressão."
    if attention_level == "ALTO":
        return "Inspeção periódica de borda florestal e notificação preventiva."
    if attention_level == "MODERADO":
        return "Acompanhamento regular e incentivo a recomposição de Reserva Legal."
    return "Monitoramento regular anual."


def compute_dense_grid_dims(
    bbox: List[float],
    target_cells: int = None,
) -> Tuple[int, int]:
    """
    Calcula rows/cols para malha densa que preenche o bbox do município
    (o mapa final só desenha células com centroide dentro do polígono).
    """
    target = target_cells or settings.default_target_bbox_cells
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    aspect = lon_span / lat_span  # cols/rows
    rows = int(round(math.sqrt(target / aspect)))
    cols = int(round(rows * aspect))
    rows = max(settings.min_grid_side, min(settings.max_grid_side, rows))
    cols = max(settings.min_grid_side, min(settings.max_grid_side, cols))
    return rows, cols


def run_city_deforestation_analysis(
    ibge_code: str,
    horizons: Optional[List[int]] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    custom_pressure_multiplier: float = 1.0,
) -> Dict[str, Any]:
    code = validate_ibge_code(ibge_code)
    valid_horizons = validate_horizons(horizons or list(SUPPORTED_HORIZONS))
    muni = get_municipality(code)
    if not muni:
        raise ValueError(f"Município com código IBGE {code} não encontrado no catálogo.")

    hist = dict(muni.get("historic_prodes_deforestation_ha") or {})
    remnant_ha = float(muni["forest_remnant_ha"])
    data_provenance = "catalogo_legado_nao_observado"
    from ..sources.mapbiomas_catalog import mapbiomas_series_for_prodes

    observed = mapbiomas_series_for_prodes(code)
    if observed and observed.get("historic_prodes_deforestation_ha"):
        hist = {str(k): float(v) for k, v in observed["historic_prodes_deforestation_ha"].items()}
        if observed.get("forest_remnant_ha"):
            remnant_ha = float(observed["forest_remnant_ha"])
        data_provenance = "mapbiomas_col11_annual_forest_loss"
    if not hist:
        raise ValueError(
            f"Município {code} não possui série observada. "
            "Projeção requer perda anual MapBiomas ou série informada — não inventamos valores."
        )

    model = get_trained_model()
    low_variability_series = _low_variability_zero_series(hist)
    projection = project_municipality_series(
        hist=hist,
        remnant_ha=remnant_ha,
        model=model,
        horizons=valid_horizons,
        pressure_multiplier=custom_pressure_multiplier,
    )
    if low_variability_series:
        _apply_historical_mean_fallback(projection, hist, remnant_ha, valid_horizons)
    city_rel = dict((model.city_reliability or {}).get(code) or {})
    if not city_rel:
        city_rel = {
            "reliable": False,
            "reason": "municipio_ausente_do_walk_forward",
            "skill_rmse": None,
            "n_cutoffs": 0,
        }
    if data_provenance != "mapbiomas_col11_annual_forest_loss":
        city_rel = {**city_rel, "reliable": False, "reason": "serie_nao_observada_mapbiomas"}
    if low_variability_series:
        city_rel = {**city_rel, "reliable": False, "reason": "serie_historica_baixa_variabilidade", "fallback": "media_historica"}
    forecast_reliable = bool(city_rel.get("reliable"))
    projection["forecast_reliability"] = city_rel

    # --- Geometria ---
    boundary_geo = get_real_boundary_geojson(code)
    # Mantém a estrutura GeoJSON completa: polígonos separados e anéis internos.
    # Achatar os anéis tratava buracos como área municipal válida e podia gerar
    # células falsas em geometrias complexas.
    boundary_polygons: List[List[List[List[float]]]] = []
    if boundary_geo and boundary_geo.get("features"):
        geom = boundary_geo["features"][0].get("geometry") or {}
        if geom.get("type") == "Polygon":
            boundary_polygons = [geom.get("coordinates") or []]
        elif geom.get("type") == "MultiPolygon":
            boundary_polygons = geom.get("coordinates") or []

    min_lon, min_lat, max_lon, max_lat = muni["bbox"]
    if grid_rows is None or grid_cols is None:
        grid_rows, grid_cols = compute_dense_grid_dims(list(muni["bbox"]))
    lon_step = (max_lon - min_lon) / grid_cols
    lat_step = (max_lat - min_lat) / grid_rows
    corridors = muni.get("road_Corridors") or []
    localities = muni.get("locality_zones") or [f"Setor {i}" for i in range(grid_rows * grid_cols)]

    cells: List[Dict[str, Any]] = []
    inside_indices: List[int] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell_id = f"CELL_{code}_R{r:02d}C{c:02d}"
            c_min_lon = round(min_lon + c * lon_step, 5)
            c_max_lon = round(min_lon + (c + 1) * lon_step, 5)
            c_min_lat = round(min_lat + r * lat_step, 5)
            c_max_lat = round(min_lat + (r + 1) * lat_step, 5)
            centroid_lon = round((c_min_lon + c_max_lon) / 2.0, 5)
            centroid_lat = round((c_min_lat + c_max_lat) / 2.0, 5)

            is_inside = True
            if boundary_polygons:
                is_inside = _point_in_polygon(centroid_lon, centroid_lat, boundary_polygons)

            dist_road_km = _min_dist_to_corridors(centroid_lon, centroid_lat, corridors)
            # road_proximity_decay_km: parâmetro de engenharia (settings), não estimado pelo XGBoost
            _decay = float(getattr(settings, "road_proximity_decay_km", 4.2))
            road_prox = round(math.exp(-dist_road_km / _decay), 4) if corridors else 0.15

            # Localidade apenas deste município (lista do catálogo da cidade consultada)
            if len(localities) == 1:
                locality_name = localities[0]
            else:
                # Particiona a malha em faixas usando só as zones desta cidade
                loc_idx = int((r / max(grid_rows, 1)) * len(localities)) % len(localities)
                # desempate por coluna para espalhar nomes ao longo do E-W
                loc_idx = (loc_idx + (c * 3) // max(grid_cols, 1)) % len(localities)
                locality_name = localities[loc_idx]

            idx = len(cells)
            if is_inside:
                inside_indices.append(idx)

            # Distância real à rede OSM fornecida, quando a célula está dentro
            # da cobertura do GeoJSON. Fora dela, mantém a distância dos
            # corredores municipais já existentes.
            road_feature_km = dist_road_km
            osm_distance = None
            road_class_features = {}
            try:
                from ..spatial.road_distance import road_distance_features, roads_path_for_municipality
                _roads_path = roads_path_for_municipality(code)
                if _roads_path is not None:
                    road_class_features = road_distance_features(centroid_lon, centroid_lat, roads_path=_roads_path)
                else:
                    road_class_features = {"road_dist_km": None, "road_dist_major_km": None,
                                           "road_dist_secondary_km": None, "road_dist_local_km": None,
                                           "road_dist_track_km": None}
                osm_distance = road_class_features.get("road_dist_km")
                if osm_distance is not None:
                    road_feature_km = osm_distance
            except Exception as exc:
                logger.warning("Distâncias OSM por categoria indisponíveis para célula %s: %s", cell_id, exc)
            # O mapa e o modelo usam a mesma rede viária. Distâncias por classe
            # permitem que o XGBoost aprenda relações distintas para vias
            # principais, secundárias/terciárias, locais e tracks florestais.
            dist_road_km = road_feature_km
            road_prox = round(math.exp(-dist_road_km / _decay), 4) if (corridors or osm_distance is not None) else 0.15
            spatial_features = {"road_dist_km": road_feature_km}
            for _key in ("road_dist_major_km", "road_dist_secondary_km", "road_dist_local_km", "road_dist_track_km"):
                _value = road_class_features.get(_key)
                spatial_features[_key] = _value

            cells.append({
                "cell_id": cell_id,
                "row": r,
                "col": c,
                "locality": f"{locality_name} (Setor R{r+1}C{c+1})",
                "centroid": [centroid_lat, centroid_lon],
                "bbox": [c_min_lat, c_min_lon, c_max_lat, c_max_lon],
                "is_inside_boundary": is_inside,
                "cell_area_ha": None,
                "initial_forest_ha": None,
                "dist_road_km": dist_road_km,
                "road_proximity_index": road_prox,
                "features": spatial_features,
                "dominant_driver": "Pressão histórica e contexto espacial (inclui distância à rede viária)",
                "scenarios": {},
            })

    n_inside = max(len(inside_indices), 1)
    cell_area_ha = round(float(muni["total_area_ha"]) / (grid_rows * grid_cols), 1)
    forest_per_inside = round(remnant_ha / n_inside, 2)

    for i, cell in enumerate(cells):
        cell["cell_area_ha"] = cell_area_ha
        if cell["is_inside_boundary"]:
            cell["initial_forest_ha"] = forest_per_inside
        else:
            cell["initial_forest_ha"] = 0.0

    # Densidade de desmate histórico (PRODES TerraBrasilis) por célula — dado real
    prodes_density = {i: 0.0 for i in inside_indices}
    try:
        from ..sources.mapbiomas_fetch import load_municipality_boundary, _wfs_yearly_features, _centroid_of_geom, WFS_LAYERS, _bbox_of_geom
        boundary_geom = None
        # reusa polígono do município se disponível em disco
        from pathlib import Path as _P
        bpath = _P(__file__).resolve().parents[3] / "data" / "boundaries" / f"{code}.geojson"
        if bpath.exists():
            import json as _json
            _gj = _json.loads(bpath.read_text(encoding="utf-8"))
            feats = _gj.get("features") or []
            boundary_geom = (feats[0].get("geometry") if feats else _gj if _gj.get("type") in ("Polygon", "MultiPolygon") else None)
        if boundary_geom is not None:
            bbox_g = _bbox_of_geom(boundary_geom)
            if bbox_g:
                dens_pts = []  # (lon, lat, ha)
                for _ws, layer in WFS_LAYERS[:1]:  # mata atlântica first; others if empty
                    for ft in _wfs_yearly_features(layer, bbox_g, count=3000):
                        props = ft.get("properties") or {}
                        g = ft.get("geometry") or {}
                        cen = _centroid_of_geom(g)
                        if cen is None:
                            continue
                        try:
                            ha = float(props.get("area_km") or 0) * 100.0
                        except Exception:
                            ha = 0.0
                        if ha > 0:
                            dens_pts.append((cen[0], cen[1], ha))
                    if dens_pts:
                        break
                # atribui a cada célula a soma de ha PRODES cujo centroide cai no bbox da célula
                for i in inside_indices:
                    c = cells[i]
                    bb = c["bbox"]  # [min_lat, min_lon, max_lat, max_lon]
                    s = 0.0
                    for lon, lat, ha in dens_pts:
                        if bb[1] <= lon <= bb[3] and bb[0] <= lat <= bb[2]:
                            s += ha
                    prodes_density[i] = s
    except Exception as exc:  # v16 (parecer M2): falha registrada, não silenciada
        logger.warning("Densidade PRODES/TerraBrasilis indisponível para %s: %s", code, exc)

    dens_max = max(prodes_density.values()) if prodes_density else 0.0
    has_prodes_spatial = dens_max > 0

    weights = []
    for i in inside_indices:
        # v18: sem distância/proximidade de estradas na alocação espacial.
        # Se houver histórico PRODES espacial, ele orienta a distribuição; caso
        # contrário, peso uniforme evita excluir áreas distantes de vias.
        if has_prodes_spatial:
            w = 0.05 + 0.95 * (prodes_density[i] / dens_max)
        else:
            w = 1.0
        weights.append(w)
        cells[i]["prodes_hist_ha_in_cell"] = round(prodes_density.get(i, 0.0), 2)
    weight_sum = sum(weights) or 1.0
    norm_weights = {inside_indices[j]: weights[j] / weight_sum for j in range(len(inside_indices))}

    # Modelo espacial (XGBoost): seleciona somente um artefato cujo manifesto
    # identifica este IBGE. O resolvedor permite a pasta municipal ou o artefato
    # legado na raiz apenas quando o código IBGE do manifesto coincide; nunca
    # pontua um município com modelo de outro município. Sem imputação de features.
    spatial_info = spatial_score_cells(
        cells,
        ibge_code=str(code),
        allow_impute=False,
    )
    if isinstance(spatial_info, dict) and "error" in spatial_info:
        logger.warning("Modelo espacial do IBGE %s falhou ao carregar: %s", code, spatial_info["error"])
        spatial_info = None
    spatial_scores = (spatial_info or {}).get("scores") if spatial_info and "scores" in (spatial_info or {}) else None

    horizon_summaries: Dict[str, Any] = {}
    base_year = projection["base_year"]

    for h in valid_horizons:
        h_key = f"{h}y"
        cum = projection["horizon_cumulative_ha"][h_key]
        total_loss_h = float(cum["expected_suppressed_ha"])

        cell_losses: Dict[int, float] = {}
        cell_probs: Dict[int, float] = {}

        # v17.4: alocação por pesos com limite explícito por célula. Os pesos
        # expressam risco relativo e não hectares. Quando não há modelo espacial,
        # preserva-se a ponderação geométrica/histórica já calculada acima.
        if spatial_scores:
            raw_sp = {i: max(0.0, float(spatial_scores.get(i, 0.0))) for i in inside_indices}
            allocation_weights = [raw_sp[i] for i in inside_indices]
            allocation_method = "modelo_espacial_municipal_com_teto_por_celula"
            for i in inside_indices:
                prob = min(0.99, max(0.01, raw_sp[i]))
                cell_probs[i] = round(prob, 4)
        else:
            allocation_weights = [max(0.0, float(norm_weights.get(i, 0.0))) for i in inside_indices]
            allocation_method = (
                "densidade_prodes_e_proximidade_rodoviaria_com_teto_por_celula"
                if has_prodes_spatial
                else "proximidade_rodoviaria_com_teto_por_celula"
            )
            for i in inside_indices:
                # Mantém a heurística de atenção anterior, calculada com o valor
                # que será de fato alocado após respeitar o teto da célula.
                cell_probs[i] = 0.02

        allocation = allocate_weighted_loss(
            requested_ha=total_loss_h,
            weights=allocation_weights,
            capacities_ha=[max(0.0, float(cells[i].get("initial_forest_ha") or 0.0)) for i in inside_indices],
        )
        for pos, i in enumerate(inside_indices):
            # Precisão interna de 6 casas reduz erro de soma sem dar aparência
            # de que a resolução dos dados é sub-hectare.
            loss_i = round(float(allocation["allocations_ha"][pos]), 6)
            cell_losses[i] = loss_i
            if not spatial_scores:
                forest_i = max(float(cells[i].get("initial_forest_ha") or 0.0), 1e-6)
                raw = loss_i / forest_i
                cell_probs[i] = round(min(0.95, max(0.02, float(math.tanh(raw * 2.5)))), 4)

        rounded_allocated_ha = sum(cell_losses.values())
        requested_ha = float(allocation["requested_ha"])
        rounded_unallocated_ha = max(0.0, requested_ha - rounded_allocated_ha)
        allocation_diagnostics = {
            "method": allocation_method,
            "requested_ha": round(requested_ha, 6),
            "allocated_ha": round(rounded_allocated_ha, 6),
            "unallocated_ha": round(rounded_unallocated_ha, 6),
            "available_forest_ha_in_grid": round(float(allocation["available_forest_ha"]), 6),
            "capped_cells_count": int(allocation["capped_cells_count"]),
            "within_capacity": bool(allocation["within_capacity"] and all(
                cell_losses[i] <= float(cells[i].get("initial_forest_ha") or 0.0) + 1e-6
                for i in inside_indices
            )),
            "balanced": abs(rounded_allocated_ha + rounded_unallocated_ha - requested_ha) <= 1e-6,
            "capacity_basis": "remanescente municipal distribuído uniformemente entre células internas; não equivale a raster observado por célula",
        }

        for i, cell in enumerate(cells):
            if i not in cell_probs:
                cell_probs[i] = 0.02
                cell_losses[i] = 0.0

        attention_points = []
        for i, cell in enumerate(cells):
            if not cell["is_inside_boundary"]:
                continue
            prob = cell_probs[i]
            # Risco espacial (XGBoost do município) é independente da confiabilidade
            # da projeção municipal Ridge. Só marca NAO_CONFIAVEL quando não há
            # escore espacial e a projeção municipal também não é confiável.
            if spatial_scores and i in cell_probs:
                level = classify_attention_level(prob)
            elif not forecast_reliable:
                level = "NAO_CONFIAVEL"
            else:
                level = classify_attention_level(prob)
            loss_i = cell_losses[i]
            remaining = round(max(0.0, (cell["initial_forest_ha"] or 0) - loss_i), 2)
            # Faixa de sensibilidade heurística (±25%) — NÃO é intervalo de confiança estatístico
            _band = float(getattr(settings, "sensitivity_band_factor", 0.25))
            sens_lo = round(max(0.0, prob * (1.0 - _band)), 4)
            sens_hi = round(min(0.99, prob * (1.0 + _band)), 4)
            cell["scenarios"][h_key] = {
                "horizon_years": h,
                "target_year": base_year + h,
                "probability": prob,
                # nomes honestos
                "lower_sensitivity_bound": sens_lo,
                "upper_sensitivity_bound": sens_hi,
                # aliases legados (compat UI/PDF) — mesmos valores heurísticos
                "ci_lower": sens_lo,
                "ci_upper": sens_hi,
                "attention_level": level,
                "expected_suppressed_ha": loss_i,
                "projected_remaining_forest_ha": remaining,
            }
            attention_points.append({
                "rank": 0,
                "cell_id": cell["cell_id"],
                "locality": cell["locality"],
                "coordinates": cell["centroid"],
                "probability": prob,
                "sensitivity_band": [sens_lo, sens_hi],
                "ci_90": [sens_lo, sens_hi],  # legado; não é CI 90% estatístico
                "attention_level": level,
                "expected_suppressed_ha": loss_i,
                "initial_forest_ha": cell["initial_forest_ha"],
                "dist_road_km": cell["dist_road_km"],
                "dominant_driver": cell["dominant_driver"],
                "recommended_action": _recommend_action(level, cell["dominant_driver"]),
            })

        attention_points.sort(key=lambda p: p["probability"], reverse=True)
        for rank, pt in enumerate(attention_points, start=1):
            pt["rank"] = rank
        top10 = attention_points[:10]

        levels = [p["attention_level"] for p in attention_points]
        mean_prob = (
            round(sum(p["probability"] for p in attention_points) / len(attention_points), 4)
            if attention_points
            else 0.0
        )
        max_prob = max((p["probability"] for p in attention_points), default=0.0)

        horizon_summaries[h_key] = {
            "horizon_years": h,
            "target_year": base_year + h,
            "mean_probability": mean_prob,
            "max_probability": max_prob,
            "critical_points_count": levels.count("CRITICO"),
            "high_points_count": levels.count("ALTO"),
            "moderate_points_count": levels.count("MODERADO"),
            "low_points_count": levels.count("BAIXO"),
            "expected_suppressed_area_ha": total_loss_h,
            "percent_of_remnant_lost": cum["percent_of_remnant"],
            "spatial_allocation": allocation_diagnostics,
            "top_attention_points": top10,
        }

    # Labels legíveis para importância de features (XGBoost — não são pesos causais)
    _FEATURE_LABELS = {
        "road_dist_km": "Distância a corredores rodoviários (km)",
        "slope_deg": "Declividade (graus)",
        "elev_m": "Elevação (m)",
        "forest_neighbor_frac": "Fração de floresta na vizinhança",
        "built_surface": "Superfície construída (proxy GHSL)",
        "years_stable_forest": "Anos de floresta estável",
        "road_proximity_index": "Índice de proximidade rodoviária",
    }

    spatial_fi_raw = (spatial_info or {}).get("feature_importance") or []
    feature_importance = []
    for item in spatial_fi_raw:
        feat = item.get("feature") or ""
        feature_importance.append(
            {
                "feature": feat,
                "label": _FEATURE_LABELS.get(feat, feat),
                "importance": float(item.get("importance") or 0.0),
            }
        )

    # Métricas espaciais (quando o XGBoost espacial carregou) têm prioridade na UI.
    # Mantém as métricas do Ridge municipal com nomes originais (mae_ha, etc.) para testes/compat.
    spatial_metrics = (spatial_info or {}).get("metrics_test") or {}
    spatial_policy = (spatial_info or {}).get("probability_policy") or {}
    metrics_apply = bool(spatial_policy.get("metrics_apply_to_displayed_map"))
    metrics_for_ui = dict(model.metrics or {})
    metrics_for_ui["skill_rmse"] = (model.walk_forward.get("overall") or {}).get("skill_rmse")
    metrics_for_ui["n_cutoffs"] = (model.walk_forward.get("overall") or {}).get("n_cutoffs")
    if spatial_scores and spatial_metrics and metrics_apply:
        metrics_for_ui.update(
            {
                "roc_auc": float(spatial_metrics.get("roc_auc") or 0.0),
                "pr_auc": float(spatial_metrics.get("pr_auc") or 0.0),
                "brier_score": float(spatial_metrics.get("brier") or 0.0),
                "recall_at_top10_hotspots": float(spatial_metrics.get("recall_at_top10pct") or 0.0),
            }
        )
    # Sem raster na grade do mapa, a AUC do painel não entra no bloco principal.
    metrics_for_ui.pop("ece_calibration_error", None)
    if not metrics_apply:
        metrics_for_ui.pop("roc_auc", None)
        metrics_for_ui.pop("pr_auc", None)
        metrics_for_ui.pop("brier_score", None)
        metrics_for_ui.pop("recall_at_top10_hotspots", None)

    return {
        "model_metadata": {
            "model_version": settings.model_version,
            "algorithm": model.model_type,
            # v16 (parecer A2): origem do número exibido como "probabilidade" no mapa
            "probability_source": (
                spatial_policy.get("probability_source")
                if spatial_scores
                else "geometric_weighting_tanh"
            ),
            "forecast_reliability": city_rel,
            "historical_series_warning": (projection.get("warnings") or [None])[0],
            "projection_method": projection.get("projection_method", model.model_type),
            "walk_forward": model.walk_forward,
            "training": {
                "n_samples": model.train_n,
                "n_lags": model.n_lags,
                "ridge_alpha": model.alpha,
                "feature_names": model.feature_names,
                "last_year_in_data": model.last_year_in_data,
            },
            "metrics_validation": metrics_for_ui,
            "validation_holdout_details": model.validation_details[:20],
            "supported_horizons_years": list(valid_horizons),
            "crs_storage": settings.crs_storage,
            "crs_metric": settings.crs_metric,
            "data_policy": (
                "Projeção municipal treinada em séries PRODES reais. "
                "Distribuição espacial: modelo XGBoost específico do município se disponível; "
                "senão densidade PRODES por célula e/ou proximidade rodoviária. Perda alocada "
                "é limitada pela capacidade disponível de cada célula."
            ),
            "spatial_model": (
                {
                    "loaded": True,
                    "path": spatial_info.get("model_path"),
                    "type": spatial_info.get("model_type"),
                    "data_source": spatial_info.get("data_source")
                    or (
                        "gee_mapbiomas_panel"
                        if spatial_scores
                        else "unavailable"
                    ),
                    # v16 (parecer C1): classificação vem do metadata do treino —
                    # só chega aqui se o modelo for do próprio município.
                    "model_class": (
                        f"municipality_trained_{spatial_info.get('model_type', 'unknown')}"
                        if spatial_scores else "municipality_model_not_applied_missing_observed_features"
                    ),
                    "scored_cell_count": len(spatial_scores or {}),
                    "unscored_cell_count": len(spatial_info.get("unscored_cells") or []),
                    "no_score_reason": (
                        None if spatial_scores else "Sem score multi-feature: features observadas insuficientes; não houve imputação por mediana."
                    ),
                    "metrics_test": spatial_info.get("metrics_test"),
                    "features": spatial_info.get("feature_names"),
                    "feature_importance": spatial_fi_raw,
                    "features_imputed": bool(spatial_info.get("features_imputed")),
                    "imputed_features": spatial_info.get("imputed_features") or [],
                    "unscored_cells": spatial_info.get("unscored_cells") or [],
                    "map_feature_mode": spatial_info.get("map_feature_mode", "observed_features_only"),
                    "metrics_apply_to_displayed_map": metrics_apply,
                    "probability_policy": spatial_policy,
                    "walk_forward": spatial_info.get("walk_forward"),
                    "training_note": spatial_info.get("training_note")
                    or "Modelo espacial do município. AUC só vale na grade em que as features foram observadas.",
                    "selection": spatial_info.get("selection") or {},
                }
                if spatial_info and isinstance(spatial_info, dict) and "error" not in spatial_info
                else {"loaded": False, "reason": "modelo ausente ou falha na inferência"}
            ),
            # Importância das features no modelo XGBoost (artefato de treinamento; não pesos causais)
            "feature_importance": feature_importance,
            "limitations": [
                "A série usada na projeção é a perda anual de floresta do MapBiomas Coleção 11 (diferença do estoque), não o incremento PRODES oficial, salvo série informada pelo usuário quando não há MapBiomas.",
                "O catálogo hardcoded anterior tinha séries com o mesmo formato reescalado e não entra no treino.",
                "Confiabilidade exige vantagem mínima de 10% no RMSE sobre o melhor baseline (persistência ou média histórica), pelo menos 5 cortes e 10 observações; caso contrário, NAO_CONFIAVEL.",
                "Séries com mais de 80% de zeros usam média histórica como fallback e são marcadas como baixa variabilidade; zeros não são automaticamente considerados dados ausentes.",
                "Faixa ±25% (ci_90 / sensitivity_band) não é intervalo de confiança.",
                "Se o mapa imputa features com a mediana do treino, a ROC-AUC do painel GEE não descreve esse mapa.",
                "Platt só é aplicado quando foi ajustado no ano de validação e nenhuma feature da célula foi imputada.",
                "Importância de features não é efeito causal.",
                "A distribuição espacial respeita tetos por célula, mas o remanescente por célula é uniforme enquanto não houver raster de cobertura observado; pesos de risco não são área observada.",
            ],
            "official_sources": [
                "IBGE Malhas Municipais (GeoJSON em data/boundaries/)",
                "INPE PRODES (séries anuais em real_prodes_data.py)",
                "IBGE Censo / projeções populacionais (quando disponíveis no catálogo)",
            ],
        },
        "municipality": {
            "ibge_code": muni["ibge_code"],
            "name": muni["name"],
            "uf": muni["uf"],
            "biome": muni["biome"],
            "total_area_ha": muni["total_area_ha"],
            "forest_remnant_ha": remnant_ha,
            "data_provenance": data_provenance,
            "forecast_reliability": city_rel,
            "census_pop_2022": muni.get("census_pop_2022"),
            "proj_pop_2026": muni.get("proj_pop_2026"),
            "historic_prodes_deforestation_ha": hist,
            "primary_loss_drivers": muni.get("primary_loss_drivers", []),
            "center": [muni["center_lon"], muni["center_lat"]],
            "bbox": muni["bbox"],
            "road_corridors": corridors,
            "grid_dimensions": {
                "rows": grid_rows,
                "cols": grid_cols,
                "total_cells": len(cells),
                "cells_inside_boundary": n_inside,
            },
            "has_real_boundary_polygon": bool(boundary_polygons),
        },
        "municipal_projection": projection,
        "horizon_summaries": horizon_summaries,
        "cells": cells,
    }
