#!/usr/bin/env python3
"""
Adiciona um município ao catálogo ForestGuard a partir do nome oficial.

Uso (qualquer um destes formatos):
  python3 scripts/add_city.py SEARA-SC
  python3 scripts/add_city.py --SEARA-SC
  python3 scripts/add_city.py --city SEARA-SC
  python3 scripts/add_city.py "Seara-SC"
  python3 scripts/add_city.py Seara/SC

O que é REAL (sempre):
  - Código IBGE, nome oficial, UF (API IBGE Localidades)
  - Polígono GeoJSON oficial (API IBGE Malhas) → data/boundaries/{ibge}.geojson
  - BBOX, centroide e área (shoelace no polígono)

O que NÃO é inventado:
  - Série PRODES / remanescente florestal
  - População (só se a API IBGE responder)

PRODES / MapBiomas:
  - Se passar --prodes-json arquivo.json com série anual, usa esses valores.
  - Caso contrário, tenta MapBiomas Coleção 11 local
    (data/mapbiomas/forest_by_municipality.json — gerado a partir do XLSX
    https://github.com/jcbdoliveira/MAPBiomas/...).
  - Sempre treina modelo espacial específico da cidade em
    data/spatial/models/{ibge}/ (painel sintético calibrado pela taxa local
    quando não há painel GEE real).

Exemplo de --prodes-json:
  {"2018": 12.5, "2019": 15.0, "2020": 18.2, "2021": 20.1, "2022": 22.0, "2023": 19.5, "2024": 21.3}
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BOUNDARIES_DIR = ROOT_DIR / "data" / "boundaries"
SRC_PY_MUNI = ROOT_DIR / "src" / "supressao_vegetacao" / "sources" / "municipality.py"
SRC_TS_MUNI = ROOT_DIR / "src" / "sources" / "municipality.ts"
SRC_PY_PRODES = ROOT_DIR / "src" / "supressao_vegetacao" / "sources" / "real_prodes_data.py"
SRC_TS_ML = ROOT_DIR / "src" / "mlEngine.ts"

UA = "ForestGuard-AI/3.0 (research; +https://github.com/local)"


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def parse_name_uf(arg: str) -> Tuple[str, str]:
    cleaned = arg.strip().lstrip("-")
    if "-" in cleaned:
        parts = cleaned.rsplit("-", 1)
        city_name, uf = parts[0].strip(), parts[1].strip().upper()
    elif "/" in cleaned:
        parts = cleaned.rsplit("/", 1)
        city_name, uf = parts[0].strip(), parts[1].strip().upper()
    else:
        tokens = cleaned.split()
        if len(tokens) >= 2 and len(tokens[-1]) == 2:
            city_name, uf = " ".join(tokens[:-1]), tokens[-1].upper()
        else:
            raise ValueError(
                f"Formato inválido: '{arg}'. Use NOME-UF (ex: SEARA-SC, 'Seara-SC', Seara/SC)."
            )
    if len(uf) != 2 or not uf.isalpha():
        raise ValueError(f"UF inválida: '{uf}'")
    return city_name, uf


def http_get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "application/json, application/vnd.geo+json, */*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def fetch_ibge_city_info(city_name: str, uf: str) -> Dict[str, Any]:
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    all_cities = http_get_json(url, timeout=60)
    city_norm = strip_accents(city_name).lower().replace(" ", "").replace("'", "").replace("-", "")
    uf_upper = uf.upper()
    exact = []
    partial = []
    for c in all_cities:
        micro = c.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}
        uf_obj = meso.get("UF") or {}
        c_uf = uf_obj.get("sigla") or ""
        if not c_uf:
            ri = c.get("regiao-imediata") or {}
            rinter = ri.get("regiao-intermediaria") or {}
            uf_obj = rinter.get("UF") or {}
            c_uf = uf_obj.get("sigla") or ""
        if str(c_uf).upper() != uf_upper:
            continue
        c_name_norm = (
            strip_accents(c["nome"]).lower().replace(" ", "").replace("'", "").replace("-", "")
        )
        if c_name_norm == city_norm:
            exact.append(c)
        elif city_norm in c_name_norm or c_name_norm in city_norm:
            partial.append(c)
    if exact:
        return exact[0]
    if partial:
        return partial[0]
    raise RuntimeError(
        f"Município '{city_name}'/{uf} não encontrado na base IBGE de municípios."
    )


def download_geojson_boundary(ibge_code: str) -> Dict[str, Any]:
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BOUNDARIES_DIR / f"{ibge_code}.geojson"
    url = (
        f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{ibge_code}"
        f"?formato=application/vnd.geo+json"
    )
    geojson_data = http_get_json(url, timeout=60)
    out_file.write_text(json.dumps(geojson_data, ensure_ascii=False), encoding="utf-8")
    return geojson_data


def _iter_rings(geojson: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    rings: List[List[Tuple[float, float]]] = []

    def handle_geom(geom: Dict[str, Any]) -> None:
        if not geom:
            return
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                rings.append([(float(p[0]), float(p[1])) for p in ring])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    rings.append([(float(p[0]), float(p[1])) for p in ring])

    if geojson.get("features"):
        for f in geojson["features"]:
            handle_geom(f.get("geometry") or {})
    else:
        handle_geom(geojson)
    return rings


def polygon_area_ha(ring: List[Tuple[float, float]]) -> float:
    """Área aproximada em ha (equiretangular local)."""
    if len(ring) < 3:
        return 0.0
    lat0 = sum(p[1] for p in ring) / len(ring)
    lon0 = sum(p[0] for p in ring) / len(ring)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    pts = [((p[0] - lon0) * m_per_deg_lon, (p[1] - lat0) * m_per_deg_lat) for p in ring]
    area = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0 / 10_000.0  # m² → ha


def compute_bbox_center_area(geojson: Dict[str, Any]) -> Tuple[List[float], Tuple[float, float], float]:
    rings = _iter_rings(geojson)
    if not rings:
        raise RuntimeError("GeoJSON sem coordenadas — malha IBGE inválida.")
    all_pts = [p for ring in rings for p in ring]
    min_lon = min(p[0] for p in all_pts)
    max_lon = max(p[0] for p in all_pts)
    min_lat = min(p[1] for p in all_pts)
    max_lat = max(p[1] for p in all_pts)
    center_lon = round((min_lon + max_lon) / 2.0, 5)
    center_lat = round((min_lat + max_lat) / 2.0, 5)
    # área = soma dos anéis exteriores (1º anel de cada polígono); ignora furos por simplicidade
    area_ha = 0.0
    seen = set()
    for ring in rings:
        key = (round(ring[0][0], 5), round(ring[0][1], 5), len(ring))
        if key in seen:
            continue
        seen.add(key)
        area_ha += polygon_area_ha(ring)
    area_ha = round(area_ha, 1)
    bbox = [round(min_lon, 4), round(min_lat, 4), round(max_lon, 4), round(max_lat, 4)]
    return bbox, (center_lat, center_lon), area_ha


def infer_biome(uf: str) -> str:
    uf = uf.upper()
    if uf in {"AM", "PA", "AC", "RO", "RR", "AP"}:
        return "Amazônia"
    if uf in {"MT", "TO"}:
        return "Cerrado / Amazônia"
    if uf in {"GO", "DF", "MS"}:
        return "Cerrado"
    if uf in {"SC", "PR", "RS"}:
        return "Mata Atlântica (Floresta Ombrófila)"
    if uf in {"SP", "RJ", "ES", "MG"}:
        return "Mata Atlântica"
    if uf in {"BA", "PE", "CE", "RN", "PB", "AL", "SE", "PI", "MA"}:
        return "Caatinga / Mata Atlântica"
    return "Não classificado automaticamente"


def try_fetch_population(ibge_code: str) -> Optional[int]:
    """Tenta estimativa populacional IBGE (pode falhar — retorna None, não inventa)."""
    # indicador 29171 = população estimada (série histórica por município; código com 6 dígitos às vezes)
    for code in (ibge_code, ibge_code[:6]):
        url = f"https://servicodados.ibge.gov.br/api/v1/pesquisas/indicadores/29171/resultados/{code}"
        try:
            data = http_get_json(url, timeout=20)
            if not data:
                continue
            res = data[0].get("res") or []
            if not res:
                continue
            series = res[0].get("res") or {}
            if not series:
                continue
            last_year = max(series.keys(), key=lambda y: int(y))
            return int(float(series[last_year]))
        except Exception:
            continue
    return None


def fetch_osm_road_corridors(bbox: List[float], timeout: int = 45) -> List[Dict[str, Any]]:
    """
    Baixa eixos rodoviários reais via Overpass API (OpenStreetMap).
    Filtra highway=primary|secondary|trunk|motorway|tertiary.
    Se falhar, retorna lista vazia (predict usará densidade PRODES, não cruz inventada).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # pequena margem
    pad = 0.01
    south, west = min_lat - pad, min_lon - pad
    north, east = max_lat + pad, max_lon + pad
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({south},{west},{north},{east});
    );
    out geom;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    data = None
    for ep in endpoints:
        try:
            req = urllib.request.Request(
                ep,
                data=query.encode("utf-8"),
                headers={"User-Agent": "ForestGuard/8.0", "Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception:
            continue
    if not data:
        return []
    corridors: List[Dict[str, Any]] = []
    for el in data.get("elements") or []:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = [[round(float(p["lon"]), 5), round(float(p["lat"]), 5)] for p in geom]
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("ref") or f"OSM way {el.get('id')}"
        hwy = tags.get("highway", "road")
        corridors.append({"name": f"{name} ({hwy})", "coords": coords, "source": "OpenStreetMap/Overpass"})
        if len(corridors) >= 40:
            break
    return corridors


def default_road_corridors(bbox: List[float], center: Tuple[float, float]) -> List[Dict[str, Any]]:
    """
    Preferência: vias OSM reais. Sem fallback de cruz pelo centroide (gerava mapa artificial).
    """
    roads = fetch_osm_road_corridors(bbox)
    if roads:
        print(f"✓ Corredores rodoviários OSM: {len(roads)} vias")
        return roads
    print("• OSM indisponível — sem eixos inventados; risco espacial usará densidade PRODES se disponível")
    return []


def write_prodes_template(path: Path, city_label: str = "") -> None:
    """
    Cria JSON modelo (anos vazios) para o usuário preencher com ha REAIS (MapBiomas/PRODES).
    Não grava valores inventados de desmate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "_instrucoes": (
            "Substitua null pelos hectares REAIS de desmate/supressão por ano "
            "(MapBiomas estatísticas municipais ou MapBiomas Alerta). "
            "Remova a chave _instrucoes antes de rodar de novo. "
            "NÃO use 0.0 como placeholder — zera a projeção ML."
        ),
        "_municipio": city_label or path.stem,
        "2018": None,
        "2019": None,
        "2020": None,
        "2021": None,
        "2022": None,
        "2023": None,
        "2024": None,
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_prodes_json(path: Optional[str], city_label: str = "") -> Optional[Dict[str, float]]:
    """
    Carrega série {ano: ha}. Se o arquivo não existir, cria um template e orienta o usuário.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        write_prodes_template(p, city_label=city_label)
        raise FileNotFoundError(
            f"--prodes-json não existia. Criei o modelo em:\n"
            f"  {p.resolve()}\n\n"
            f"1) Abra o arquivo e preencha cada ano com hectares REAIS (MapBiomas).\n"
            f"2) Apague a chave \"_instrucoes\" (e \"_municipio\" se quiser).\n"
            f"3) Rode o mesmo comando de novo:\n"
            f"   python scripts/add_city.py {city_label or 'NOME-UF'} "
            f"--prodes-json {path} --forest-remnant-ha <ha>"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("prodes-json deve ser um objeto {\"YYYY\": ha, ...}")

    # Rejeita template ainda não preenchido
    if data.get("_instrucoes") is not None:
        raise ValueError(
            f"O arquivo {p} ainda é o modelo gerado automaticamente.\n"
            f"Preencha os anos com ha reais (MapBiomas), remova \"_instrucoes\" e rode de novo."
        )

    out: Dict[str, float] = {}
    for k, v in data.items():
        if str(k).startswith("_"):
            continue
        year = str(k).strip()
        if not re.fullmatch(r"20\d{2}", year):
            raise ValueError(f"Ano inválido na série PRODES: {k}")
        if v is None:
            raise ValueError(
                f"Ano {year} está null no {p}. Preencha com hectares reais ou remova a chave."
            )
        out[year] = float(v)

    if not out:
        raise ValueError(f"Nenhum ano válido em {p}")

    # Aviso se série toda zero (mascara análise)
    if all(abs(v) < 1e-12 for v in out.values()):
        print(
            "AVISO: série PRODES toda zerada — a projeção ML será artificialmente baixa. "
            "Prefira hectares reais do MapBiomas.",
            file=sys.stderr,
        )
    return out


def _py_repr_str_list(items: List[str], indent: int = 12) -> str:
    pad = " " * indent
    inner = ",\n".join(f'{pad}"{x}"' for x in items)
    return "[\n" + inner + "\n" + (" " * (indent - 4)) + "]"


def add_to_python_catalogs(
    city_dict: Dict[str, Any],
    prodes_dict: Optional[Dict[str, Any]],
) -> None:
    code = city_dict["ibge_code"]
    py_code_entry = f'''    "{code}": {{
        "ibge_code": "{city_dict['ibge_code']}",
        "name": "{city_dict['name']}",
        "uf": "{city_dict['uf']}",
        "biome": "{city_dict['biome']}",
        "total_area_ha": {city_dict['total_area_ha']},
        "forest_remnant_ha": {city_dict['forest_remnant_ha']},
        "center_lat": {city_dict['center_lat']},
        "center_lon": {city_dict['center_lon']},
        "bbox": {city_dict['bbox']},
        "base_annual_loss_rate": {city_dict['base_annual_loss_rate']},
        "road_Corridors": {json.dumps(city_dict['road_Corridors'], ensure_ascii=False)},
        "locality_zones": {_py_repr_str_list(city_dict['locality_zones'])}
    }},'''

    content_muni = SRC_PY_MUNI.read_text(encoding="utf-8")
    if f'"{code}":' not in content_muni:
        target = "MUNICIPALITIES_CATALOG: Dict[str, Dict[str, Any]] = {"
        if target not in content_muni:
            raise RuntimeError("Não achei MUNICIPALITIES_CATALOG em municipality.py")
        content_muni = content_muni.replace(target, target + "\n" + py_code_entry, 1)
        SRC_PY_MUNI.write_text(content_muni, encoding="utf-8")
        print(f"✓ Catálogo Python: {SRC_PY_MUNI.name}")
    else:
        print(f"• Já existia em {SRC_PY_MUNI.name} ({code})")

    if prodes_dict is None:
        print("• Série PRODES não informada — não gravo valores inventados em real_prodes_data.py")
        return

    content_prodes = SRC_PY_PRODES.read_text(encoding="utf-8")
    if f'"{code}":' in content_prodes:
        print(f"• Já existia em {SRC_PY_PRODES.name} ({code})")
        return

    hist_lines = ",\n".join(
        f'            "{y}": {prodes_dict["historic_prodes_deforestation_ha"][y]}'
        for y in sorted(prodes_dict["historic_prodes_deforestation_ha"].keys())
    )
    drivers = prodes_dict.get("primary_loss_drivers") or []
    drivers_repr = json.dumps(drivers, ensure_ascii=False, indent=12)
    pop = prodes_dict.get("census_pop_2022")
    pop_repr = "None" if pop is None else str(int(pop))
    prodes_entry = f'''    "{code}": {{
        "ibge_code": "{prodes_dict['ibge_code']}",
        "name": "{prodes_dict['name']}",
        "uf": "{prodes_dict['uf']}",
        "biome": "{prodes_dict['biome']}",
        "source": "{prodes_dict['source']}",
        "census_pop_2022": {pop_repr},
        "proj_pop_2026": None,
        "official_total_area_ha": {prodes_dict['official_total_area_ha']},
        "official_forest_remnant_ha": {prodes_dict['official_forest_remnant_ha']},
        "historic_prodes_deforestation_ha": {{
{hist_lines}
        }},
        "mean_annual_deforestation_rate_pct": {prodes_dict['mean_annual_deforestation_rate_pct']},
        "primary_loss_drivers": {drivers_repr}
    }},'''
    target = "REAL_HISTORICAL_DATA: Dict[str, Dict[str, Any]] = {"
    if target not in content_prodes:
        raise RuntimeError("Não achei REAL_HISTORICAL_DATA em real_prodes_data.py")
    content_prodes = content_prodes.replace(target, target + "\n" + prodes_entry, 1)
    SRC_PY_PRODES.write_text(content_prodes, encoding="utf-8")
    print(f"✓ Série PRODES real gravada em {SRC_PY_PRODES.name}")


def add_to_typescript_catalog(city_dict: Dict[str, Any], prodes_hist: Optional[Dict[str, float]]) -> None:
    code = city_dict["ibge_code"]
    content = SRC_TS_MUNI.read_text(encoding="utf-8")
    if f'"{code}":' in content:
        print(f"• Já existia em {SRC_TS_MUNI.name} ({code})")
    else:
        entry = f'''  "{code}": {{
    ibge_code: "{city_dict['ibge_code']}",
    name: "{city_dict['name']}",
    uf: "{city_dict['uf']}",
    biome: "{city_dict['biome']}",
    total_area_ha: {city_dict['total_area_ha']},
    forest_remnant_ha: {city_dict['forest_remnant_ha']},
    center_lat: {city_dict['center_lat']},
    center_lon: {city_dict['center_lon']},
    bbox: {json.dumps(city_dict['bbox'])},
    base_annual_loss_rate: {city_dict['base_annual_loss_rate']},
    road_Corridors: {json.dumps(city_dict['road_Corridors'], ensure_ascii=False)},
    locality_zones: {json.dumps(city_dict['locality_zones'], ensure_ascii=False)}
  }},
'''
        target = "export const MUNICIPALITIES_CATALOG: Record<string, MunicipalityData> = {"
        if target not in content:
            raise RuntimeError("Não achei MUNICIPALITIES_CATALOG em municipality.ts")
        content = content.replace(target, target + "\n" + entry, 1)
        SRC_TS_MUNI.write_text(content, encoding="utf-8")
        print(f"✓ Catálogo TypeScript: {SRC_TS_MUNI.name}")

    if prodes_hist and SRC_TS_ML.exists():
        ml = SRC_TS_ML.read_text(encoding="utf-8")
        if f"'{code}'" not in ml and f'"{code}"' not in ml:
            # insere no REAL_PRODES
            marker = "const REAL_PRODES: Record<string, Record<string, number>> = {"
            if marker in ml:
                hist_js = json.dumps(prodes_hist, ensure_ascii=False)
                ml = ml.replace(marker, marker + f"\n  '{code}': {hist_js},", 1)
                SRC_TS_ML.write_text(ml, encoding="utf-8")
                print(f"✓ REAL_PRODES atualizado em mlEngine.ts")


def run_add_city(
    name_uf: str,
    prodes_json_path: Optional[str] = None,
    forest_remnant_ha: Optional[float] = None,
) -> str:
    city_name, uf = parse_name_uf(name_uf)
    print(f"→ Buscando IBGE: {city_name}/{uf} ...")
    info = fetch_ibge_city_info(city_name, uf)
    ibge_code = str(info["id"])
    official_name = info["nome"]
    # UF oficial
    micro = info.get("microrregiao") or {}
    meso = micro.get("mesorregiao") or {}
    official_uf = ((meso.get("UF") or {}).get("sigla")
        or (((info.get("regiao-imediata") or {}).get("regiao-intermediaria") or {}).get("UF") or {}).get("sigla")
        or uf)
    print(f"✓ IBGE {ibge_code} — {official_name}/{official_uf}")

    print("→ Baixando malha municipal (GeoJSON)...")
    geojson = download_geojson_boundary(ibge_code)
    bbox, (center_lat, center_lon), area_ha = compute_bbox_center_area(geojson)
    print(f"✓ Área estimada do polígono: {area_ha:,.1f} ha | centroide ({center_lat}, {center_lon})")

    biome = infer_biome(official_uf)
    pop = try_fetch_population(ibge_code)
    if pop is not None:
        print(f"✓ População (estimativa IBGE): {pop:,}")
    else:
        print("• População IBGE indisponível nesta API — campo fica None (não inventado)")

    # Série PRODES/MapBiomas oficial (TerraBrasilis) se arquivo ausente
    prodes_hist = None
    if prodes_json_path:
        pjson = Path(prodes_json_path)
        needs_fetch = not pjson.exists()
        if pjson.exists():
            try:
                raw = json.loads(pjson.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and (raw.get("_instrucoes") is not None or all(v is None for k,v in raw.items() if not str(k).startswith("_"))):
                    needs_fetch = True
            except Exception:
                needs_fetch = True
        if needs_fetch:
            print(f"→ Gerando série oficial (TerraBrasilis/INPE PRODES) → {pjson}...")
            try:
                from supressao_vegetacao.sources.mapbiomas_fetch import fetch_and_write_prodes_json
            except ImportError:
                from pathlib import Path as _P
                import sys as _sys
                _sys.path.insert(0, str(ROOT_DIR / "src"))
                from supressao_vegetacao.sources.mapbiomas_fetch import fetch_and_write_prodes_json
            # boundary must exist (saved below) — save boundary first if needed
            # geojson já baixado em download_geojson_boundary; garantir gravação
            bpath = BOUNDARIES_DIR / f"{ibge_code}.geojson"
            if not bpath.exists():
                bpath.parent.mkdir(parents=True, exist_ok=True)
                bpath.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
            prodes_hist = fetch_and_write_prodes_json(
                ibge_code, pjson, city_label=f"{official_name}-{official_uf}"
            )
            print(f"✓ Série oficial gravada em {pjson} ({len(prodes_hist)} anos)")
        else:
            prodes_hist = load_prodes_json(prodes_json_path, city_label=name_uf)
    else:
        # auto: data/prodes_{slug}.json
        slug = re.sub(r"[^a-z0-9]+", "_", f"{official_name}_{official_uf}".lower()).strip("_")
        auto_path = ROOT_DIR / "data" / f"prodes_{slug}.json"
        print(f"→ --prodes-json não informado — gerando série oficial em {auto_path.name}...")
        try:
            from supressao_vegetacao.sources.mapbiomas_fetch import fetch_and_write_prodes_json
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(ROOT_DIR / "src"))
            from supressao_vegetacao.sources.mapbiomas_fetch import fetch_and_write_prodes_json
        bpath = BOUNDARIES_DIR / f"{ibge_code}.geojson"
        if not bpath.exists():
            bpath.parent.mkdir(parents=True, exist_ok=True)
            bpath.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
        prodes_hist = fetch_and_write_prodes_json(
            ibge_code, auto_path, city_label=f"{official_name}-{official_uf}"
        )
        print(f"✓ Série oficial gravada em {auto_path} ({len(prodes_hist)} anos)")
    if forest_remnant_ha is not None and forest_remnant_ha < 0:
        raise ValueError("--forest-remnant-ha deve ser >= 0")

    if prodes_hist:
        mean_annual = sum(prodes_hist.values()) / len(prodes_hist)
        if forest_remnant_ha is None:
            print("→ --forest-remnant-ha ausente — estimando remanescente via PRODES residual (TerraBrasilis)...")
            try:
                from supressao_vegetacao.sources.mapbiomas_fetch import estimate_forest_remnant_ha
            except ImportError:
                import sys as _sys
                _sys.path.insert(0, str(ROOT_DIR / "src"))
                from supressao_vegetacao.sources.mapbiomas_fetch import estimate_forest_remnant_ha
            est = estimate_forest_remnant_ha(ibge_code)
            if est is not None and est > 0:
                remnant = float(est)
                base_rate = round(mean_annual / max(remnant, 1.0), 6)
                print(f"✓ Remanescente estimado (PRODES residual): {remnant:,.1f} ha")
            else:
                print(
                    "⚠ Não foi possível estimar remanescente. "
                    "Informe --forest-remnant-ha <ha> (MapBiomas) para projeção ML completa."
                )
                remnant = 0.0
                base_rate = 0.0
        else:
            remnant = float(forest_remnant_ha)
            base_rate = round(mean_annual / max(remnant, 1.0), 6)
        prodes_dict: Optional[Dict[str, Any]] = {
            "ibge_code": ibge_code,
            "name": official_name,
            "uf": official_uf,
            "biome": biome,
            "source": "Informado pelo usuário via --prodes-json (não gerado pelo script)",
            "census_pop_2022": pop,
            "official_total_area_ha": area_ha,
            "official_forest_remnant_ha": remnant,
            "historic_prodes_deforestation_ha": prodes_hist,
            "mean_annual_deforestation_rate_pct": round(base_rate * 100, 4),
            "primary_loss_drivers": [],
        }
    else:
        # Auto MapBiomas Coleção 11 (índice local)
        prodes_dict = None
        remnant = float(forest_remnant_ha) if forest_remnant_ha is not None else 0.0
        base_rate = 0.0
        try:
            import sys as _sys
            if str(ROOT_DIR / "src") not in _sys.path:
                _sys.path.insert(0, str(ROOT_DIR / "src"))
            from supressao_vegetacao.sources.mapbiomas_catalog import mapbiomas_series_for_prodes
            mb = mapbiomas_series_for_prodes(ibge_code)
            if mb:
                if forest_remnant_ha is None:
                    remnant = float(mb["forest_remnant_ha"])
                prodes_hist = mb["historic_prodes_deforestation_ha"]
                mean_annual = sum(prodes_hist.values()) / max(len(prodes_hist), 1)
                base_rate = mean_annual / max(remnant, 1.0) if remnant else 0.0
                if not biome or biome == "Não informado":
                    biome = mb.get("biome") or biome
                prodes_dict = {
                    "ibge_code": ibge_code,
                    "name": official_name,
                    "uf": official_uf,
                    "biome": biome,
                    "source": mb.get("source") or "MapBiomas Coleção 11",
                    "source_url": mb.get("source_url"),
                    "census_pop_2022": pop,
                    "official_total_area_ha": area_ha,
                    "official_forest_remnant_ha": remnant,
                    "historic_prodes_deforestation_ha": prodes_hist,
                    "mean_annual_deforestation_rate_pct": round(base_rate * 100, 4),
                    "primary_loss_drivers": [],
                }
                print(
                    f"✓ MapBiomas Coleção 11: remanescente={remnant:,.1f} ha | "
                    f"série perda {min(prodes_hist)}–{max(prodes_hist)} | "
                    f"taxa≈{base_rate*100:.3f}%/ano"
                )
            else:
                print(
                    "⚠ Sem --prodes-json e sem MapBiomas para este IBGE. "
                    "Município entra só com geometria IBGE."
                )
        except Exception as _mb_exc:
            print(f"⚠ MapBiomas indisponível ({_mb_exc}). Município só com geometria IBGE.")

    city_dict = {
        "ibge_code": ibge_code,
        "name": official_name,
        "uf": official_uf,
        "biome": biome,
        "total_area_ha": area_ha,
        "forest_remnant_ha": remnant,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "bbox": bbox,
        "base_annual_loss_rate": base_rate,
        "road_Corridors": default_road_corridors(bbox, (center_lat, center_lon)),
        "locality_zones": [
            f"Centro / Núcleo Urbano de {official_name}",
            f"Distrito Norte de {official_name}",
            f"Distrito Sul ({official_name})",
            "Setor Rural Leste",
            "Setor Rural Oeste",
            "Eixo Rodoviário Principal",
            "Bacia Hidrográfica Local",
            "Zona de Amortecimento Florestal",
        ],
    }

    add_to_python_catalogs(city_dict, prodes_dict)
    add_to_typescript_catalog(city_dict, prodes_hist)

    print("\n" + "=" * 68)
    print(f"Município {official_name}/{official_uf} ({ibge_code}) registrado.")
    print(f"• GeoJSON: data/boundaries/{ibge_code}.geojson")
    print(f"• Área polígono: {area_ha:,.1f} ha | Remanescente informado: {remnant:,.1f} ha")
    if prodes_hist:
        print(f"• PRODES: {len(prodes_hist)} anos (fonte: arquivo do usuário)")
        print(f"  python3 scripts/run_inference.py --ibge-code {ibge_code} --horizons 3 5 8 10 20")
    else:
        print("• PRODES: ausente — para habilitar ML, rode de novo com:")
        print(f'  python3 scripts/add_city.py {official_name.replace(" ", "")}-{official_uf} \\')
        print('    --prodes-json caminho/serie.json --forest-remnant-ha <ha>')
    print("=" * 68)

    # Treino de modelo espacial específico da cidade
    try:
        import sys as _sys
        if str(ROOT_DIR / "src") not in _sys.path:
            _sys.path.insert(0, str(ROOT_DIR / "src"))
        from supressao_vegetacao.spatial.train_city_model import train_city_spatial_model
        rate = float(base_rate) if base_rate else 0.005
        try:
            tr = train_city_spatial_model(ibge_code, mean_loss_rate=rate)
            print(
                f"✓ Modelo espacial real da cidade: {tr['model_dir']} "
                f"| ROC-AUC teste={tr.get('metrics_test',{}).get('roc_auc')} "
                f"| features={tr.get('feature_names')}"
            )
        except FileNotFoundError as exc:
            print(f"! Modelo espacial não treinado: {exc}")
            print("  Município cadastrado sem modelo; exporte primeiro o painel GEE/MapBiomas real.")
    except Exception as _tr_exc:
        print(f"⚠ Treino espacial por cidade falhou: {_tr_exc}")

    return ibge_code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adiciona município (NOME-UF) com dados reais do IBGE. Não inventa PRODES.",
        epilog="Exemplos:\n  python3 scripts/add_city.py SEARA-SC\n  python3 scripts/add_city.py --SEARA-SC\n  python3 scripts/add_city.py --city Seara-SC --prodes-json serie.json --forest-remnant-ha 12000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "city_positional",
        nargs="?",
        default=None,
        help="NOME-UF (ex: SEARA-SC)",
    )
    p.add_argument(
        "--city",
        dest="city_opt",
        default=None,
        help="NOME-UF alternativo (ex: --city SEARA-SC)",
    )
    p.add_argument(
        "--prodes-json",
        default=None,
        help="JSON {\"2018\": ha, ...} com série real (obrigatório para ML)",
    )
    p.add_argument(
        "--forest-remnant-ha",
        type=float,
        default=None,
        help="Remanescente florestal em ha (MapBiomas/PRODES) — não inventado pelo script",
    )
    return p


def resolve_city_arg(argv: List[str]) -> Tuple[argparse.Namespace, str]:
    """
    Aceita flags no estilo --SEARA-SC (nome da cidade como flag booleana).
    """
    parser = build_parser()
    known_prefixes = {"--city", "--prodes-json", "--forest-remnant-ha", "-h", "--help"}
    passthrough: List[str] = []
    city_from_flag: Optional[str] = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and a not in known_prefixes and not a.startswith("--prodes") and not a.startswith("--forest"):
            # --SEARA-SC → city
            if re.match(r"^--[A-Za-zÀ-ú0-9 ].+-[A-Za-z]{2}$", a) or re.match(r"^--[A-Za-z].+-[A-Za-z]{2}$", a):
                city_from_flag = a[2:]
                i += 1
                continue
        passthrough.append(a)
        i += 1

    args = parser.parse_args(passthrough)
    city = args.city_opt or args.city_positional or city_from_flag
    if not city:
        parser.error("Informe o município: SEARA-SC  ou  --SEARA-SC  ou  --city SEARA-SC")
    return args, city


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args, city = resolve_city_arg(argv)
        run_add_city(
            name_uf=city,
            prodes_json_path=args.prodes_json,
            forest_remnant_ha=args.forest_remnant_ha,
        )
        return 0
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
