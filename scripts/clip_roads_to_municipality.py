#!/usr/bin/env python3
"""Recorta uma rede OSM para o limite de um município e grava data/roads/{IBGE}.geojson.

Uso: python scripts/clip_roads_to_municipality.py --ibge-code 4210100
Fonte padrão: data/roads/export.geojson; limite: data/boundaries/{IBGE}.geojson.
A fonte deve conter a rede OSM relevante; o recorte nunca amplia a rede para municípios vizinhos.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--ibge-code', required=True)
    p.add_argument('--source', default=str(ROOT / 'data' / 'roads' / 'export.geojson'))
    p.add_argument('--boundary', default=None)
    p.add_argument('--output', default=None)
    a = p.parse_args()
    code = ''.join(ch for ch in a.ibge_code if ch.isdigit())
    if len(code) != 7:
        p.error('--ibge-code precisa ter 7 dígitos')
    boundary_path = Path(a.boundary) if a.boundary else ROOT / 'data' / 'boundaries' / f'{code}.geojson'
    source_path = Path(a.source)
    output_path = Path(a.output) if a.output else ROOT / 'data' / 'roads' / f'{code}.geojson'
    if not boundary_path.is_file():
        print(f'Limite municipal não encontrado: {boundary_path}', file=sys.stderr); return 2
    if not source_path.is_file():
        print(f'GeoJSON de estradas não encontrado: {source_path}', file=sys.stderr); return 2
    boundary_data = json.loads(boundary_path.read_text(encoding='utf-8'))
    boundary = unary_union([shape(f['geometry']) for f in boundary_data.get('features', []) if f.get('geometry')])
    source = json.loads(source_path.read_text(encoding='utf-8'))
    output = []
    excluded = {'footway','path','steps','pedestrian','bridleway','cycleway'}
    for feature in source.get('features', []):
        if not feature.get('geometry'): continue
        props = feature.get('properties') or {}
        if not props.get('highway') or props.get('highway') in excluded: continue
        try:
            geom = shape(feature['geometry'])
            if not geom.intersects(boundary): continue
            clipped = geom.intersection(boundary)
            if clipped.is_empty or clipped.geom_type not in ('LineString','MultiLineString','GeometryCollection'): continue
            if clipped.geom_type == 'GeometryCollection':
                from shapely.geometry import MultiLineString
                lines = [g for g in clipped.geoms if g.geom_type in ('LineString','MultiLineString') and not g.is_empty]
                if not lines: continue
                parts = [line for g in lines for line in ([g] if g.geom_type == 'LineString' else list(g.geoms))]
                clipped = parts[0] if len(parts) == 1 else MultiLineString(parts)
            if clipped.length > 0:
                output.append({'type':'Feature','properties':props,'geometry':mapping(clipped)})
        except Exception: continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({'type':'FeatureCollection','name':f'roads_{code}_clipped_to_boundary','features':output}, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
    print(f'GeoJSON municipal: {output_path} | vias recortadas: {len(output)}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
