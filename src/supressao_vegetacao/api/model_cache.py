"""
Cache em memória de painéis CSV, modelos espaciais, features e análise completa.

Uso típico na subida da API:
  FORESTGUARD_PRELOAD_IBGE=4210100
  ou: python scripts/run_api.py --preload-ibge 4210100

No startup carrega:
  1) todos os panel_*.csv do município (Panel em RAM)
  2) modelo XGBoost
  3) matriz de features pronta para /v1/predict
  4) análise completa horizontes 1–20 (pressure=1.0) para /v1/analyze

Evita releitura dos CSVs e reexecução pesada a cada POST.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("forestguard.api.cache")

_lock = threading.RLock()

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class CachedModel:
    ibge_code: str
    model_path: Path
    model: Any
    feature_names: List[str]
    meta: Dict[str, Any]
    loaded_at: float = field(default_factory=time.time)


@dataclass
class CachedPanel:
    """Painel completo (todos os panel_*.csv) em memória."""
    ibge_code: str
    panel_dir: Path
    panel: Any  # Panel
    n_rows: int
    n_years: int
    years: List[int]
    csv_files: List[str]
    loaded_at: float = field(default_factory=time.time)


@dataclass
class CachedFeatures:
    ibge_code: str
    horizon_years: int
    panel_year: int
    prepared: Dict[str, Any]
    loaded_at: float = field(default_factory=time.time)


@dataclass
class CachedAnalysis:
    """Resultado completo de run_city_deforestation_analysis em RAM."""
    ibge_code: str
    pressure: float
    horizons: Tuple[int, ...]
    result: Dict[str, Any]
    loaded_at: float = field(default_factory=time.time)


_models: Dict[str, CachedModel] = {}
_panels: Dict[str, CachedPanel] = {}
_features: Dict[Tuple[str, int, int], CachedFeatures] = {}
# chave: (ibge, pressure_rounded_3, horizons_tuple_sorted)
_analyses: Dict[Tuple[str, float, Tuple[int, ...]], CachedAnalysis] = {}


def get_cached_model(ibge_code: str) -> Optional[CachedModel]:
    code = str(ibge_code).strip()
    with _lock:
        return _models.get(code)


def get_cached_panel(ibge_code: str) -> Optional[CachedPanel]:
    code = str(ibge_code).strip()
    with _lock:
        return _panels.get(code)


def put_model(
    ibge_code: str,
    model_path: Path,
    model: Any,
    feature_names: List[str],
    meta: Dict[str, Any],
) -> CachedModel:
    code = str(ibge_code).strip()
    entry = CachedModel(
        ibge_code=code,
        model_path=Path(model_path),
        model=model,
        feature_names=list(feature_names),
        meta=dict(meta),
    )
    with _lock:
        _models[code] = entry
    logger.info(
        "Modelo em cache: IBGE=%s path=%s features=%d target=%s",
        code,
        model_path,
        len(feature_names),
        meta.get("target_col"),
    )
    return entry


def get_or_load_model(ibge_code: str) -> CachedModel:
    """Retorna modelo do cache ou carrega do disco e armazena."""
    code = str(ibge_code).strip()
    hit = get_cached_model(code)
    if hit is not None:
        return hit

    from supressao_vegetacao.spatial.infer import find_spatial_model, load_spatial_model

    model_path = find_spatial_model(ibge_code=code)
    if model_path is None:
        raise FileNotFoundError(f"Não há modelo espacial treinado para o IBGE {code}.")
    model, feature_names, meta = load_spatial_model(model_path)
    if str(meta.get("ibge_code") or "").strip() not in ("", code):
        raise ValueError(
            f"Manifesto do modelo ({meta.get('ibge_code')}) não corresponde ao IBGE {code}."
        )
    return put_model(code, model_path, model, feature_names, meta)


def _panel_dir_mtime(panel_dir: Path) -> float:
    try:
        mt = 0.0
        for fp in Path(panel_dir).glob("panel_*.csv"):
            mt = max(mt, fp.stat().st_mtime)
        return mt
    except OSError:
        return 0.0


def load_panel_csvs(ibge_code: str, force: bool = False) -> CachedPanel:
    """
    Lê todos os panel_*.csv do município e guarda o Panel em memória.
    É o passo principal pedido para o startup do FastAPI.
    """
    from supressao_vegetacao.spatial.panel import load_panel_csv

    code = str(ibge_code).strip()
    panel_dir = ROOT / "data" / "spatial" / "panels" / code
    if not panel_dir.is_dir():
        raise FileNotFoundError(f"Diretório de painel inexistente: {panel_dir}")

    csv_files = sorted(p.name for p in panel_dir.glob("panel_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum panel_*.csv em {panel_dir}")

    if not force:
        hit = get_cached_panel(code)
        if hit is not None and _panel_dir_mtime(panel_dir) <= hit.loaded_at:
            logger.info(
                "Painel CSV já em memória: IBGE=%s rows=%d years=%d files=%d",
                code,
                hit.n_rows,
                hit.n_years,
                len(hit.csv_files),
            )
            return hit

    t0 = time.time()
    logger.info(
        "Carregando %d CSV(s) do painel IBGE=%s a partir de %s …",
        len(csv_files),
        code,
        panel_dir,
    )
    try:
        panel = load_panel_csv(
            panel_dir,
            target_col="y_loss_1y",
            require_forest_t0=True,
            augment_osm_features=False,
        )
    except (ValueError, RuntimeError):
        panel = load_panel_csv(
            panel_dir,
            target_col="y_loss_3y",
            require_forest_t0=True,
            augment_osm_features=False,
        )

    years = sorted({int(y) for y in panel.years.tolist()})
    entry = CachedPanel(
        ibge_code=code,
        panel_dir=panel_dir,
        panel=panel,
        n_rows=int(len(panel.y)),
        n_years=len(years),
        years=years,
        csv_files=csv_files,
        loaded_at=time.time(),
    )
    with _lock:
        _panels[code] = entry
        # painel mudou → invalida features derivadas
        for k in [k for k in list(_features.keys()) if k[0] == code]:
            _features.pop(k, None)

    logger.info(
        "CSVs em memória: IBGE=%s rows=%d years=%s..%s files=%d (%.2fs)",
        code,
        entry.n_rows,
        years[0] if years else None,
        years[-1] if years else None,
        len(csv_files),
        time.time() - t0,
    )
    return entry


def get_or_prepare_features(
    ibge_code: str,
    feature_names: List[str],
    horizon_years: int,
    panel_dir: Path,
    target_year: Optional[int] = None,
    require_forest_t0: bool = True,
) -> Dict[str, Any]:
    """
    Prepara (ou reutiliza) a matriz X + meta para inferência.
    Usa o Panel já carregado dos CSVs quando disponível.
    """
    from supressao_vegetacao.spatial.panel import prepare_inference_features

    code = str(ibge_code).strip()
    H = int(horizon_years)
    cache_key_year = int(target_year) if target_year is not None else 0
    key = (code, H, cache_key_year)

    with _lock:
        hit = _features.get(key)
    if hit is not None:
        panel_hit = get_cached_panel(code)
        mtime = _panel_dir_mtime(panel_dir)
        if panel_hit is not None and mtime <= panel_hit.loaded_at and mtime <= hit.loaded_at:
            return hit.prepared
        if panel_hit is None and mtime <= hit.loaded_at:
            return hit.prepared

    # Garante CSVs em memória
    try:
        cached_panel = load_panel_csvs(code)
        panel_obj = cached_panel.panel
    except Exception:
        panel_obj = None
        cached_panel = None

    t0 = time.time()
    prepared = prepare_inference_features(
        panel_dir,
        feature_names=feature_names,
        target_year=target_year,
        horizon_years=H,
        require_forest_t0=require_forest_t0,
        panel=panel_obj,
    )
    elapsed = time.time() - t0
    panel_year = int(prepared.get("panel_year") or 0)
    entry = CachedFeatures(
        ibge_code=code,
        horizon_years=H,
        panel_year=panel_year,
        prepared=prepared,
        loaded_at=time.time(),
    )
    with _lock:
        _features[key] = entry
        if cache_key_year == 0 and panel_year:
            _features[(code, H, panel_year)] = entry
    logger.info(
        "Features em cache: IBGE=%s horizon=%d panel_year=%d cells=%d (%.2fs, csv_ram=%s)",
        code,
        H,
        panel_year,
        int(prepared.get("X").shape[0]) if prepared.get("X") is not None else 0,
        elapsed,
        cached_panel is not None,
    )
    return prepared


def _analysis_key(
    ibge_code: str, pressure: float, horizons: List[int]
) -> Tuple[str, float, Tuple[int, ...]]:
    code = str(ibge_code).strip()
    hs = tuple(sorted({int(h) for h in horizons}))
    return (code, round(float(pressure), 3), hs)


def get_cached_analysis(
    ibge_code: str, pressure: float, horizons: List[int]
) -> Optional[Dict[str, Any]]:
    """Retorna cópia do resultado em cache se a chave bater exatamente."""
    key = _analysis_key(ibge_code, pressure, horizons)
    with _lock:
        entry = _analyses.get(key)
        if entry is None:
            return None
        return copy.deepcopy(entry.result)


def put_analysis(
    ibge_code: str, pressure: float, horizons: List[int], result: Dict[str, Any]
) -> None:
    key = _analysis_key(ibge_code, pressure, horizons)
    entry = CachedAnalysis(
        ibge_code=key[0],
        pressure=key[1],
        horizons=key[2],
        result=copy.deepcopy(result),
    )
    with _lock:
        _analyses[key] = entry
    logger.info(
        "Análise em cache: IBGE=%s pressure=%s horizons=%s cells=%s",
        key[0],
        key[1],
        list(key[2]),
        len((result.get("cells") or [])),
    )


def get_or_run_analysis(
    ibge_code: str,
    horizons: List[int],
    pressure: float = 1.0,
) -> Dict[str, Any]:
    """
    Devolve análise do cache ou executa e armazena.

    Se existir cache com superconjunto de horizontes (ex.: 1–20) e a mesma
    pressure, filtra o resultado em vez de reexecutar.
    """
    code = str(ibge_code).strip()
    hs = sorted({int(h) for h in horizons})
    p = round(float(pressure), 3)

    exact = get_cached_analysis(code, p, hs)
    if exact is not None:
        logger.info("Cache HIT análise IBGE=%s horizons=%s", code, hs)
        return exact

    # Tentar superconjunto já em memória (ex.: preload 1..20)
    with _lock:
        candidates = [
            e
            for e in _analyses.values()
            if e.ibge_code == code and abs(e.pressure - p) < 1e-9
            and set(hs).issubset(set(e.horizons))
        ]
    if candidates:
        # preferir o menor superconjunto
        candidates.sort(key=lambda e: len(e.horizons))
        full = copy.deepcopy(candidates[0].result)
        wanted = {f"{h}y" for h in hs}
        summaries = full.get("horizon_summaries") or {}
        full["horizon_summaries"] = {k: v for k, v in summaries.items() if k in wanted}
        # células: manter scenarios só dos horizontes pedidos
        for cell in full.get("cells") or []:
            sc = cell.get("scenarios") or {}
            cell["scenarios"] = {k: v for k, v in sc.items() if k in wanted}
        logger.info(
            "Cache HIT (subset) análise IBGE=%s requested=%s from=%s",
            code,
            hs,
            list(candidates[0].horizons),
        )
        return full

    from supressao_vegetacao.inference.predict import run_city_deforestation_analysis

    t0 = time.time()
    result = run_city_deforestation_analysis(
        code, horizons=hs, custom_pressure_multiplier=p
    )
    put_analysis(code, p, hs, result)
    logger.info(
        "Cache MISS análise IBGE=%s horizons=%s (%.2fs)",
        code,
        hs,
        time.time() - t0,
    )
    return result


def preload_ibge(
    ibge_code: str,
    also_features: bool = True,
    also_analysis: bool = True,
    analysis_horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Startup: carrega CSVs do painel + modelo + features + análise 1–20.
    """
    code = str(ibge_code).strip()
    t0 = time.time()
    logger.info("Preload iniciado para IBGE %s (CSVs + modelo + análise) …", code)

    panel_entry = load_panel_csvs(code, force=True)
    model_entry = get_or_load_model(code)

    result: Dict[str, Any] = {
        "ibge_code": code,
        "csv_files": panel_entry.csv_files,
        "csv_rows": panel_entry.n_rows,
        "csv_years": panel_entry.years,
        "panel_dir": str(panel_entry.panel_dir),
        "model_path": str(model_entry.model_path),
        "target_col": model_entry.meta.get("target_col"),
        "n_features": len(model_entry.feature_names),
        "features_preloaded": False,
        "analysis_preloaded": False,
        "analysis_horizons": [],
        "cells_scored": 0,
        "panel_year": None,
        "elapsed_s": 0.0,
    }

    if also_features:
        horizon = int(model_entry.meta.get("horizon_years") or 1)
        prepared = get_or_prepare_features(
            code,
            feature_names=model_entry.feature_names,
            horizon_years=horizon,
            panel_dir=panel_entry.panel_dir,
            target_year=None,
        )
        result["features_preloaded"] = True
        result["panel_year"] = prepared.get("panel_year")
        result["cells_scored"] = (
            int(prepared["X"].shape[0]) if prepared.get("X") is not None else 0
        )

    if also_analysis:
        from supressao_vegetacao.settings import PRELOAD_HORIZONS

        hs = list(analysis_horizons) if analysis_horizons else list(PRELOAD_HORIZONS)
        try:
            analysis = get_or_run_analysis(code, horizons=hs, pressure=1.0)
            result["analysis_preloaded"] = True
            result["analysis_horizons"] = hs
            result["analysis_cells"] = len(analysis.get("cells") or [])
        except Exception as exc:
            logger.exception("Preload de análise falhou IBGE=%s: %s", code, exc)
            result["analysis_error"] = str(exc)

    result["elapsed_s"] = round(time.time() - t0, 3)
    logger.info(
        "Preload OK IBGE=%s csv_rows=%s model=%s features=%s analysis=%s (%.2fs)",
        code,
        result["csv_rows"],
        result["model_path"],
        result["features_preloaded"],
        result["analysis_preloaded"],
        result["elapsed_s"],
    )
    return result


def preload_many(codes: List[str], also_features: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in codes:
        code = str(raw).strip()
        if not code:
            continue
        try:
            out.append(preload_ibge(code, also_features=also_features))
        except Exception as exc:
            logger.exception("Falha no preload do IBGE %s", code)
            out.append({"ibge_code": code, "error": str(exc)})
    return out


def status() -> Dict[str, Any]:
    with _lock:
        models = [
            {
                "ibge_code": m.ibge_code,
                "model_path": str(m.model_path),
                "target_col": m.meta.get("target_col"),
                "n_features": len(m.feature_names),
                "loaded_at": m.loaded_at,
            }
            for m in _models.values()
        ]
        panels = [
            {
                "ibge_code": p.ibge_code,
                "panel_dir": str(p.panel_dir),
                "n_rows": p.n_rows,
                "n_years": p.n_years,
                "years": p.years,
                "csv_files": p.csv_files,
                "loaded_at": p.loaded_at,
            }
            for p in _panels.values()
        ]
        feats = [
            {
                "ibge_code": f.ibge_code,
                "horizon_years": f.horizon_years,
                "panel_year": f.panel_year,
                "n_cells": int(f.prepared["X"].shape[0]) if f.prepared.get("X") is not None else 0,
                "loaded_at": f.loaded_at,
            }
            for f in _features.values()
        ]
        analyses = [
            {
                "ibge_code": a.ibge_code,
                "pressure": a.pressure,
                "horizons": list(a.horizons),
                "n_cells": len((a.result.get("cells") or [])),
                "loaded_at": a.loaded_at,
            }
            for a in _analyses.values()
        ]
    return {
        "panels": panels,
        "models": models,
        "features": feats,
        "analyses": analyses,
    }


def clear_cache(ibge_code: Optional[str] = None) -> None:
    with _lock:
        if ibge_code is None:
            _models.clear()
            _panels.clear()
            _features.clear()
            _analyses.clear()
            return
        code = str(ibge_code).strip()
        _models.pop(code, None)
        _panels.pop(code, None)
        for k in [k for k in list(_features.keys()) if k[0] == code]:
            _features.pop(k, None)
        for k in [k for k in list(_analyses.keys()) if k[0] == code]:
            _analyses.pop(k, None)
