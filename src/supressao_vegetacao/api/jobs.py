"""
Orquestração de jobs: resolver cidade → (opcional) cadastrar → analisar → PDF → artefatos.
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = ROOT / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued | running | succeeded | failed
    created_at: str
    updated_at: str
    request: Dict[str, Any]
    ibge_code: Optional[str] = None
    municipality_name: Optional[str] = None
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_record(rec: JobRecord) -> None:
    path = _job_dir(rec.job_id) / "job.json"
    path.write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2), encoding="utf-8")


def load_job(job_id: str) -> Optional[JobRecord]:
    path = JOBS_DIR / job_id / "job.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobRecord(**data)


def list_jobs(limit: int = 50) -> List[JobRecord]:
    items: List[JobRecord] = []
    if not JOBS_DIR.exists():
        return items
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        rec = load_job(d.name)
        if rec:
            items.append(rec)
        if len(items) >= limit:
            break
    return items


def create_job(request: Dict[str, Any]) -> JobRecord:
    job_id = uuid.uuid4().hex[:12]
    rec = JobRecord(
        job_id=job_id,
        status="queued",
        created_at=_now(),
        updated_at=_now(),
        request=request,
    )
    _save_record(rec)
    return rec


def run_job(job_id: str) -> JobRecord:
    """Executa o pipeline completo de forma síncrona (adequado a cargas de segundos)."""
    rec = load_job(job_id)
    if not rec:
        raise ValueError(f"Job {job_id} não encontrado")

    rec.status = "running"
    rec.updated_at = _now()
    _save_record(rec)
    out = _job_dir(job_id)

    try:
        from ..sources.city_resolver import resolve_municipality_code  # v16 (parecer C3): sem dependência de GUI
        from ..inference.predict import run_city_deforestation_analysis
        from ..desktop.pdf_report import generate_technical_report_pdf
        from ..sources.municipality import get_municipality

        req = rec.request
        city = req.get("city") or req.get("municipality") or req.get("ibge_code")
        if not city:
            raise ValueError("Informe 'city' (NOME-UF) ou 'ibge_code'.")

        # Cadastro opcional se ainda não estiver no catálogo
        auto_add = bool(req.get("auto_add_city", True))
        prodes_json = req.get("prodes_json")  # dict ou caminho
        forest_remnant_ha = req.get("forest_remnant_ha")

        try:
            ibge = resolve_municipality_code(str(city))
        except ValueError:
            if not auto_add:
                raise
            # tenta adicionar via script add_city
            ibge = _add_city_via_api(
                str(city),
                prodes_json=prodes_json,
                forest_remnant_ha=forest_remnant_ha,
            )

        muni = get_municipality(ibge)
        if not muni:
            raise ValueError(f"Município {ibge} não encontrado após resolução/cadastro.")

        # Se veio série PRODES no request e ainda não há no catálogo, registra
        if isinstance(prodes_json, dict) and prodes_json:
            _ensure_prodes(ibge, muni, prodes_json, forest_remnant_ha)

        horizons = req.get("horizons") or [3, 5, 8, 10, 20]
        pressure = float(req.get("pressure", 1.0))
        grid_rows = req.get("grid_rows")
        grid_cols = req.get("grid_cols")
        if grid_rows is not None:
            grid_rows = int(grid_rows)
        if grid_cols is not None:
            grid_cols = int(grid_cols)

        # Treino espacial automático:
        # 1) path explícito no request
        # 2) painel GEE em data/spatial/<ibge>/ ou data/spatial/<slug>/
        # 3) se não houver painel, garante modelo geometric-only (road_dist_km)
        spatial_panel = req.get("spatial_panel_path") or _discover_spatial_panel(ibge, muni)
        spatial_train_info = None
        if spatial_panel:
            # Treino APENAS com painel real do próprio município.
            # v16 (parecer C1/C2): o modelo fica isolado no diretório do job —
            # nunca é publicado no diretório global data/spatial/models.
            spatial_train_info = _train_and_publish_spatial(
                panel_path=spatial_panel,
                model_dir=out / "spatial_model",
                target_col=req.get("spatial_target", "y_loss_3y"),
                out_metrics=out / "spatial_train_metrics.json",
                ibge=ibge,
            )
        else:
            # v16 (parecer C1): fallback sintético ELIMINADO. Sem painel GEE real,
            # a inferência usa a ponderação geométrica (distância a vias + densidade
            # PRODES por célula) já computada em predict.py — nenhum modelo com
            # rótulos artificiais é treinado ou carregado.
            spatial_train_info = {
                "auto_trained": False,
                "skipped": True,
                "reason": (
                    "Sem painel GEE real para este município. Inferência usa "
                    "ponderação geométrica (proximidade rodoviária + densidade "
                    "PRODES). Nenhum modelo sintético foi treinado."
                ),
            }

        result = run_city_deforestation_analysis(
            ibge_code=ibge,
            horizons=list(horizons),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            custom_pressure_multiplier=pressure,
        )

        # Artefato JSON completo
        json_path = out / "analysis.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # Resumo leve
        summary = {
            "ibge_code": result["municipality"]["ibge_code"],
            "name": result["municipality"]["name"],
            "uf": result["municipality"]["uf"],
            "forest_remnant_ha": result["municipality"]["forest_remnant_ha"],
            "horizons": {
                k: {
                    "target_year": v["target_year"],
                    "expected_suppressed_area_ha": v["expected_suppressed_area_ha"],
                    "percent_of_remnant_lost": v["percent_of_remnant_lost"],
                    "mean_probability": v["mean_probability"],
                    "critical_points_count": v["critical_points_count"],
                }
                for k, v in result.get("horizon_summaries", {}).items()
            },
            "spatial_model": result.get("model_metadata", {}).get("spatial_model"),
            "spatial_train": spatial_train_info,
            "model_version": result.get("model_metadata", {}).get("model_version"),
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # PDF
        pdf_path = out / f"Relatorio_{ibge}_{result['municipality']['name'].replace(' ', '_')}.pdf"
        generate_technical_report_pdf(result, str(pdf_path))

        # GeoJSON boundary copy
        boundary_src = ROOT / "data" / "boundaries" / f"{ibge}.geojson"
        if boundary_src.exists():
            import shutil

            shutil.copy2(boundary_src, out / f"{ibge}.geojson")

        artifacts = {
            "analysis_json": str(json_path.relative_to(ROOT)),
            "summary_json": str((out / "summary.json").relative_to(ROOT)),
            "pdf": str(pdf_path.relative_to(ROOT)),
        }
        if (out / f"{ibge}.geojson").exists():
            artifacts["boundary_geojson"] = str((out / f"{ibge}.geojson").relative_to(ROOT))
        if (out / "spatial_train_metrics.json").exists():
            artifacts["spatial_train_metrics"] = str(
                (out / "spatial_train_metrics.json").relative_to(ROOT)
            )

        rec.status = "succeeded"
        rec.ibge_code = ibge
        rec.municipality_name = result["municipality"]["name"]
        rec.artifacts = artifacts
        rec.result_summary = summary
        rec.updated_at = _now()
        _save_record(rec)
        return rec

    except Exception as exc:
        rec.status = "failed"
        rec.error = f"{exc}\n{traceback.format_exc()}"
        rec.updated_at = _now()
        _save_record(rec)
        return rec




def _discover_spatial_panel(ibge: str, muni: Dict[str, Any]) -> Optional[str]:
    """Procura painel GEE exportado em data/spatial/<ibge>/ ou data/spatial/<nome>/."""
    base = ROOT / "data" / "spatial"
    if not base.exists():
        return None
    candidates = [
        base / str(ibge),
        base / str(ibge) / "panel.csv",
        base / f"{ibge}.csv",
    ]
    slug = (muni.get("name") or "").lower().replace(" ", "_")
    uf = (muni.get("uf") or "").lower()
    if slug:
        candidates.extend([
            base / slug,
            base / f"{slug}_{uf}",
            base / slug / "panel.csv",
        ])
    for c in candidates:
        if c.is_file() and c.suffix.lower() == ".csv":
            return str(c)
        if c.is_dir():
            # qualquer csv no diretório
            csvs = sorted(c.glob("*.csv"))
            if csvs:
                return str(csvs[0])
    return None


def _train_and_publish_spatial(
    panel_path: str,
    model_dir: Path,
    target_col: str,
    out_metrics: Path,
    ibge: Optional[str] = None,
) -> Dict[str, Any]:
    from ..spatial.train_spatial import train_from_csv

    model_dir.mkdir(parents=True, exist_ok=True)
    train_result = train_from_csv(
        panel_path,
        target_col=target_col,
        model_dir=model_dir,
    )
    metrics = {
        "source_panel": panel_path,
        "ibge_code": ibge,
        "model_type": train_result.model_type,
        "metrics_test": train_result.metrics_test,
        "metrics_valid": train_result.metrics_valid,
        "n_train": train_result.n_train,
        "n_test": train_result.n_test,
        "feature_importance": train_result.feature_importance,
        "auto_trained": True,
        # v16 (parecer C2): escopo explícito — o modelo vale só para este município
        "scope": f"municipality:{ibge}" if ibge else "job_only",
    }
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # v16 (parecer C1/C2): REMOVIDA a cópia para ROOT/data/spatial/models.
    # O modelo treinado fica isolado no diretório do job e é usado apenas
    # pela análise deste job — não contamina outros municípios.
    return metrics


# v16 (parecer C1): a função _ensure_default_road_spatial_model foi REMOVIDA.
# Ela treinava um modelo com rótulos artificiais (seed=42) e o publicava no
# diretório global data/spatial/models, contaminando a inferência de todos os
# municípios. Sem painel GEE real, a inferência usa a ponderação geométrica
# documentada em predict.py — nenhum modelo sintético é treinado ou carregado.


def _add_city_via_api(
    city: str,
    prodes_json: Any = None,
    forest_remnant_ha: Optional[float] = None,
) -> str:
    """Chama a lógica de add_city (IBGE real) e devolve o código IBGE."""
    import importlib.util
    import sys

    script = ROOT / "scripts" / "add_city.py"
    spec = importlib.util.spec_from_file_location("add_city_mod", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Não foi possível carregar scripts/add_city.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["add_city_mod"] = mod
    spec.loader.exec_module(mod)

    prodes_path = None
    if isinstance(prodes_json, dict):
        tmp = _job_dir("tmp_prodes") / f"prodes_{uuid.uuid4().hex[:8]}.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(prodes_json), encoding="utf-8")
        prodes_path = str(tmp)
    elif isinstance(prodes_json, str):
        prodes_path = prodes_json

    ibge = mod.run_add_city(
        name_uf=city,
        prodes_json_path=prodes_path,
        forest_remnant_ha=float(forest_remnant_ha) if forest_remnant_ha is not None else None,
    )
    return ibge


def _ensure_prodes(
    ibge: str,
    muni: Dict[str, Any],
    prodes_hist: Dict[str, Any],
    forest_remnant_ha: Optional[float],
) -> None:
    """Garante série PRODES no catálogo quando enviada na API."""
    from ..sources.real_prodes_data import REAL_HISTORICAL_DATA

    if ibge in REAL_HISTORICAL_DATA and REAL_HISTORICAL_DATA[ibge].get(
        "historic_prodes_deforestation_ha"
    ):
        return
    # Reusa add_city para gravar
    _add_city_via_api(
        f"{muni['name']}-{muni['uf']}",
        prodes_json=prodes_hist,
        forest_remnant_ha=forest_remnant_ha or muni.get("forest_remnant_ha"),
    )
