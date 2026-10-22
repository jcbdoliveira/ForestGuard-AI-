"""
Agregação de dados para o dashboard da API (schema v2 + modelo anual).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]


def _panel_yearly_loss(panel_dir: Path) -> List[Dict[str, Any]]:
    """Contagens de perda observada (y_loss_1y) por year_t0 em células florestais."""
    files = sorted(panel_dir.glob("panel_*.csv"))
    by_year: Dict[int, Dict[str, float]] = {}
    for fp in files:
        with fp.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    year = int(float(row.get("year_t0") or 0))
                except (TypeError, ValueError):
                    continue
                if year <= 0:
                    continue
                slot = by_year.setdefault(
                    year,
                    {
                        "year": year,
                        "n_cells": 0,
                        "n_forest": 0,
                        "n_loss_1y": 0,
                        "loss_rate_among_forest": 0.0,
                    },
                )
                slot["n_cells"] += 1
                try:
                    forest = float(row.get("forest_t0") or 0)
                except (TypeError, ValueError):
                    forest = 0.0
                if forest >= 0.5:
                    slot["n_forest"] += 1
                    try:
                        y1 = float(row.get("y_loss_1y") or 0)
                    except (TypeError, ValueError):
                        y1 = 0.0
                    if y1 >= 0.5:
                        slot["n_loss_1y"] += 1
    out = []
    for year in sorted(by_year):
        s = by_year[year]
        nf = max(int(s["n_forest"]), 1)
        s["loss_rate_among_forest"] = round(float(s["n_loss_1y"]) / nf, 6)
        out.append(s)
    return out


def _prodes_series(ibge: str) -> Dict[str, Any]:
    cache = ROOT / "data" / "mapbiomas" / "cache" / f"{ibge}_prodes.json"
    alt = ROOT / "data" / f"prodes_{ibge}.json"
    path = cache if cache.is_file() else alt
    if not path.is_file():
        # tentar nomes legados (ex.: prodes_mafra_sc.json)
        for candidate in sorted((ROOT / "data").glob("prodes_*.json")):
            try:
                blob = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if ibge in blob[:500] or ibge in candidate.name:
                path = candidate
                break
    if not path.is_file():
        return {"available": False, "series_ha": [], "source": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "series_ha": [], "source": None}
    hist = data.get("historic_prodes_deforestation_ha") or {}
    series = [
        {"year": int(y), "deforestation_ha": float(v)}
        for y, v in sorted(hist.items(), key=lambda kv: int(kv[0]))
        if str(y).isdigit()
    ]
    return {
        "available": bool(series),
        "series_ha": series,
        "source": data.get("source") or str(path.relative_to(ROOT)),
        "n_polygons": data.get("n_polygons"),
        "note": data.get("note"),
    }


def _load_model_meta(ibge: str) -> Dict[str, Any]:
    candidates = [
        ROOT / "data" / "spatial" / "models" / f"{ibge}_annual" / "spatial_model_meta.json",
        ROOT / "data" / "spatial" / "models" / ibge / "spatial_model_meta.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        code = str(meta.get("ibge_code") or "").strip()
        parent = path.parent.name
        if parent in {ibge, f"{ibge}_annual"} or code == ibge or parent.replace("_annual", "") == ibge:
            meta["_meta_path"] = str(path.relative_to(ROOT))
            return meta
    return {}


def _xgboost_details(model_path: Optional[Path], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Hiperparâmetros e estrutura do booster, quando o arquivo existir."""
    out: Dict[str, Any] = {
        "model_type": meta.get("model_type") or "unknown",
        "algorithm": "XGBoost gradient boosting (binary:logistic)" if "xgb" in str(meta.get("model_type") or "").lower() or "XGBoost" in str(meta.get("model_type") or "") else meta.get("model_type"),
        "target_col": meta.get("target_col"),
        "horizon_years": meta.get("horizon_years"),
        "n_features": len(meta.get("feature_names") or []),
        "feature_names": meta.get("feature_names") or [],
        "n_train": meta.get("n_train"),
        "n_valid": meta.get("n_valid"),
        "n_test": meta.get("n_test"),
        "train_years": meta.get("train_years"),
        "test_year": meta.get("test_year"),
        "train_cutoff_year": meta.get("train_cutoff_year"),
        "positive_rate_test": meta.get("positive_rate_test"),
        "params": {},
        "booster_stats": {},
        "notes": [
            "Gradient boosting sequencial de árvores de decisão.",
            "Função objetivo típica: binary:logistic (probabilidade de perda anual).",
            "NaN em features opcionais é suportado nativamente pelo XGBoost (sem imputar 50 km).",
            "Validação temporal com embargo: treino não usa rótulos futuros.",
        ],
    }
    if model_path is None or not model_path.is_file():
        out["loaded"] = False
        return out
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        cfg = json.loads(booster.save_config())
        learner = cfg.get("learner") or {}
        gbtree = learner.get("gradient_booster") or {}
        tree_train = gbtree.get("tree_train_param") or gbtree.get("gbtree_train_param") or {}
        obj = learner.get("objective") or {}
        out["params"] = {
            "objective": obj.get("name") or obj.get("name") or str(obj)[:80],
            "eta_learning_rate": tree_train.get("eta") or (learner.get("gradient_booster") or {}).get("eta"),
            "max_depth": tree_train.get("max_depth"),
            "min_child_weight": tree_train.get("min_child_weight"),
            "subsample": tree_train.get("subsample"),
            "colsample_bytree": tree_train.get("colsample_bytree"),
            "gamma": tree_train.get("gamma"),
            "reg_lambda": tree_train.get("lambda") or tree_train.get("reg_lambda"),
            "reg_alpha": tree_train.get("alpha") or tree_train.get("reg_alpha"),
        }
        # limpar Nones
        out["params"] = {k: v for k, v in out["params"].items() if v is not None and str(v) != "None"}
        try:
            n_trees = len(booster.get_dump())
        except Exception:
            n_trees = None
        out["booster_stats"] = {
            "num_boosted_rounds": n_trees,
            "feature_names_in_booster": list(booster.feature_names) if booster.feature_names else None,
        }
        out["loaded"] = True
    except Exception as exc:
        out["loaded"] = False
        out["load_error"] = str(exc)
    return out



def _recompute_cost_metrics(
    ibge: str,
    meta: Dict[str, Any],
    model_path: Optional[Path],
    cost_fp: float,
    cost_fn: float,
) -> Optional[Dict[str, Any]]:
    """Recalcula métricas de limiar/custo no holdout de teste com custos customizados."""
    if model_path is None or not model_path.is_file():
        return None
    panel_dir = ROOT / "data" / "spatial" / "panels" / ibge
    if not panel_dir.is_dir():
        return None
    try:
        import numpy as np
        from supressao_vegetacao.spatial.panel import (
            load_panel_csv,
            augment_with_spatial_neighbors,
            temporal_split,
        )
        from supressao_vegetacao.spatial.infer import load_spatial_model, _positive_class_probabilities
        from supressao_vegetacao.spatial.train_spatial import evaluate_binary

        target = str(meta.get("target_col") or "y_loss_1y")
        horizon = int(meta.get("horizon_years") or (1 if target.endswith("1y") else 3))
        model, feature_names, _ = load_spatial_model(model_path)
        panel = load_panel_csv(panel_dir, target_col=target)
        panel = augment_with_spatial_neighbors(panel, horizon_years=horizon)

        # Garante ordem de colunas do manifesto do modelo
        name_to_idx = {n: i for i, n in enumerate(panel.feature_names)}
        missing = [n for n in feature_names if n not in name_to_idx]
        if missing:
            # colunas ausentes → NaN (XGBoost aceita)
            for n in missing:
                panel.feature_names.append(n)
                panel.X = np.column_stack([panel.X, np.full(len(panel.y), np.nan)])
                name_to_idx[n] = len(panel.feature_names) - 1
        order = [name_to_idx[n] for n in feature_names]
        panel.X = panel.X[:, order]
        panel.feature_names = list(feature_names)

        test_year = meta.get("test_year")
        split = temporal_split(
            panel,
            test_year=int(test_year) if test_year is not None else None,
            horizon_years=horizon,
        )
        X_te, y_te = split["test"]
        if len(y_te) == 0:
            return None
        proba = _positive_class_probabilities(model, X_te)
        return evaluate_binary(
            y_te, proba, threshold=0.5,
            cost_fp=cost_fp, cost_fn=cost_fn, include_curves=True,
        )
    except Exception:
        return None


def build_dashboard_payload(

    ibge: str,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Dict[str, Any]:
    ibge = str(ibge).strip()
    cost_fp = float(cost_fp)
    cost_fn = float(cost_fn)
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("cost_fp e cost_fn devem ser >= 0")

    panel_dir = ROOT / "data" / "spatial" / "panels" / ibge
    meta = _load_model_meta(ibge)
    model_rel = meta.get("model_path")
    model_path = None
    if model_rel:
        candidate = Path(model_rel)
        if not candidate.is_file():
            candidate = ROOT / model_rel
        if candidate.is_file():
            model_path = candidate
    if model_path is None:
        for cand in (
            ROOT / "data" / "spatial" / "models" / f"{ibge}_annual" / "spatial_model.json",
            ROOT / "data" / "spatial" / "models" / ibge / "spatial_model.json",
        ):
            if cand.is_file():
                model_path = cand
                break

    panel_series = _panel_yearly_loss(panel_dir) if panel_dir.is_dir() else []
    prodes = _prodes_series(ibge)
    importance = meta.get("feature_importance") or []
    if isinstance(importance, list) and importance and isinstance(importance[0], dict):
        importance_sorted = sorted(importance, key=lambda x: float(x.get("importance") or 0), reverse=True)
    else:
        importance_sorted = importance

    wf = meta.get("walk_forward") or {}
    if not wf:
        wf_path = (model_path.parent / "walk_forward_annual.json") if model_path else None
        if wf_path and wf_path.is_file():
            try:
                wf = json.loads(wf_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                wf = {}

    metrics_test = dict(meta.get("metrics_test") or {})
    metrics_valid = dict(meta.get("metrics_valid") or {})

    # Recalcula otimização de limiar/custo se o usuário passou custos (ou sempre, para refletir query)
    recomputed = _recompute_cost_metrics(ibge, meta, model_path, cost_fp=cost_fp, cost_fn=cost_fn)
    if recomputed:
        # preserva ranking metrics do manifesto se recompute falhar parcialmente; sobrescreve bloco de custo/PR/ECE
        for key in (
            "threshold_optimization", "threshold_best_f1", "threshold_best_cost",
            "expected_cost_at_best", "expected_cost_at_0_5",
            "ece", "calibration", "pr_curve",
            "precision", "recall", "f1", "specificity",
            "tn", "fp", "fn", "tp", "accuracy", "balanced_accuracy",
            "threshold",
        ):
            if key in recomputed:
                metrics_test[key] = recomputed[key]
        metrics_test["cost_fp"] = cost_fp
        metrics_test["cost_fn"] = cost_fn
    else:
        metrics_test.setdefault("cost_fp", cost_fp)
        metrics_test.setdefault("cost_fn", cost_fn)

    return {
        "ibge_code": ibge,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "panel_schema": meta.get("panel_schema") or ("v2" if panel_series else None),
        "project_version": meta.get("project_version") or "v25.0.1",
        "deforestation": {
            "panel_y_loss_1y_by_year": panel_series,
            "prodes_ha_by_year": prodes,
            "interpretation": (
                "panel_y_loss_1y_by_year: contagem de células florestais com perda observada até t0+1. "
                "prodes_ha_by_year: hectares oficiais PRODES (quando o cache existir)."
            ),
        },
        "metrics": {
            "test": metrics_test,
            "validation": metrics_valid,
            "walk_forward_summary": {
                "n_cutoffs": wf.get("n_cutoffs"),
                "median_roc_auc": wf.get("median_roc_auc"),
                "median_pr_auc": wf.get("median_pr_auc"),
                "horizon_years": wf.get("horizon_years"),
                "comparison_protocol": wf.get("comparison_protocol"),
            },
            "definitions": {
                "roc_auc": "Área sob a curva ROC (discriminação; 0.5 = aleatório).",
                "pr_auc": "Área sob Precision-Recall (útil com evento raro).",
                "brier": "Erro quadrático médio das probabilidades (menor = melhor).",
                "log_loss": "Entropia cruzada das probabilidades (menor = melhor).",
                "r2": "R² entre rótulo 0/1 e probabilidade (pode ser negativo se pior que prever a média).",
                "mae": "Erro absoluto médio |y − p|.",
                "rmse": "Raiz do erro quadrático médio entre y e p.",
                "accuracy": "Acurácia no limiar (default 0,5).",
                "balanced_accuracy": "Média de sensibilidade e especificidade.",
                "precision": "Precisão (VP / (VP+FP)) no limiar.",
                "recall": "Sensibilidade / recall no limiar.",
                "f1": "Média harmônica de precisão e recall.",
                "specificity": "Taxa de verdadeiros negativos.",
                "recall_at_top10pct": "Positivos capturados nos 10% maiores scores (ranking operacional).",
                "positive_rate": "Taxa de positivos no conjunto de avaliação.",
                "ece": "Expected Calibration Error — desvio médio |confiança − frequência| nos bins.",
                "threshold_best_f1": "Limiar que maximiza F1 no conjunto de avaliação.",
                "threshold_best_cost": "Limiar que minimiza cost_fp*FP + cost_fn*FN (default FN:FP=5:1).",
                "expected_cost_at_best": "Custo esperado no limiar ótimo por custo.",
                "expected_cost_at_0_5": "Custo esperado no limiar 0,5 (referência).",
            },
        },
        "feature_importance": importance_sorted,
        "ml_model": _xgboost_details(model_path, meta),
        "training_context": {
            "data_source": meta.get("data_source"),
            "training_note": meta.get("training_note"),
            "calibration_note": meta.get("calibration_note"),
            "metadata_file": meta.get("_meta_path"),
        },
        "warnings": [
            "Métricas descrevem holdout/walk-forward históricos, não desempenho futuro garantido.",
            "O dashboard não substitui fiscalização de campo.",            
        ],
    }


FEATURE_CATALOG: Dict[str, Dict[str, str]] = {
    "slope_deg": {
        "label": "Declividade (°)",
        "origin": "Modelo digital de elevação (SRTM/GEE no export do painel)",
        "meaning": "Inclinação do terreno na célula. Áreas muito íngremes ou planas podem ter padrões diferentes de uso e pressão.",
    },
    "elev_m": {
        "label": "Altitude (m)",
        "origin": "Modelo digital de elevação (SRTM/GEE no export do painel)",
        "meaning": "Elevação média da célula em metros. Ajuda a capturar diferenças de relevo e acessibilidade no município.",
    },
    "forest_neighbor_frac": {
        "label": "Fração florestal dos vizinhos",
        "origin": "Calculada no painel a partir da cobertura MapBiomas das células vizinhas em t0",
        "meaning": "Quanto da vizinhança imediata ainda era floresta no ano de referência. Baixa fração sugere efeito de borda e maior risco.",
    },
    "built_surface": {
        "label": "Superfície construída",
        "origin": "Camada de área construída no export GEE (quando disponível; senão NaN)",
        "meaning": "Intensidade de edificação na célula. Pode indicar urbanização ou infraestrutura próxima; ausência de dado não invalida a célula.",
    },
    "built_dist_km": {
        "label": "Distância à área construída (km)",
        "origin": "Distância à superfície construída no export GEE (quando disponível; senão NaN)",
        "meaning": "Proximidade a áreas edificadas. Valores próximos de zero sugerem maior pressão antrópica local.",
    },
    "years_stable_forest": {
        "label": "Anos de floresta estável",
        "origin": "Série temporal MapBiomas no export do painel (contagem de anos consecutivos com floresta)",
        "meaning": "Há quantos anos a célula permanece como floresta. Floresta estável há muito tempo costuma ter menor risco imediato de conversão.",
    },
    "forest_fraction_t0": {
        "label": "Fração de floresta na célula (t0)",
        "origin": "MapBiomas no ano de referência (export GEE do painel v2)",
        "meaning": "Proporção da célula coberta por floresta em t0. É o sinal mais forte no modelo atual de Mafra: quanto menos floresta residual, maior a associação com perda no ano seguinte.",
    },
    "valid_fraction_t0": {
        "label": "Fração válida do pixel (t0)",
        "origin": "Máscara de qualidade/cobertura do export GEE-MapBiomas",
        "meaning": "Quanto da célula tinha observação válida no mapa. Útil para não confiar demais em células com muita nuvem ou dado incompleto.",
    },
    "loss_neighbors_history_1y": {
        "label": "Perda recente nos vizinhos (1 ano)",
        "origin": "Derivada no treino/inferência: rótulos y_loss_1y de células vizinhas já observáveis (embargo temporal)",
        "meaning": "Quantos vizinhos distintos já tiveram perda florestal no horizonte de 1 ano, usando só informação disponível antes do ano previsto. Captura contágio espacial sem vazamento temporal.",
    },
    "loss_neighbors_history_3y": {
        "label": "Perda recente nos vizinhos (3 anos)",
        "origin": "Derivada no treino (legado y_loss_3y) com embargo temporal",
        "meaning": "Versão de 3 anos da vizinhança histórica de perda. Usada em modelos legados, não no default anual.",
    },
    "road_dist_km": {
        "label": "Distância a qualquer estrada (km)",
        "origin": "OpenStreetMap recortado ao município (data/roads/{IBGE}.geojson)",
        "meaning": "Proximidade à malha viária. Estradas facilitam acesso e estão associadas historicamente a frentes de abertura.",
    },
    "road_dist_major_km": {
        "label": "Distância a estrada principal (km)",
        "origin": "OpenStreetMap (classes principais) em data/roads/",
        "meaning": "Proximidade a vias de maior hierarquia (ex.: rodovias).",
    },
    "road_dist_secondary_km": {
        "label": "Distância a estrada secundária (km)",
        "origin": "OpenStreetMap em data/roads/",
        "meaning": "Proximidade a vias secundárias.",
    },
    "road_dist_local_km": {
        "label": "Distância a estrada local (km)",
        "origin": "OpenStreetMap em data/roads/",
        "meaning": "Proximidade a vias locais/municipais.",
    },
    "road_dist_track_km": {
        "label": "Distância a trilha/estrada de terra (km)",
        "origin": "OpenStreetMap em data/roads/",
        "meaning": "Proximidade a trilhas e caminhos não pavimentados, comuns em frentes de exploração.",
    },
}


def _feature_help(name: str) -> Dict[str, str]:
    key = str(name or "").strip()
    if key in FEATURE_CATALOG:
        return FEATURE_CATALOG[key]
    return {
        "label": key.replace("_", " "),
        "origin": "Feature do painel espacial / modelo (ver schema v2)",
        "meaning": "Variável usada pelo classificador anual; consulte PANEL_SCHEMA_V2.md para detalhes.",
    }


def render_dashboard_html(payload: Dict[str, Any]) -> str:
    """HTML autocontido com Chart.js (CDN), textos explicativos e catálogo de features."""
    ibge = payload.get("ibge_code", "")
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    feat_catalog_json = json.dumps(FEATURE_CATALOG, ensure_ascii=False).replace("</", "<\\/")

    metrics = (payload.get("metrics") or {}).get("test") or {}
    ml = payload.get("ml_model") or {}

    def _fmt(v: Any, digits: int = 4) -> str:
        if v is None or v == "":
            return "—"
        try:
            f = float(v)
            if abs(f) >= 100:
                return f"{f:.1f}"
            if abs(f) >= 1:
                return f"{f:.3f}".rstrip("0").rstrip(".")
            return f"{f:.{digits}f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(v)

    # Apenas métricas de ranking — as demais (R², ECE, Brier, log-loss, F1@0.5,
    # limiares e custos) confundem leigos e podem parecer "modelo ruim" em evento raro.
    cards = [
        (
            "ROC-AUC",
            _fmt(metrics.get("roc_auc"), 3),
            "Capacidade de ordenar células de maior para menor risco. "
            "0,5 = aleatório; quanto mais perto de 1, melhor o ranking.",
        ),
        (
            "PR-AUC",
            _fmt(metrics.get("pr_auc"), 3),
            "Qualidade do ranking quando a perda é rara (poucos positivos). "
            "Mais informativa que a ROC em desmate pontual.",
        ),
        (
            "Recall@top 10%",
            _fmt(metrics.get("recall_at_top10pct"), 3),
            "Dos eventos de perda reais no holdout, quantos caem nos 10% de células "
            "com maior escore. Métrica prática de priorização de campo.",
        ),
    ]
    import html as _html
    cards_html = "".join(
        f'<div class="card">'
        f'<div class="card-top">'
        f'<div class="label">{_html.escape(lab)}</div>'
        f'<button type="button" class="info-btn" data-title="{_html.escape(lab, quote=True)}" '
        f'data-hint="{_html.escape(hint, quote=True)}" '
        f'aria-label="Explicação de {_html.escape(lab, quote=True)}">i</button>'
        f'</div>'
        f'<div class="value">{_html.escape(str(val))}</div>'
        f"</div>"
        for lab, val, hint in cards
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ForestGuard Dashboard — {ibge}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f3faf3;
      --panel: #ffffff;
      --line: #c8e6c9;
      --text: #1b3d1f;
      --muted: #5a7260;
      --accent: #2e7d32;
      --accent-bright: #43a047;
      --good: #66bb6a;
      --warn: #ef6c00;
      --banner: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #66bb6a 100%);
      --leaf: #81c784;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "Segoe UI", Inter, system-ui, sans-serif;
      background:
        radial-gradient(ellipse at top right, rgba(129,199,132,0.25), transparent 50%),
        radial-gradient(ellipse at bottom left, rgba(46,125,50,0.12), transparent 45%),
        var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 1.1rem 1.5rem 1.25rem;
      background: var(--banner);
      color: #fff;
      display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; justify-content: space-between;
      box-shadow: 0 4px 18px rgba(27,94,32,0.28);
    }}
    header .brand {{
      display: flex; gap: 1rem; align-items: center;
    }}
    header .brand-logo {{
      width: 72px; height: auto; max-height: 88px;
      object-fit: contain; background: #fff; border-radius: 12px;
      padding: 4px 6px; border: 2px solid rgba(255,255,255,0.55);
      box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }}
    header h1 {{
      margin: 0; font-size: 1.5rem; letter-spacing: 0.01em;
      font-weight: 800; color: #fff;
      text-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }}
    header h1 span.fg-ai {{ color: #c8e6c9; }}
    header .tagline {{
      margin: 0.2rem 0 0; font-size: 0.78rem; font-weight: 600;
      letter-spacing: 0.07em; text-transform: uppercase; color: #e8f5e9;
    }}
    .sub {{ color: #e8f5e9; font-size: 0.88rem; line-height: 1.45; max-width: 48rem; opacity: 0.95; }}
    .sub strong {{ color: #fff; }}
    main {{ padding: 1.25rem 1.5rem 2.5rem; max-width: 1200px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem; }}
    .card {{
      background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
      padding: 0.9rem 1rem;
      box-shadow: 0 2px 8px rgba(46,125,50,0.06);
    }}
    .card .card-top {{
      display: flex; align-items: center; justify-content: space-between; gap: 0.4rem;
    }}
    .card .label {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .card .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 0.35rem; color: var(--accent); }}
    .info-btn {{
      width: 22px; height: 22px; border-radius: 50%; border: 1px solid #a5d6a7;
      background: #e8f5e9; color: #1b5e20; font-size: 0.72rem; font-weight: 800;
      font-family: Georgia, serif; font-style: italic; cursor: pointer;
      line-height: 1; padding: 0; flex-shrink: 0;
    }}
    .info-btn:hover {{ background: #2e7d32; color: #fff; border-color: #2e7d32; }}
    .modal-backdrop {{
      display: none; position: fixed; inset: 0; background: rgba(27,61,31,0.45);
      z-index: 1000; align-items: center; justify-content: center; padding: 1rem;
    }}
    .modal-backdrop.open {{ display: flex; }}
    .modal {{
      background: #fff; border-radius: 16px; max-width: 420px; width: 100%;
      border: 2px solid #81c784; box-shadow: 0 12px 40px rgba(27,94,32,0.25);
      padding: 1.15rem 1.25rem 1rem; position: relative;
    }}
    .modal h3 {{ margin: 0 0 0.55rem; color: var(--accent); font-size: 1.05rem; }}
    .modal p {{ margin: 0; color: var(--text); font-size: 0.92rem; line-height: 1.5; }}
    .modal .close {{
      position: absolute; top: 0.55rem; right: 0.65rem;
      border: none; background: transparent; font-size: 1.25rem; cursor: pointer;
      color: var(--muted); line-height: 1;
    }}
    .modal .close:hover {{ color: var(--accent); }}
    section {{
      background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
      padding: 1rem 1.1rem; margin-bottom: 1rem;
      box-shadow: 0 2px 8px rgba(46,125,50,0.05);
    }}
    section h2 {{ margin: 0 0 0.35rem; font-size: 1.05rem; color: var(--accent); }}
    .section-help {{ color: var(--muted); font-size: 0.82rem; line-height: 1.45; margin: 0 0 0.85rem; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
    canvas {{ width: 100% !important; max-height: 280px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.45rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    .feat-name {{ font-weight: 600; color: var(--text); }}
    .feat-code {{ font-family: ui-monospace, monospace; font-size: 0.75rem; color: #689f38; }}
    .feat-origin {{ font-size: 0.78rem; color: #7a927f; margin-top: 0.15rem; }}
    .feat-meaning {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; line-height: 1.35; }}
    .bar {{ height: 8px; background: #e8f5e9; border-radius: 99px; overflow: hidden; min-width: 64px; }}
    .bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, #2e7d32, #81c784); }}
    .warn {{ color: var(--warn); font-size: 0.85rem; line-height: 1.4; }}
    .tag {{
      display: inline-block; background: #e8f5e9; color: #1b5e20;
      padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.78rem;
      font-weight: 700; border: 1px solid #a5d6a7;
    }}
    header .tag {{ background: rgba(255,255,255,0.18); color: #fff; border-color: rgba(255,255,255,0.35); }}
    pre {{ background: #e8f5e9; border-radius: 8px; padding: 0.75rem; overflow: auto; font-size: 0.8rem; color: #1b3d1f; border: 1px solid var(--line); }}
    .legend-box {{
      background: #e8f5e9; border: 1px solid var(--line); border-radius: 12px;
      padding: 0.85rem 1rem; font-size: 0.85rem; color: var(--muted); line-height: 1.45; margin-bottom: 1rem;
    }}
    .legend-box strong {{ color: var(--accent); }}
    .brand-tip {{
      display: flex; gap: 0.9rem; align-items: center;
      background: #fff; border: 2px solid #81c784; border-radius: 16px;
      padding: 0.85rem 1rem; margin-bottom: 1.1rem;
      box-shadow: 0 3px 12px rgba(46,125,50,0.1);
    }}
    .brand-tip img {{
      width: 56px; height: auto; object-fit: contain; flex-shrink: 0;
      background: #fff; border-radius: 10px; padding: 2px;
      border: 1px solid #c8e6c9;
    }}
    .brand-tip .bubble {{
      background: #f1f8e9; border-radius: 12px; padding: 0.65rem 0.85rem;
      font-size: 0.9rem; color: var(--text); line-height: 1.4;
      border: 1px solid #c5e1a5;
    }}
    .brand-tip .bubble b {{ color: var(--accent); }}
    footer.dash-foot {{
      text-align: center; padding: 1rem; color: var(--muted); font-size: 0.78rem;
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <img class="brand-logo" src="data:image/png;base64,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" alt="ForestGuard AI"/>
      <div>
        <div class="tagline">Environmental AI System</div>
        <h1>Forest<span class="fg-ai">Guard</span> AI · Dashboard</h1>
        <div class="sub">
          Município IBGE <strong>{ibge}</strong> · painel schema v2 · alvo <strong>y_loss_1y</strong>
          (probabilidade de perda florestal no <em>próximo ano</em> por célula com floresta em t0).
        </div>
      </div>
    </div>
    <span class="tag">XGBoost · previsão anual</span>
  </header>
  <main>
    <div class="brand-tip">
      <img src="data:image/png;base64,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" alt="ForestGuard AI"/>
      <div class="bubble">
        <b>ForestGuard AI</b> — sistema de apoio à decisão para estimar risco relativo de
        supressão de vegetação no próximo ano. Use as métricas de ranking abaixo para
        priorizar onde olhar no território (não é auto de infração nem desmate consumado).
      </div>
    </div>
    <div class="legend-box">
      <strong>Como ler este painel.</strong>
      O modelo atribui a cada célula florestal um escore de risco entre 0 e 1.
      As três métricas acima medem a qualidade do <em>ranking</em> no holdout temporal
      (quão bem as células de maior risco concentram as perdas reais).
      Clique no botão <strong>i</strong> de cada card para a definição completa.
    </div>

    <div class="grid">{cards_html}</div>

    <div class="charts">
      <section>
        <h2>Perda observada no painel (células y_loss_1y)</h2>
        <p class="section-help">
          Contagem anual de células que eram floresta em t0 e perderam cobertura até t0+1
          (rótulo MapBiomas do painel GEE). A linha verde mostra o total de células florestais observadas.
          Serve para ver a prevalência histórica do evento que o modelo tenta prever.
        </p>
        <canvas id="chartPanel"></canvas>
      </section>
      <section>
        <h2>PRODES — hectares por ano</h2>
        <p class="section-help">
          Série oficial de desmatamento em hectares (cache PRODES/MapBiomas do município, quando disponível).
          É uma referência municipal agregada — não é o mesmo recorte espacial das células do painel,
          mas ajuda a contextualizar a magnitude da perda no território.
        </p>
        <canvas id="chartProdes"></canvas>
      </section>
    </div>

    <section>
      <h2>Importância das features</h2>
      <p class="section-help">
        Contribuição relativa de cada variável no XGBoost (gain normalizado no manifesto).
        Não implica causalidade: indica o que o modelo usou para separar células com e sem perda no treino.
        Features com importância 0 podem estar ausentes (NaN) neste município ou não ter sido úteis no holdout.
      </p>
      <table id="featTable">
        <thead>
          <tr>
            <th style="width:28%">Feature</th>
            <th style="width:10%">Importância</th>
            <th style="width:12%"></th>
            <th>Origem e interpretação</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>

    <div class="charts">
      <section>
        <h2>Curva Precision–Recall</h2>
        <p class="section-help">
          Cada ponto é um limiar de escore: no eixo X a fração dos desmates capturados (recall),
          no eixo Y a fração de alertas que eram corretos (precisão).
          Com eventos raros, esta curva é mais útil que a ROC para escolher o ponto de operação.
        </p>
        <canvas id="chartPR"></canvas>
      </section>
      <!-- seção de custo/limiar removida: gerava leitura confusa para o uso principal (ranking) -->
    </div>

    <section>
      <h2>Modelo — XGBoost / gradient boosting</h2>
      <p class="section-help">
        Classificador binário (<code>binary:logistic</code>) treinado só com informação conhecida em t0,
        com embargo temporal para não vazar o futuro. O alvo padrão é <code>y_loss_1y</code>.
      </p>
      <div class="sub" style="margin-bottom:0.75rem">{ml.get('algorithm') or ml.get('model_type') or '—'}</div>
      <div class="charts">
        <div>
          <strong>Contexto de treino</strong>
          <pre id="mlCtx"></pre>
        </div>
        <div>
          <strong>Hiperparâmetros / booster</strong>
          <pre id="mlParams"></pre>
        </div>
      </div>
      <p class="warn" id="warns"></p>
    </section>
  
    <div class="modal-backdrop" id="metricModal" role="dialog" aria-modal="true" aria-labelledby="metricModalTitle">
      <div class="modal">
        <button type="button" class="close" id="metricModalClose" aria-label="Fechar">&times;</button>
        <h3 id="metricModalTitle">Métrica</h3>
        <p id="metricModalBody"></p>
      </div>
    </div>

  </main>
  <script>
    const DATA = {data_json};
    const FEAT_CATALOG = {feat_catalog_json};

    function featureHelp(name) {{
      const key = String(name || '');
      if (FEAT_CATALOG[key]) return FEAT_CATALOG[key];
      return {{
        label: key.replace(/_/g, ' '),
        origin: 'Feature do painel espacial / modelo',
        meaning: 'Variável do classificador anual (ver schema v2).'
      }};
    }}

    const panel = (DATA.deforestation && DATA.deforestation.panel_y_loss_1y_by_year) || [];
    new Chart(document.getElementById('chartPanel'), {{
      type: 'bar',
      data: {{
        labels: panel.map(r => r.year),
        datasets: [
          {{ label: 'Células com perda 1y', data: panel.map(r => r.n_loss_1y), backgroundColor: '#ef6c00' }},
          {{ label: 'Células florestais', data: panel.map(r => r.n_forest), backgroundColor: '#22c55e55', type: 'line', borderColor: '#22c55e', tension: 0.2 }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#1b3d1f' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#5a7260' }}, grid: {{ color: '#c8e6c9' }} }},
          y: {{ ticks: {{ color: '#5a7260' }}, grid: {{ color: '#c8e6c9' }} }}
        }}
      }}
    }});

    const prodes = (DATA.deforestation && DATA.deforestation.prodes_ha_by_year && DATA.deforestation.prodes_ha_by_year.series_ha) || [];
    new Chart(document.getElementById('chartProdes'), {{
      type: 'line',
      data: {{
        labels: prodes.map(r => r.year),
        datasets: [{{
          label: 'ha desmatados (PRODES)',
          data: prodes.map(r => r.deforestation_ha),
          borderColor: '#2e7d32',
          backgroundColor: '#2e7d3233',
          fill: true,
          tension: 0.25
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#1b3d1f' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#5a7260' }}, grid: {{ color: '#c8e6c9' }} }},
          y: {{ ticks: {{ color: '#5a7260' }}, grid: {{ color: '#c8e6c9' }} }}
        }}
      }}
    }});

    const imp = DATA.feature_importance || [];
    const maxImp = Math.max(...imp.map(x => Number(x.importance || 0)), 1e-9);
    const tbody = document.querySelector('#featTable tbody');
    imp.forEach(row => {{
      const name = row.feature || '';
      const help = featureHelp(name);
      const tr = document.createElement('tr');
      const pct = (100 * Number(row.importance || 0) / maxImp).toFixed(1);
      tr.innerHTML = `
        <td>
          <div class="feat-name">${{help.label}}</div>
          <div class="feat-code">${{name}}</div>
        </td>
        <td>${{Number(row.importance || 0).toFixed(4)}}</td>
        <td><div class="bar"><span style="width:${{pct}}%"></span></div></td>
        <td>
          <div class="feat-origin"><strong>Origem:</strong> ${{help.origin}}</div>
          <div class="feat-meaning">${{help.meaning}}</div>
        </td>`;
      tbody.appendChild(tr);
    }});

    const ml = DATA.ml_model || {{}};
    document.getElementById('mlCtx').textContent = JSON.stringify({{
      target_col: ml.target_col,
      horizon_years: ml.horizon_years,
      n_train: ml.n_train,
      n_valid: ml.n_valid,
      n_test: ml.n_test,
      train_years: ml.train_years,
      test_year: ml.test_year,
      positive_rate_test: ml.positive_rate_test,
      n_features: ml.n_features,
      feature_names: ml.feature_names
    }}, null, 2);
    document.getElementById('mlParams').textContent = JSON.stringify({{
      params: ml.params,
      booster_stats: ml.booster_stats,
      notes: ml.notes
    }}, null, 2);

    const pr = (DATA.metrics && DATA.metrics.test && DATA.metrics.test.pr_curve) || [];
    if (pr.length) {{
      new Chart(document.getElementById('chartPR'), {{
        type: 'line',
        data: {{
          labels: pr.map((_, i) => i),
          datasets: [{{
            label: 'Precision–Recall',
            data: pr.map(p => ({{x: p.recall, y: p.precision}})),
            borderColor: '#558b2f',
            tension: 0.15,
            pointRadius: 2,
          }}]
        }},
        options: {{
          parsing: false,
          scales: {{
            x: {{ type: 'linear', title: {{display:true, text:'Recall (desmates capturados)', color:'#5a7260'}}, min:0, max:1, ticks:{{color:'#5a7260'}}, grid:{{color:'#c8e6c9'}} }},
            y: {{ title: {{display:true, text:'Precision (alertas corretos)', color:'#5a7260'}}, min:0, max:1, ticks:{{color:'#5a7260'}}, grid:{{color:'#c8e6c9'}} }}
          }},
          plugins: {{ legend: {{ labels: {{ color: '#1b3d1f' }} }} }}
        }}
      }});
    }}
    // Gráfico de custo/limiar desativado no dashboard (métricas confusas para o uso principal).

    document.getElementById('warns').textContent = (DATA.warnings || []).join(' · ');
  
  (function() {{
    const backdrop = document.getElementById('metricModal');
    const titleEl = document.getElementById('metricModalTitle');
    const bodyEl = document.getElementById('metricModalBody');
    const closeBtn = document.getElementById('metricModalClose');
    function openModal(title, hint) {{
      titleEl.textContent = title || 'Métrica';
      bodyEl.textContent = hint || '';
      backdrop.classList.add('open');
    }}
    function closeModal() {{ backdrop.classList.remove('open'); }}
    document.querySelectorAll('.info-btn').forEach(btn => {{
      btn.addEventListener('click', () => openModal(btn.dataset.title, btn.dataset.hint));
    }});
    closeBtn.addEventListener('click', closeModal);
    backdrop.addEventListener('click', (e) => {{ if (e.target === backdrop) closeModal(); }});
    document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeModal(); }});
  }})();

</script>
</body>
</html>
"""
