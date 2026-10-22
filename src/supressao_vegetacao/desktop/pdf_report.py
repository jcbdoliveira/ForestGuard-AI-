"""
Relatório Técnico ForestGuard — PDF profissional (ReportLab).

Gera documento A4 com:
  1. Capa + logo + metadados + nota metodológica
  2. Sumário executivo e perfil municipal
  3+. Cenários por horizonte com top células
  Final. Comparação de horizontes, métricas e manifesto

Compatível com a assinatura legada:
  generate_technical_report_pdf(analysis_result, output_path) -> str
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as pdfcanvas

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

GREEN_DARK = colors.Color(0.08, 0.35, 0.22)
GREEN_MID = colors.Color(0.12, 0.45, 0.28)
GREEN_SOFT = colors.Color(0.90, 0.96, 0.92)
AMBER_BG = colors.Color(0.99, 0.96, 0.88)
AMBER_BORDER = colors.Color(0.85, 0.65, 0.15)
GRAY = colors.Color(0.25, 0.25, 0.25)
LIGHT_GRAY = colors.Color(0.94, 0.94, 0.94)
WHITE = colors.white


def _find_logo_path() -> str | None:
    """Localiza logo ForestGuard. Preferir PNG (transparência); JPEG como fallback."""
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "..", "..", "..", "assets", "logo_vigia.png"),
        os.path.join(here, "..", "..", "..", "public", "logo_vigia.png"),
        os.path.join(os.getcwd(), "assets", "logo_vigia.png"),
        os.path.join(os.getcwd(), "public", "logo_vigia.png"),
        os.path.join(here, "..", "..", "..", "assets", "logo_vigia.jpg"),
        os.path.join(here, "..", "..", "..", "public", "logo_vigia.jpg"),
        os.path.join(os.getcwd(), "assets", "logo_vigia.jpg"),
        os.path.join(os.getcwd(), "public", "logo_vigia.jpg"),
        os.environ.get("FORESTGUARD_LOGO", ""),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "cover_kicker", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9, textColor=GRAY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=18, textColor=GREEN_DARK, alignment=TA_CENTER, leading=22, spaceAfter=4,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=12,
        ),
        "brand": ParagraphStyle(
            "brand", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=14, textColor=GREEN_DARK, alignment=TA_CENTER, spaceAfter=2,
        ),
        "brand_tag": ParagraphStyle(
            "brand_tag", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, textColor=GRAY, alignment=TA_CENTER, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, textColor=GREEN_DARK, spaceBefore=8, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, textColor=GREEN_MID, spaceBefore=10, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=GRAY, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.5, leading=10, textColor=GRAY,
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=colors.Color(0.35, 0.25, 0.05), alignment=TA_JUSTIFY,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10, textColor=GRAY,
        ),
        "cell_b": ParagraphStyle(
            "cell_b", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=WHITE,
        ),
    }


class ForestGuardLogo(Flowable):
    """Logo ForestGuard (árvore-dados). PNG com transparência ou JPEG RGB."""

    def __init__(self, width: float = 120, height: float = 138):
        super().__init__()
        self.logo_path = _find_logo_path()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        if self.logo_path:
            try:
                from reportlab.lib.utils import ImageReader
                img = ImageReader(self.logo_path)
                c.drawImage(
                    img,
                    0,
                    0,
                    width=self.width,
                    height=self.height,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
                return
            except Exception:
                try:
                    c.drawImage(
                        self.logo_path,
                        0,
                        0,
                        width=self.width,
                        height=self.height,
                        preserveAspectRatio=True,
                        anchor="c",
                        mask="auto",
                    )
                    return
                except Exception:
                    pass
        # Fallback vetorial discreto (árvore geométrica)
        s = min(self.width, self.height)
        ox = (self.width - s) / 2
        oy = (self.height - s) / 2
        c.setFillColor(GREEN_DARK)
        path = c.beginPath()
        path.moveTo(ox + s * 0.50, oy + s * 0.88)
        path.lineTo(ox + s * 0.18, oy + s * 0.42)
        path.lineTo(ox + s * 0.82, oy + s * 0.42)
        path.close()
        c.drawPath(path, fill=1, stroke=0)
        c.setFillColor(colors.Color(0.35, 0.25, 0.12))
        c.rect(ox + s * 0.44, oy + s * 0.18, s * 0.12, s * 0.30, fill=1, stroke=0)


class ColoredBox(Flowable):
    def __init__(self, content: Paragraph, width: float, bg=AMBER_BG, border=AMBER_BORDER, pad=8):
        super().__init__()
        self.content = content
        self.box_width = width
        self.bg = bg
        self.border = border
        self.pad = pad
        w, h = content.wrap(width - 2 * pad, 500)
        self._cw, self._ch = w, h
        self.width = width
        self.height = h + 2 * pad

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.setStrokeColor(self.border)
        c.setLineWidth(1.2)
        c.roundRect(0, 0, self.box_width, self.height, 4, fill=1, stroke=1)
        self.content.drawOn(c, self.pad, self.pad)


def _horizon_keys(summaries: Dict[str, Any]) -> List[str]:
    order = ["1y", "3y", "5y", "8y", "10y", "20y"]
    return [k for k in order if k in summaries] or sorted(summaries.keys())


def _fmt_ha(v: Any) -> str:
    try:
        return f"{float(v):,.2f} ha".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(v)


def _fmt_pct(v: Any) -> str:
    """Formata percentual. Aceita fração (0.013) ou já em % (1.3 / 13)."""
    try:
        fv = float(v)
        # Valores já em escala percentual (ex.: 1.37 ou 13.5)
        if abs(fv) > 1.0:
            return f"{fv:.1f}%".replace(".", ",")
        return f"{fv * 100:.1f}%".replace(".", ",")
    except Exception:
        return str(v)


def _meta_table(rows: List[Tuple[str, str]], styles: Dict[str, ParagraphStyle], col_w: Sequence[float]) -> Table:
    data = [[Paragraph(a, styles["cell"]), Paragraph(b, styles["cell"])] for a, b in rows]
    t = Table(data, colWidths=list(col_w))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
        ("BACKGROUND", (1, 0), (1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.Color(0.8, 0.8, 0.8)),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.Color(0.85, 0.85, 0.85)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _data_table(header: List[str], rows: List[List[str]], styles: Dict[str, ParagraphStyle], col_w: Sequence[float]) -> Table:
    head = [Paragraph(h, styles["cell_b"]) for h in header]
    body = [[Paragraph(str(c), styles["cell"]) for c in row] for row in rows]
    t = Table([head] + body, colWidths=list(col_w))
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.Color(0.75, 0.75, 0.75)),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.Color(0.88, 0.88, 0.88)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(body) + 1):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), GREEN_SOFT))
    t.setStyle(TableStyle(cmds))
    return t


def _page_footer(canvas: pdfcanvas.Canvas, doc, muni_name: str, ibge: str, version: str):
    canvas.saveState()
    canvas.setStrokeColor(GREEN_MID)
    canvas.setLineWidth(0.8)
    canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.Color(0.4, 0.4, 0.4))
    canvas.drawString(MARGIN, 7 * mm, f"ForestGuard IA {version} | {muni_name} (IBGE {ibge})")
    canvas.drawRightString(PAGE_W - MARGIN, 7 * mm, f"Página {doc.page}")
    canvas.restoreState()


def generate_technical_report_pdf(analysis_result: Dict[str, Any], output_path: str) -> str:
    m = analysis_result.get("municipality") or {}
    meta = analysis_result.get("model_metadata") or {}
    summaries = analysis_result.get("horizon_summaries") or {}
    styles = _styles()

    muni_name = str(m.get("name") or "Municipio")
    muni_uf = str(m.get("uf") or "")
    ibge = str(m.get("ibge_code") or "")
    version = str(meta.get("model_version") or "v13.0.0")
    biome = str(m.get("biome") or "-")
    total_ha = float(m.get("total_area_ha") or 0)
    remnant_ha = float(m.get("forest_remnant_ha") or 0)
    grid = m.get("grid_dimensions") or {}
    n_cells = int(grid.get("total_cells") or len(analysis_result.get("cells") or []))
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    spatial = meta.get("spatial_model") or {}
    metrics = meta.get("metrics_validation") or {}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm, bottomMargin=18 * mm,
        title=f"ForestGuard - {muni_name}/{muni_uf}", author="ForestGuard AI",
    )
    story: List[Any] = []
    usable_w = PAGE_W - 2 * MARGIN

    story.append(Spacer(1, 6 * mm))
    # Logo centralizado sem faixa preta lateral (só a largura do símbolo)
    logo_w = min(usable_w * 0.38, 50 * mm)
    logo_h = logo_w * (761 / 664)
    logo_table = Table(
        [[ForestGuardLogo(width=logo_w, height=logo_h)]],
        colWidths=[logo_w],
    )
    logo_table.hAlign = "CENTER"
    logo_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(logo_table)
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("RELATÓRIO TÉCNICO DE PREDIÇÃO ESPACIAL", styles["cover_kicker"]))
    story.append(Paragraph("Estimativa de Risco Relativo de<br/>Supressão de Vegetação", styles["cover_title"]))
    story.append(Paragraph("Ferramenta acadêmica e modular de apoio à tomada de decisão", styles["cover_sub"]))
    line = Table([[""]], colWidths=[usable_w], rowHeights=[2])
    line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREEN_DARK)]))
    story.append(line)
    story.append(Spacer(1, 6 * mm))
    story.append(_meta_table([
        ("Municipio", f"{muni_name} - {muni_uf}"),
        ("Codigo IBGE", ibge),
        ("Data de emissao", issued),
        ("Versao do modelo", version),
        ("ID de auditoria", f"AUDIT_ML_{ibge}_001"),
        ("Unidade espacial", "Grade regular (poligono IBGE / SIRGAS 2000)"),
        ("Fontes declaradas", "IBGE Malhas, INPE PRODES (catalogo), modelo espacial XGBoost"),
    ], styles, [45 * mm, usable_w - 45 * mm]))
    story.append(Spacer(1, 6 * mm))

    # Nota metodológica dinâmica: distingue painel GEE real vs sintético / confiabilidade
    mp = analysis_result.get("municipal_projection") or {}
    rel = mp.get("forecast_reliability") or meta.get("forecast_reliability") or {}
    spatial_src = str(
        (spatial or {}).get("data_source")
        or meta.get("data_source")
        or ""
    )
    model_class = str((spatial or {}).get("model_class") or "")
    spatial_path = str((spatial or {}).get("path") or "")
    spatial_loaded = bool((spatial or {}).get("loaded"))
    is_real_panel = (
        "gee_mapbiomas" in spatial_src
        or spatial_src == "gee_mapbiomas_panel"
        or "municipality_trained" in model_class
        or ("4210100" in spatial_path and spatial_loaded)
    )
    reliable = bool(rel.get("reliable"))
    rel_reason = str(rel.get("reason") or "nao_informado")

    if is_real_panel and spatial_loaded:
        model_class_txt = (
            "modelo espacial XGBoost treinado em <b>painel GEE/MapBiomas real</b> "
            f"(classe: {model_class or 'municipality_trained'}; "
            f"fonte: {spatial_src or 'gee_mapbiomas_panel'})."
        )
    elif spatial_loaded:
        model_class_txt = (
            "modelo espacial carregado, mas a origem do painel nao foi confirmada "
            f"(classe: {model_class or 'desconhecida'})."
        )
    else:
        model_class_txt = (
            "modelo espacial em modo <b>heuristico/demonstrativo</b> "
            "(painel sintetico ou features incompletas quando o painel GEE nao esta disponivel)."
        )
    if not reliable:
        model_class_txt += (
            f" Projecao municipal Ridge classificada como <b>NAO CONFIAVEL</b> "
            f"(motivo: {rel_reason}). "
            "Isso afeta a perda municipal esperada; os niveis de risco por celula "
            "continuam baseados no XGBoost espacial quando disponivel."
        )

    note_txt = (
        "<b>NOTA METODOLOGICA OBRIGATORIA:</b> Este documento e uma estimativa de "
        "<b>risco relativo</b> baseada em modelagem matematica. Nao substitui validacao "
        "de campo, auditoria presencial ou manifestacao de orgaos ambientais competentes. "
        "A classificacao em vermelho representa <b>maior risco relativo estimado</b> e "
        "<b>nao</b> constitui desmatamento consumado, denuncia criminal ou auto de infracao. "
        f"Neste municipio: {model_class_txt}"
    )
    story.append(ColoredBox(Paragraph(note_txt, styles["note"]), usable_w))
    story.append(PageBreak())

    story.append(Paragraph("1. Sumario executivo", styles["h1"]))
    hkeys = _horizon_keys(summaries)
    story.append(Paragraph(
        f"Diagnostico espacial do risco relativo estimado de supressao da vegetacao nativa "
        f"para <b>{muni_name}/{muni_uf}</b> (IBGE {ibge}), com malha de <b>{n_cells}</b> celulas "
        f"sobre o poligono oficial. Remanescente declarado: <b>{_fmt_ha(remnant_ha)}</b> "
        f"em area total de <b>{_fmt_ha(total_ha)}</b>.",
        styles["body"],
    ))
    if hkeys:
        s0 = summaries[hkeys[0]]
        crit = int(s0.get("critical_points_count") or 0)
        high = int(s0.get("high_points_count") or 0)
        mod = int(s0.get("moderate_points_count") or 0)
        low = int(s0.get("low_points_count") or 0)
        ranked = crit + high + mod + low
        rel_line = ""
        if not reliable:
            rel_line = (
                f" <b>Atencao:</b> a projecao municipal Ridge esta "
                f"<b>NAO CONFIAVEL</b> ({rel_reason}); a perda em hectares deve ser "
                f"interpretada com cautela. Os contadores de risco por celula usam o "
                f"modelo espacial XGBoost quando disponivel."
            )
        if ranked == 0:
            rel_line += (
                " Nenhuma celula cruzou os limiares de risco (baixo &lt; 0,25; "
                "moderado 0,25–0,50; alto 0,50–0,75; critico ≥ 0,75)."
            )
        story.append(Paragraph(
            f"No horizonte de {s0.get('horizon_years', hkeys[0])} anos: "
            f"<b>{crit}</b> celulas criticas, "
            f"<b>{high}</b> alto, "
            f"<b>{mod}</b> moderado e "
            f"<b>{low}</b> baixo. "
            f"Perda municipal esperada: <b>{_fmt_ha(s0.get('expected_suppressed_area_ha'))}</b>."
            f"{rel_line}",
            styles["body"],
        ))

    story.append(Paragraph("2. Perfil municipal", styles["h1"]))
    story.append(_data_table(
        ["Indicador", "Valor", "Fonte"],
        [
            ["Codigo IBGE", ibge, "IBGE"],
            ["UF / municipio", f"{muni_name}/{muni_uf}", "Catalogo ForestGuard"],
            ["Bioma", biome, "Catalogo"],
            ["Area total", _fmt_ha(total_ha), "Malha IBGE / catalogo"],
            ["Remanescente florestal", _fmt_ha(remnant_ha), "Catalogo / MapBiomas (ref.)"],
            ["Celulas na malha", str(n_cells), "Grade metrica"],
        ],
        styles, [55 * mm, 55 * mm, usable_w - 110 * mm],
    ))

    story.append(Paragraph("3. Metodologia (resumo)", styles["h1"]))
    story.append(Paragraph(
        "A projecao municipal utiliza regressao Ridge treinada nas series PRODES do catalogo "
        "(validacao temporal). A distribuicao espacial do risco usa modelo XGBoost quando "
        "disponivel; features sem raster podem ser imputadas com medianas do treino. "
        "A distribuicao de hectares e ponderada pelo risco relativo e limitada pela capacidade "
        "de floresta de cada celula, com sobras redistribuidas. Sem raster observado por celula, "
        "o remanescente e distribuido uniformemente na grade. Limiares: baixo &lt; 0,25; "
        "moderado 0,25-0,50; alto 0,50-0,75; critico &gt;= 0,75. A faixa de sensibilidade "
        "(+/-25%) <b>nao</b> e intervalo de confianca estatistico.",
        styles["body"],
    ))
    sm_feats = spatial.get("features") or []
    story.append(Paragraph(
        f"Modelo espacial: <b>{'carregado' if spatial.get('loaded') else 'nao carregado'}</b> "
        f"| classe: <b>{spatial.get('model_class') or 'n/d'}</b> "
        f"| features: <b>{', '.join(sm_feats) if sm_feats else 'n/d'}</b> "
        f"| imputacao: <b>{'sim' if spatial.get('features_imputed') else 'nao'}</b>.",
        styles["body"],
    ))

    for idx, hk in enumerate(hkeys):
        story.append(PageBreak())
        s = summaries[hk]
        years = s.get("horizon_years") or hk.replace("y", "")
        story.append(Paragraph(f"4.{idx + 1} Cenario - {years} anos (horizonte {hk})", styles["h1"]))
        story.append(_data_table(
            ["Metrica", "Valor"],
            [
                ["Ano-alvo", str(s.get("target_year") or "-")],
                ["Perda esperada (ha)", _fmt_ha(s.get("expected_suppressed_area_ha"))],
                ["% do remanescente", _fmt_pct(s.get("percent_of_remnant_lost"))],
                ["Prob. media", f"{float(s.get('mean_probability') or 0):.3f}"],
                ["Prob. maxima", f"{float(s.get('max_probability') or 0):.3f}"],
                ["Critico / alto / mod. / baixo",
                 f"{s.get('critical_points_count', 0)} / {s.get('high_points_count', 0)} / "
                 f"{s.get('moderate_points_count', 0)} / {s.get('low_points_count', 0)}"],
                ["Perda alocada / nao alocada (ha)",
                 f"{_fmt_ha((s.get('spatial_allocation') or {}).get('allocated_ha'))} / "
                 f"{_fmt_ha((s.get('spatial_allocation') or {}).get('unallocated_ha'))}"],
                ["Metodo de alocacao", str((s.get('spatial_allocation') or {}).get('method') or "n/d")],
            ],
            styles, [70 * mm, usable_w - 70 * mm],
        ))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Top 10 celulas de atencao", styles["h2"]))
        tops = (s.get("top_attention_points") or [])[:10]
        if tops:
            rows = []
            for pt in tops:
                coords = pt.get("coordinates") or pt.get("centroid") or [None, None]
                lat, lon = (coords[0], coords[1]) if len(coords) >= 2 else (None, None)
                if lat is not None and lon is not None:
                    try:
                        lat_f, lon_f = float(lat), float(lon)
                        gmaps = (
                            f"https://www.google.com/maps?q={lat_f:.6f},{lon_f:.6f}"
                            f"&ll={lat_f:.6f},{lon_f:.6f}&z=17"
                            )
                        # Coordenadas + link clicavel
                        coord_cell = (
                            f"{lat_f:.4f}, {lon_f:.4f}<br/>"
                            f'<link href="{gmaps}" color="#1565C0">'
                            f"<u>Geolocalização</u></link>"
                        )
                    except (TypeError, ValueError):
                        coord_cell = "-"

                else:
                    coord_cell = "-"                
                rows.append([
                    str(pt.get("rank") or ""),
                    str(pt.get("cell_id") or "")[:22],
                    str(pt.get("attention_level") or ""),
                    f"{float(pt.get('probability') or 0):.3f}",
                    _fmt_ha(pt.get("expected_suppressed_ha")),
                    coord_cell,
                ])
            story.append(_data_table(
                ["#", "Celula", "Nivel", "Prob.", "Perda (ha)", "Lat, Lon / Geolocalização"],
                rows, styles,
                [10 * mm, 30 * mm, 20 * mm, 16 * mm, 24 * mm, usable_w - 100 * mm],
           ))
        else:
            story.append(Paragraph("Sem pontos de atencao disponiveis neste horizonte.", styles["body"]))
        story.append(Paragraph(
            "Aviso: cores e ranks indicam risco relativo estimado, nao ocorrencia confirmada de desmate.",
            styles["small"],
        ))

    story.append(PageBreak())
    story.append(Paragraph("5. Comparacao entre horizontes", styles["h1"]))
    comp_rows = []
    for hk in hkeys:
        s = summaries[hk]
        comp_rows.append([
            str(s.get("horizon_years") or hk),
            str(s.get("target_year") or "-"),
            _fmt_ha(s.get("expected_suppressed_area_ha")),
            f"{float(s.get('mean_probability') or 0):.3f}",
            str(s.get("critical_points_count") or 0),
        ])
    story.append(_data_table(
        ["Horizonte (a)", "Ano-alvo", "Perda esperada", "Prob. media", "Criticos"],
        comp_rows, styles,
        [30 * mm, 30 * mm, 45 * mm, 35 * mm, usable_w - 140 * mm],
    ))

    story.append(Paragraph("6. Metricas de validacao", styles["h1"]))
    projection_info = analysis_result.get("municipal_projection") or {}
    for warning in (projection_info.get("warnings") or []):
        story.append(ColoredBox(Paragraph(f"<b>Ressalva metodologica:</b> {warning}", styles["note"]), usable_w))
    forecast = meta.get("forecast_reliability") or {}
    story.append(Paragraph(
        "Confiabilidade municipal: <b>%s</b>. RMSE da persistencia (ha): <b>%s</b>; "
        "RMSE da media historica (ha): <b>%s</b>; cortes walk-forward: <b>%s</b>. "
        "A classificacao so e positiva quando supera a melhor referencia com margem minima e amostra suficiente. "
        "Mesmo assim, o resultado e exploratorio e nao equivale a probabilidade calibrada nem substitui validacao de campo."
        % ("CONFIAVEL PARA EXPLORACAO" if forecast.get("reliable") else "NAO CONFIAVEL PARA DECISAO",           
           forecast.get("rmse_persistence_ha", "-"), forecast.get("rmse_historical_mean_ha", "-"),
           forecast.get("n_cutoffs", "-")), styles["body"]))
    story.append(_data_table(
        ["Metrica", "Valor", "Observacao"],
        [
            ["ROC-AUC espacial", f"{float(metrics.get('roc_auc') or 0):.3f}",
             "Holdout temporal do painel GEE" if is_real_panel else "Pode ser instavel em painel sintetico"],
            ["PR-AUC", f"{float(metrics.get('pr_auc') or 0):.3f}", "-"],
            ["Brier score", f"{float(metrics.get('brier_score') or 0):.3f}", "Menor e melhor"],
            ["Recall top-10%", f"{float(metrics.get('recall_at_top10_hotspots') or 0):.3f}", "-"],
            ["MAE municipal (ha)", str(metrics.get("mae_ha") or "-"), "Modelo Ridge municipal"],
        ],
        styles, [45 * mm, 30 * mm, usable_w - 75 * mm],
    ))

    story.append(Paragraph("7. Manifesto de limitacoes", styles["h1"]))
    lims = list(meta.get("limitations") or [])
    if not lims:
        lims = [
            "Series e atributos municipais estao no catalogo local (proveniencia parcial).",
            "Nao substitui analise de campo nem decisao administrativa formal.",
        ]
        if is_real_panel:
            lims.insert(1, "Modelo espacial treinado em painel GEE/MapBiomas real; calibração de probabilidade ainda limitada.")
        else:
            lims.insert(1, "Modelo espacial pode usar painel sintetico e imputacao de features.")
        if not reliable:
            lims.insert(0, f"Projecao municipal NAO CONFIAVEL ({rel_reason}): skill negativo ou amostra insuficiente vs baselines.")
    for item in lims:
        story.append(Paragraph(f"• {item}", styles["body"]))

    story.append(Spacer(1, 8 * mm))
    if is_real_panel and reliable:
        class_txt = (
            "<b>Classificacao do produto:</b> modelo espacial com painel GEE real e projecao municipal "
            "com skill positivo. Ainda assim e <b>exploratorio</b> — nao substitui validacao de campo."
        )
    elif is_real_panel:
        class_txt = (
            "<b>Classificacao do produto:</b> modelo espacial com painel GEE real, porem projecao "
            "municipal <b>NAO CONFIAVEL</b> para decisao. Uso: pesquisa e desenvolvimento de pipeline."
        )
    else:
        class_txt = (
            "<b>Classificacao do produto:</b> prototipo / heuristica espacial demonstrativa. "
            "Uso adequado: pesquisa, desenvolvimento de pipeline e apoio exploratorio — "
            "nao evidencia operacional isolada."
        )
    story.append(ColoredBox(
        Paragraph(class_txt, styles["note"]),
        usable_w, bg=GREEN_SOFT, border=GREEN_MID,
    ))

    def _on_page(canvas, doc_):
        _page_footer(canvas, doc_, muni_name, ibge, version)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return output_path
