"""
Módulo de Interface Gráfica Desktop (Tkinter) para Inferência e Diagnóstico de Desmatamento.
Suporta:
- Mapa interativo no Tkinter Canvas com grade territorial de maior resolução,
  colorização de risco relativo, contornos e seleção interativa com clique.
- Botão "Gerar Relatório Técnico em PDF" exatamente idêntico ao padrão oficial de 7 páginas.
- Seletor de horizontes interativo (1, 3, 5, 8, 10 e 20 anos). Risco espacial anual quando modelo y_loss_1y disponível.
- Painel de métricas e síntese espacial em tempo real.
- Suporte a execução em ambientes headless via Console Fallback com exportação automática de PDF.
"""

import os
import sys
import webbrowser
import math
from pathlib import Path
from typing import Optional, Dict, Any

from ..sources.municipality import list_municipalities, get_municipality
from ..sources.city_resolver import resolve_municipality_code  # v16: résolução movida p/ sources (parecer C3)
from ..inference.predict import run_city_deforestation_analysis
from ..settings import SUPPORTED_HORIZONS
from .pdf_report import generate_technical_report_pdf


MAP_METRIC_LOSS = "Perda projetada (ha)"
MAP_METRIC_RISK = "Risco espacial (%)"
MAP_COLOR_CLASSES = 15
MAP_NO_DATA_COLOR = "#334155"
# Paleta ordenada do menor ao maior valor (verde → amarelo → vermelho).
# Quinze classes discretas; a mesma paleta é usada nas duas camadas.
MAP_COLOR_PALETTE = (
    "#006837",
    "#1a9850",
    "#3daa5c",
    "#66bd63",
    "#8ece6e",
    "#a6d96a",
    "#c4e576",
    "#d9ef8b",
    "#fee08b",
    "#fdd06e",
    "#fdae61",
    "#f7944e",
    "#f46d43",
    "#e34933",
    "#d73027",
)


def get_risk_color_class(prob: float) -> Optional[int]:
    """Converte um escore entre 0 e 1 em uma de quinze classes de risco."""
    try:
        value = float(prob)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    value = max(0.0, min(1.0, value))
    return min(MAP_COLOR_CLASSES - 1, int(value * MAP_COLOR_CLASSES))


def get_risk_color(level: str, prob: float) -> str:
    """Cor de risco em quinze faixas do escore; cinza indica valor indisponível.

    ``level`` é mantido na assinatura por compatibilidade; as faixas são
    definidas numericamente pelo escore para que a legenda corresponda à cor.
    """
    color_class = get_risk_color_class(prob)
    if color_class is None:
        return MAP_NO_DATA_COLOR
    return MAP_COLOR_PALETTE[color_class]


def get_global_loss_scale(cells: list[dict]) -> float:
    """Maior perda por célula entre todos os horizontes, para manter escala fixa."""
    values = []
    for cell in cells:
        for scenario in (cell.get("scenarios") or {}).values():
            try:
                value = float(scenario.get("expected_suppressed_ha", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)
    return max(values, default=0.0)


def get_loss_color_class(loss_ha: float, scale_max_ha: float) -> Optional[int]:
    """Classifica a perda em quinze faixas com escala fixa entre horizontes.

    A raiz quadrada mantém maior resolução visual para perdas pequenas. Os
    limites das classes são definidos na mesma escala global do município.
    ``None`` representa ausência de perda/escala válida, não uma classe de risco.
    """
    try:
        value = float(loss_ha)
        maximum = float(scale_max_ha)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not math.isfinite(maximum) or value <= 0.0 or maximum <= 0.0:
        return None
    t = math.sqrt(min(1.0, max(0.0, value / maximum)))
    return min(MAP_COLOR_CLASSES - 1, int(t * MAP_COLOR_CLASSES))


def get_loss_color(loss_ha: float, scale_max_ha: float) -> str:
    """Cor discreta (uma de quinze) para a perda acumulada por célula."""
    color_class = get_loss_color_class(loss_ha, scale_max_ha)
    if color_class is None:
        return MAP_NO_DATA_COLOR
    return MAP_COLOR_PALETTE[color_class]


def get_cell_map_color(cell: dict, horizon: str, metric: str, loss_scale_max_ha: float) -> str:
    """Retorna a cor do modo escolhido para o cenário do horizonte solicitado."""
    scenario = (cell.get("scenarios") or {}).get(horizon) or {}
    if metric == MAP_METRIC_LOSS:
        return get_loss_color(scenario.get("expected_suppressed_ha", 0.0), loss_scale_max_ha)
    raw_probability = scenario.get("probability")
    if raw_probability is None:
        return MAP_NO_DATA_COLOR
    try:
        probability = float(raw_probability)
    except (TypeError, ValueError):
        return MAP_NO_DATA_COLOR
    return get_risk_color(str(scenario.get("attention_level", "BAIXO")), probability)


def run_desktop_app():
    """Inicia a interface Tkinter interativa com Canvas cartográfico dinâmico e exportador de PDF."""
    # v16 (parecer C3): imports de GUI movidos para dentro da função — o módulo
    # permanece importável em servidores headless (a API não depende mais dele,
    # mas scripts legados ainda o importam).
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except (ImportError, ModuleNotFoundError):
        print("\n[Aviso] O módulo 'tkinter' não está instalado neste ambiente Linux headless.")
        print("[Desktop] Iniciando modo interativo via terminal com geração de PDF...\n")
        _run_terminal_interactive_gui()
        return

    # Se tkinter existe, tenta criar a janela raiz
    try:
        root = tk.Tk()
    except Exception as e:
        print(f"\n[Aviso] Servidor gráfico não detectado ($DISPLAY ausente: {e}).")
        print("[Desktop] Executando em modo interativo de console com geração de PDF...\n")
        _run_terminal_interactive_gui()
        return

    root.title("ForestGuard AI — Diagnóstico Territorial & Relatório Técnico PDF")
    root.geometry("1180x760")
    root.minsize(980, 640)
    root.configure(bg="#0b1120")
    # Maximiza a janela principal (Windows: zoomed; Linux/mac: fallback)
    try:
        root.state("zoomed")
    except Exception:
        try:
            root.attributes("-zoomed", True)
        except Exception:
            try:
                root.update_idletasks()
                sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                root.geometry(f"{sw}x{sh}+0+0")
            except Exception:
                pass

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#0b1120", foreground="#f1f5f9")
    style.configure("TFrame", background="#0b1120")
    style.configure("TLabelframe", background="#0f172a", foreground="#38bdf8")
    style.configure("TLabelframe.Label", background="#0f172a", foreground="#38bdf8", font=("Helvetica", 10, "bold"))

    # Estado da aplicação
    current_res: Dict[str, Any] = {}
    selected_horizon = tk.StringVar(value="1y")
    selected_cell_id = tk.StringVar(value="")

    # Barra Superior de Controles e Ações
    top_bar = tk.Frame(root, bg="#0f172a", padx=16, pady=12, highlightbackground="#1e293b", highlightthickness=1)
    top_bar.pack(fill="x", side="top")

    tk.Label(top_bar, text="Município / IBGE:", font=("Helvetica", 10, "bold"), bg="#0f172a", fg="#94a3b8").pack(side="left", padx=(0, 6))

    entry_muni = tk.Entry(top_bar, font=("Helvetica", 10), bg="#1e293b", fg="#ffffff", insertbackground="white", width=18, relief="flat", highlightbackground="#334155", highlightthickness=1)
    entry_muni.insert(0, "MAFRA-SC")
    entry_muni.pack(side="left", padx=(0, 12))

    tk.Label(top_bar, text="Pressão (%):", font=("Helvetica", 10), bg="#0f172a", fg="#94a3b8").pack(side="left", padx=(0, 4))
    entry_pressure = tk.Entry(top_bar, font=("Helvetica", 10), bg="#1e293b", fg="#ffffff", insertbackground="white", width=6, relief="flat", highlightbackground="#334155", highlightthickness=1)
    entry_pressure.insert(0, "100")
    entry_pressure.pack(side="left", padx=(0, 12))

    # Botão Consultar
    btn_calc = tk.Button(
        top_bar, text="🔍 Consultar Análise", font=("Helvetica", 10, "bold"),
        bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
        padx=12, pady=4, relief="flat", cursor="hand2", command=lambda: process_query()
    )
    btn_calc.pack(side="left", padx=(0, 16))

    # Seletor de Horizontes
    tk.Label(top_bar, text="Horizonte:", font=("Helvetica", 10, "bold"), bg="#0f172a", fg="#94a3b8").pack(side="left", padx=(8, 6))
    for h in ["1y", "3y", "5y", "8y", "10y", "20y"]:
        rb = tk.Radiobutton(
            top_bar, text=f"{h[:-1]} Anos", variable=selected_horizon, value=h,
            bg="#0f172a", fg="#e2e8f0", selectcolor="#0369a1", activebackground="#0f172a",
            activeforeground="#38bdf8", font=("Helvetica", 9, "bold"), command=lambda: update_map_view()
        )
        rb.pack(side="left", padx=4)

    # Botão de Gerar PDF Oficial (Destaque Verde Esmeralda)
    btn_pdf = tk.Button(
        top_bar, text="📄 Gerar Relatório Técnico em PDF", font=("Helvetica", 10, "bold"),
        bg="#10b981", fg="#022c22", activebackground="#34d399", activeforeground="#022c22",
        padx=14, pady=5, relief="flat", cursor="hand2", command=lambda: export_pdf_action()
    )
    btn_pdf.pack(side="right", padx=(10, 0))

    # Layout Principal: Esquerda (Mapa Interativo Canvas) | Direita (Indicadores e Detalhes)
    main_container = tk.Frame(root, bg="#0b1120", padx=12, pady=10)
    main_container.pack(fill="both", expand=True)

    # Painel Esquerdo: mapa por horizonte. O risco espacial tende a ser estático;
    # a perda projetada por célula é o cenário que deve variar ao selecionar outro ano.
    map_frame = tk.LabelFrame(main_container, text=" Malha Territorial Cartográfica (SIRGAS 2000) ", bg="#0f172a", fg="#38bdf8", font=("Helvetica", 10, "bold"), padx=8, pady=8)
    map_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

    map_toolbar = tk.Frame(map_frame, bg="#0f172a")
    map_toolbar.pack(side="top", fill="x", pady=(0, 6))
    tk.Label(map_toolbar, text="Camada:", bg="#0f172a", fg="#94a3b8", font=("Helvetica", 9, "bold")).pack(side="left", padx=(2, 6))
    selected_map_metric = tk.StringVar(value=MAP_METRIC_LOSS)
    map_metric_combo = ttk.Combobox(
        map_toolbar,
        textvariable=selected_map_metric,
        values=(MAP_METRIC_LOSS, MAP_METRIC_RISK),
        state="readonly",
        width=23,
    )
    map_metric_combo.pack(side="left")
    map_metric_combo.bind("<<ComboboxSelected>>", lambda event: update_map_view())
    lbl_map_hint = tk.Label(
        map_toolbar,
        text="Perda acumulada por célula; escala fixa entre horizontes.",
        bg="#0f172a", fg="#94a3b8", font=("Helvetica", 8), anchor="e",
    )
    lbl_map_hint.pack(side="left", fill="x", expand=True, padx=(10, 2))

    canvas = tk.Canvas(map_frame, bg="#020617", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Painel Direito: Indicadores, Célula Selecionada e Ranking
    side_frame = tk.Frame(main_container, bg="#0f172a", width=380, padx=12, pady=10, highlightbackground="#1e293b", highlightthickness=1)
    side_frame.pack(side="right", fill="y", padx=(4, 0))
    side_frame.pack_propagate(False)

    lbl_muni_title = tk.Label(side_frame, text="Carregando...", font=("Helvetica", 13, "bold"), bg="#0f172a", fg="#ffffff", anchor="w")
    lbl_muni_title.pack(fill="x", pady=(0, 4))

    lbl_muni_sub = tk.Label(side_frame, text="Aguardando consulta...", font=("Helvetica", 9), bg="#0f172a", fg="#94a3b8", anchor="w")
    lbl_muni_sub.pack(fill="x", pady=(0, 10))

    # Cartões de KPIs
    kpi_frame = tk.Frame(side_frame, bg="#0f172a")
    kpi_frame.pack(fill="x", pady=(0, 10))

    lbl_kpi_risk = tk.Label(kpi_frame, text="Prob. Média: --", font=("Helvetica", 10, "bold"), bg="#1e293b", fg="#fbbf24", padx=8, pady=6, anchor="w")
    lbl_kpi_risk.pack(fill="x", pady=2)

    lbl_kpi_loss = tk.Label(kpi_frame, text="Perda Est.: --", font=("Helvetica", 10, "bold"), bg="#1e293b", fg="#f87171", padx=8, pady=6, anchor="w")
    lbl_kpi_loss.pack(fill="x", pady=2)

    lbl_kpi_crit = tk.Label(kpi_frame, text="Células Críticas: --", font=("Helvetica", 10, "bold"), bg="#1e293b", fg="#ef4444", padx=8, pady=6, anchor="w")
    lbl_kpi_crit.pack(fill="x", pady=2)

    # Detalhes da Célula Inspecionada
    tk.Label(side_frame, text="CÉLULA INSPECIONADA:", font=("Helvetica", 9, "bold"), bg="#0f172a", fg="#38bdf8", anchor="w").pack(fill="x", pady=(10, 2))
    cell_info_txt = tk.Text(side_frame, height=7, bg="#020617", fg="#f1f5f9", font=("Courier", 8), relief="flat", padx=6, pady=6)
    cell_info_txt.pack(fill="x", pady=(0, 6))

    #btn_gmaps = tk.Button(
    #    side_frame, text="📍 Abrir no Google Maps (Navegador)", font=("Helvetica", 9, "bold"),
    #    bg="#0284c7", fg="#ffffff", activebackground="#38bdf8", activeforeground="#0f172a",
    #    padx=8, pady=4, relief="flat", cursor="hand2", command=lambda: open_in_google_maps()
    #)

    btn_gmaps = tk.Button(
        side_frame, text="📍 Abrir no Google Maps (Navegador)", font=("Helvetica", 9, "bold"),
        bg="#0284c7", fg="#ffffff", activebackground="#38bdf8", activeforeground="#0f172a",
        padx=8, pady=4, relief="flat", cursor="hand2", command=lambda: open_in_google_maps()
    )

    btn_gmaps.pack(fill="x", pady=(0, 10))

    # Top Hotspots: ordena pelo mesmo indicador visualizado no mapa.
    lbl_hotspots_title = tk.Label(side_frame, text="TOP 5 POR PERDA PROJETADA:", font=("Helvetica", 9, "bold"), bg="#0f172a", fg="#38bdf8", anchor="w")
    lbl_hotspots_title.pack(fill="x", pady=(4, 2))
    hotspots_txt = tk.Text(side_frame, height=10, bg="#020617", fg="#cbd5e1", font=("Courier", 8), relief="flat", padx=6, pady=6)
    hotspots_txt.pack(fill="both", expand=True)

    def process_query():
        nonlocal current_res
        raw_muni = entry_muni.get().strip() or "MAFRA-SC"
        try:
            ibge = resolve_municipality_code(raw_muni)
        except ValueError as err:
            messagebox.showerror("Município", str(err))
            return
        try:
            pressure = float(entry_pressure.get().strip() or 100) / 100.0
        except ValueError:
            pressure = 1.0

        current_res = run_city_deforestation_analysis(ibge, [1, 3, 5, 8, 10, 20], custom_pressure_multiplier=pressure)
        m = current_res["municipality"]
        lbl_muni_title.config(text=f"{m['name']} - {m['uf']}")
        lbl_muni_sub.config(text=f"IBGE: {m['ibge_code']} | Remanescente: {m['forest_remnant_ha']:,.0f} ha")
        update_map_view()

    def update_map_view():
        if not current_res:
            return
        h = selected_horizon.get()
        summary = current_res["horizon_summaries"].get(h, {})
        cells = current_res.get("cells", [])
        map_metric = selected_map_metric.get()
        # A mesma escala global é usada para todos os horizontes. Não normalizamos
        # separadamente cada ano, pois isso faria mapas temporalmente distintos
        # parecerem iguais em cores.
        loss_scale_max_ha = get_global_loss_scale(cells)
        target_year = summary.get("target_year", "ano-alvo não informado")
        if map_metric == MAP_METRIC_LOSS:
            map_frame.config(text=f" Perda acumulada projetada por célula até {target_year} — horizonte {h[:-1]} anos ")
            lbl_map_hint.config(text="Cores = hectares alocados por célula; escala fixa entre horizontes.")
            lbl_hotspots_title.config(text="TOP 5 POR PERDA PROJETADA:")
        else:
            map_frame.config(text=" Risco espacial por célula — escore do modelo (camada não temporal) ")
            lbl_map_hint.config(text="O escore de risco pode permanecer igual entre anos; use Perda projetada para comparar cenários.")
            lbl_hotspots_title.config(text="TOP 5 POR RISCO ESPACIAL:")

        # Atualiza KPIs
        lbl_kpi_risk.config(text=f"Prob. Média: {summary.get('mean_probability', 0)*100:.1f}% (Pico: {summary.get('max_probability', 0)*100:.1f}%)")
        lbl_kpi_loss.config(text=f"Perda Projetada: {summary.get('expected_suppressed_area_ha', 0):,.1f} ha ({summary.get('percent_of_remnant_lost', 0):.2f}%)")
        lbl_kpi_crit.config(text=f"Células Críticas (P≥72%): {summary.get('critical_points_count', 0)} ({summary.get('high_points_count', 0)} em risco alto)")

        # Atualiza Top Hotspots
        hotspots_txt.delete("1.0", tk.END)
        if map_metric == MAP_METRIC_LOSS:
            # O resumo do backend guarda os 10 maiores escores de risco, não
            # necessariamente as 10 maiores alocações quando há células limitadas.
            hotspot_points = []
            for cell in cells:
                if not cell.get("is_inside_boundary", True):
                    continue
                scenario = (cell.get("scenarios") or {}).get(h, {})
                hotspot_points.append({
                    "locality": cell.get("locality", cell.get("cell_id", "Célula")),
                    "probability": float(scenario.get("probability") or 0.0),
                    "attention_level": scenario.get("attention_level", "BAIXO"),
                    "expected_suppressed_ha": float(scenario.get("expected_suppressed_ha") or 0.0),
                    "dominant_driver": cell.get("dominant_driver", "Não informado"),
                })
            hotspot_points.sort(key=lambda pt: float(pt.get("expected_suppressed_ha") or 0.0), reverse=True)
        else:
            hotspot_points = list(summary.get("top_attention_points", []))
            hotspot_points.sort(key=lambda pt: float(pt.get("probability") or 0.0), reverse=True)
        for rank, pt in enumerate(hotspot_points[:5], start=1):
            hotspots_txt.insert(
                tk.END,
                f"#{rank} {pt['locality']}\n"
                f"   Risco: {pt['probability']*100:.1f}% [{pt['attention_level']}]\n"
                f"   Perda: {pt['expected_suppressed_ha']:.3f} ha | {pt['dominant_driver']}\n\n"
            )

        # Redesenha Canvas — cidade inteira em quadrados, cabendo na área visível
        canvas.delete("all")
        cw = max(canvas.winfo_width() or 640, 320)
        ch = max(canvas.winfo_height() or 540, 280)

        inside = [c for c in cells if c.get("is_inside_boundary", True)]
        if not inside:
            canvas.create_text(cw/2, ch/2, text="Sem células no polígono", fill="#94a3b8")
            return

        min_r = min(c.get("row", 0) for c in inside)
        max_r = max(c.get("row", 0) for c in inside)
        min_c = min(c.get("col", 0) for c in inside)
        max_c = max(c.get("col", 0) for c in inside)
        n_r = max_r - min_r + 1
        n_c = max_c - min_c + 1

        margin_x = 16
        margin_top = 12
        margin_bottom = 50  # espaço para legenda e faixa em hectares
        avail_w = cw - 2 * margin_x
        avail_h = ch - margin_top - margin_bottom
        # células quadradas que cabem inteiras na tela
        side = min(avail_w / n_c, avail_h / n_r)
        grid_w = side * n_c
        grid_h = side * n_r
        origin_x = margin_x + (avail_w - grid_w) / 2
        origin_y = margin_top + (avail_h - grid_h) / 2
        gap = max(1.0, side * 0.04)

        font_size = max(6, min(11, int(side * 0.22)))
        font_id = max(5, min(8, int(side * 0.14)))

        for c in inside:
            sc = c.get("scenarios", {}).get(h, {})
            prob = sc.get("probability", 0.0)
            level = sc.get("attention_level", "BAIXO")
            col = c.get("col", 0)
            row = c.get("row", 0)
            x1 = origin_x + (col - min_c) * side + gap / 2
            y1 = origin_y + (max_r - row) * side + gap / 2
            x2 = x1 + side - gap
            y2 = y1 + side - gap

            color = get_cell_map_color(c, h, map_metric, loss_scale_max_ha)
            tag_name = f"cell_{c['cell_id']}"

            outline_col = "#38bdf8" if selected_cell_id.get() == c["cell_id"] else "#334155"
            outline_w = 2.5 if selected_cell_id.get() == c["cell_id"] else 1
            canvas.create_rectangle(
                x1, y1, x2, y2, fill=color, outline=outline_col, width=outline_w, tags=(tag_name, "cell")
            )

            rgb = tuple(int(color[idx:idx + 2], 16) for idx in (1, 3, 5))
            text_color = "#0f172a" if sum(rgb) >= 400 else "#f8fafc"
            # O rótulo precisa refletir a camada atual: hectares no modo de perda,
            # porcentagem no modo de risco. Só é mostrado para células legíveis.
            if side >= 22:
                if map_metric == MAP_METRIC_LOSS:
                    map_label = f"{float(sc.get('expected_suppressed_ha', 0.0) or 0.0):.1f}"
                else:
                    map_label = f"{prob*100:.0f}%"
                canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text=map_label, fill=text_color,
                    font=("Helvetica", font_size, "bold"), tags=(tag_name, "cell_text"),
                )

            def make_handler(cid):
                return lambda event: inspect_cell(cid)
            canvas.tag_bind(tag_name, "<Button-1>", make_handler(c["cell_id"]))

        # Rosa dos ventos (N)
        nx, ny = 28, 36
        canvas.create_line(nx, ny + 18, nx, ny - 14, fill="#e2e8f0", width=2, arrow="last")
        canvas.create_text(nx, ny - 22, text="N", fill="#e2e8f0", font=("Helvetica", 10, "bold"))

        # Legenda com quinze classes discretas. A perda usa a escala global de
        # todos os horizontes; o risco usa faixas fixas de 12,5 pontos percentuais.
        lx = margin_x
        leg_y = ch - 30
        if map_metric == MAP_METRIC_LOSS:
            canvas.create_text(lx, leg_y, text="Perda (ha)", fill="#cbd5e1", font=("Helvetica", 8, "bold"), anchor="w")
            bar_x = lx + 76
            bar_w = max(70.0, cw - bar_x - margin_x - 8)
            segment_w = bar_w / MAP_COLOR_CLASSES
            for j, fill in enumerate(MAP_COLOR_PALETTE):
                x1 = bar_x + j * segment_w
                x2 = bar_x + (j + 1) * segment_w
                canvas.create_rectangle(x1, leg_y - 6, x2 + 0.5, leg_y + 4, fill=fill, outline="")
                if segment_w >= 39:
                    lo = loss_scale_max_ha * (j / MAP_COLOR_CLASSES) ** 2
                    hi = loss_scale_max_ha * ((j + 1) / MAP_COLOR_CLASSES) ** 2
                    fmt = lambda n: f"{n:,.1f}" if n >= 10 else f"{n:.2g}"
                    label = f"{fmt(lo)}–{fmt(hi)}"
                else:
                    label = str(j + 1)
                canvas.create_text(
                    (x1 + x2) / 2, leg_y + 6, text=label,
                    fill="#94a3b8", font=("Helvetica", 6 if segment_w >= 39 else 7), anchor="n",
                )
        else:
            canvas.create_text(lx, leg_y, text="Risco (%)", fill="#cbd5e1", font=("Helvetica", 8, "bold"), anchor="w")
            bar_x = lx + 76
            bar_w = max(70.0, cw - bar_x - margin_x - 8)
            segment_w = bar_w / MAP_COLOR_CLASSES
            for j, fill in enumerate(MAP_COLOR_PALETTE):
                x1 = bar_x + j * segment_w
                x2 = bar_x + (j + 1) * segment_w
                canvas.create_rectangle(x1, leg_y - 6, x2 + 0.5, leg_y + 4, fill=fill, outline="")
                if segment_w >= 39:
                    lo_pct, hi_pct = j * (100 / MAP_COLOR_CLASSES), (j + 1) * (100 / MAP_COLOR_CLASSES)
                    label = f"{lo_pct:g}–{hi_pct:g}"
                else:
                    label = str(j + 1)
                canvas.create_text(
                    (x1 + x2) / 2, leg_y + 6, text=label,
                    fill="#94a3b8", font=("Helvetica", 6 if segment_w >= 39 else 7), anchor="n",
                )

        # O detalhe selecionado acompanha automaticamente a troca do horizonte.
        selected_id = selected_cell_id.get()
        if selected_id:
            selected_cell = next((item for item in cells if item.get("cell_id") == selected_id), None)
            if selected_cell:
                selected_scenario = (selected_cell.get("scenarios") or {}).get(h, {})
                lat, lon = selected_cell["centroid"]
                cell_info_txt.delete("1.0", tk.END)
                cell_info_txt.insert(
                    tk.END,
                    f"ID: {selected_cell['cell_id']} ({selected_cell['locality']})\n"
                    f"Coordenadas: {lat:.4f}°, {lon:.4f}°\n"
                    f"Risco espacial: {selected_scenario.get('probability', 0)*100:.1f}% [{selected_scenario.get('attention_level', 'BAIXO')}]\n"
                    f"Perda acumulada ({h}, até {selected_scenario.get('target_year', target_year)}): {selected_scenario.get('expected_suppressed_ha', 0):.3f} ha\n"
                    f"Dist. rodovias: {selected_cell.get('dist_road_km', 0):.1f} km\n"
                    f"Floresta inicial estimada: {selected_cell.get('initial_forest_ha', 0):.1f} ha"
                )

    def _cell_google_maps_url(cell: dict) -> str:
        """
        Abre o Google Maps com o retângulo da célula desenhado (rota fechada azul)
        usando os 4 cantos do bbox da grade.
        bbox da célula: [min_lat, min_lon, max_lat, max_lon]
        """
        lat, lon = cell["centroid"]
        bb = cell.get("bbox")
        if bb and len(bb) >= 4:
            min_lat, min_lon, max_lat, max_lon = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            # 4 cantos + fecha o polígono (Google desenha a linha em azul)
            corners = [
                (min_lat, min_lon),
                (min_lat, max_lon),
                (max_lat, max_lon),
                (max_lat, min_lon),
                (min_lat, min_lon),
            ]
            path = "/".join(f"{a:.6f},{b:.6f}" for a, b in corners)
            # zoom ~17 para enquadrar uma célula ~1km
            return (
                f"https://www.google.com/maps/dir/{path}"
                f"/@{lat:.6f},{lon:.6f},17z/data=!3m1!4b1!4m2!4m1!3e2"
            )
        return f"https://www.google.com/maps/@{lat},{lon},17z"

    def open_in_google_maps():
        cid = selected_cell_id.get()
        if not cid or not current_res:
            messagebox.showinfo("Google Maps", "Por favor, clique em um quadrado no mapa para inspecionar.")
            return
        cell = next((c for c in current_res.get("cells", []) if c["cell_id"] == cid), None)
        if not cell:
            return
        #gmaps_url = _cell_google_maps_url(cell)

        # Contexto de ano para a barra de título do mapa embutido
        h_key = selected_horizon.get() or "1y"
        sc = (cell.get("scenarios") or {}).get(h_key) or {}
        target_year = sc.get("target_year")
        base_year = None
        try:
            meta = (current_res.get("model_metadata") or {})
            base_year = meta.get("base_year") or meta.get("panel_year_t0")
            if base_year is None:
                # tenta inferir do horizonte
                hy = int(str(h_key).replace("y", "") or "1")
                if target_year is not None:
                    base_year = int(target_year) - hy
        except Exception:
            base_year = None
        abrir_mapa_satelite_embutido(
            cell,
            horizon_key=h_key,
            target_year=target_year,
            base_year=base_year,
        )
        #try:
        #    webbrowser.open(gmaps_url)
        #except Exception as e:
        #    messagebox.showerror("Erro ao Abrir Navegador", f"Não foi possível abrir o navegador: {e}")

    def inspect_cell(cell_id: str):
        selected_cell_id.set(cell_id)
        update_map_view()
        cell = next((c for c in current_res.get("cells", []) if c["cell_id"] == cell_id), None)
        if not cell:
            return
        h = selected_horizon.get()
        sc = cell.get("scenarios", {}).get(h, {})
        lat, lon = cell["centroid"]
        cell_info_txt.delete("1.0", tk.END)
        cell_info_txt.insert(
            tk.END,
            f"ID: {cell['cell_id']} ({cell['locality']})\n"
            f"Coordenadas: {lat:.4f}°, {lon:.4f}°\n"
            f"Risco ({h}): {sc.get('probability', 0)*100:.1f}% [{sc.get('attention_level', 'BAIXO')}]\n"
            f"Perda acumulada ({h}, até {sc.get('target_year', '-') }): {sc.get('expected_suppressed_ha', 0):.3f} ha\n"
            f"Dist. Rodovias: {cell.get('dist_road_km', 0):.1f} km\n"
            f"Cobertura Florestal: {cell.get('initial_forest_ha', 0):.1f} ha"
        )

    def export_pdf_action():
        if not current_res:
            messagebox.showwarning("Aviso", "Por favor, realize uma consulta antes de gerar o relatório em PDF.")
            return

        m = current_res["municipality"]
        default_filename = f"Relatorio_Supressao_Vegetal_{m['ibge_code']}_{m['name']}.pdf"

        try:
            out_file = filedialog.asksaveasfilename(
                title="Salvar Relatório Técnico Oficial em PDF",
                initialfile=default_filename,
                defaultextension=".pdf",
                filetypes=[("Arquivos PDF (*.pdf)", "*.pdf"), ("Todos os Arquivos", "*.*")]
            )
        except Exception:
            out_file = default_filename

        if not out_file:
            return

        try:
            generate_technical_report_pdf(current_res, out_file)
            messagebox.showinfo(
                "PDF Gerado com Sucesso",
                f"O Relatório Técnico de 7 páginas foi gerado com sucesso!\n\nSalvo em:\n{out_file}"
            )
        except Exception as err:
            messagebox.showerror("Erro ao Gerar PDF", f"Falha na compilação do relatório: {err}")

    # Bind de redimensionamento do canvas
    canvas.bind("<Configure>", lambda e: update_map_view())

    # Inicializa com a cidade padrão
    root.after(100, process_query)
    root.mainloop()

def abrir_mapa_satelite_embutido(
    cell: dict,
    horizon_key=None,
    target_year=None,
    base_year=None,
):
    """
    Abre janela maximizada com mapa de satélite do Google e o polígono da célula.
    A barra de título indica o ano-base do painel e o horizonte analisado.
    """
    import tkinter as tk
    import tkintermapview

    lat, lon = cell["centroid"]
    bb = cell.get("bbox")
    cid = cell.get("cell_id") or "—"

    # Texto do ano para a barra de título
    if base_year is not None:
        try:
            by = int(base_year)
        except (TypeError, ValueError):
            by = base_year
        year_txt = f"ano-base do painel: {by}"
    else:
        year_txt = "ano-base do painel: n/d"
    if target_year is not None:
        try:
            ty = int(target_year)
        except (TypeError, ValueError):
            ty = target_year
        year_txt += f" · horizonte até {ty}"
    if horizon_key:
        year_txt += f" ({horizon_key})"
    # Nota honesta: tiles Google não informam data de captura

    janela_mapa = tk.Toplevel()
    janela_mapa.title(f"ForestGuard · Célula {cid} · {year_txt}")
    janela_mapa.geometry("1024x768")
    try:
        janela_mapa.state("zoomed")
    except Exception:
        try:
            janela_mapa.attributes("-zoomed", True)
        except Exception:
            try:
                janela_mapa.update_idletasks()
                sw, sh = janela_mapa.winfo_screenwidth(), janela_mapa.winfo_screenheight()
                janela_mapa.geometry(f"{sw}x{sh}+0+0")
            except Exception:
                pass
    try:
        janela_mapa.grab_set()
    except Exception:
        pass

    mapa = tkintermapview.TkinterMapView(janela_mapa, corner_radius=0)
    mapa.pack(fill="both", expand=True)

    mapa.set_tile_server("https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", max_zoom=22)
    mapa.set_position(lat, lon)
    mapa.set_zoom(17)
    mapa.set_marker(lat, lon, text="Centro do alerta")

    if bb and len(bb) >= 4:
        min_lat, min_lon, max_lat, max_lon = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        coordenadas_quadrado = [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon),
            (min_lat, min_lon),
        ]
        mapa.set_polygon(
            coordenadas_quadrado,
            fill_color=None,
            outline_color="#EF4444",
            border_width=3,
            name="bbox_alerta",
        )

def _run_terminal_interactive_gui():
    print("=" * 78)
    print("APLICAÇÃO DESKTOP SUPRESSÃO VEGETAL ML — CONSOLE FALLBACK COM PDF")
    print("Horizontes: 3, 5, 8 e 10 Anos | Grade Regular SIRGAS 2000")
    print("=" * 78)

    muni_input = "MAFRA-SC"
    ibge = resolve_municipality_code(muni_input)
    res = run_city_deforestation_analysis(ibge_code=ibge, horizons=[1, 3, 5, 8, 10])
    m = res["municipality"]

    print(f"\n[Consulta Oficial: {muni_input} -> {m['name']}/{m['uf']} (IBGE {m['ibge_code']})]")
    print(f"Bioma: {m['biome']} | Remanescente PRODES: {m['forest_remnant_ha']:,.1f} ha\n")

    for h in [1, 3, 5, 8, 10]:
        h_key = f"{h}y"
        s = res["horizon_summaries"][h_key]
        print(f"► Cenário {h} Anos ({s['target_year']}):")
        print(f"   Risco Médio: {s['mean_probability']*100:.1f}% | Críticos: {s['critical_points_count']} | Perda: {s['expected_suppressed_area_ha']:.1f} ha ({s['percent_of_remnant_lost']:.2f}%)")
        top1 = s["top_attention_points"][0]
        print(f"   Hotspot #1: {top1['locality']} ({top1['probability']*100:.1f}% risco - {top1['dominant_driver']})\n")

    # Gera PDF automaticamente no console fallback
    pdf_out = f"Relatorio_Tecnico_Oficial_{ibge}_{m['name']}.pdf"
    generate_technical_report_pdf(res, pdf_out)
    print(f"🎉 Relatório Técnico em PDF (7 páginas oficiais) gerado com sucesso:")
    print(f"   Arquivo: {os.path.abspath(pdf_out)} ({os.path.getsize(pdf_out):,} bytes)")
    print("=" * 78)
