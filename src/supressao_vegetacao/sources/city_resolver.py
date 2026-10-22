"""
Resolução de município (nome/UF → código IBGE), independente de GUI.

Movido de desktop/app.py na v16 (parecer C3): a API não pode depender de
tkinter/tkintermapview para resolver nomes de municípios em servidores headless.
"""

import unicodedata
from typing import List

from .municipality import get_municipality, list_municipalities


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def resolve_municipality_code(user_input: str) -> str:
    """
    Resolve código IBGE a partir de:
      - 7 dígitos (4217808)
      - NOME-UF / NOME/UF (TAIO-SC, Taió-SC, Taió/SC)
      - nome isolado (Taió), se único no catálogo

    Comparação sem acentos e sem cair silenciosamente em outro município.
    """
    raw = (user_input or "").strip()
    if not raw:
        raise ValueError("Informe o município (ex: TAIO-SC ou 4217808).")

    # Código IBGE direto
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 7:
        m = get_municipality(digits)
        if m:
            return digits
        raise ValueError(f"Código IBGE {digits} não está no catálogo.")

    cleaned = strip_accents(raw).upper().replace(" ", "").replace("_", "-").replace("/", "-")
    cities = list_municipalities()

    def norm_city_name(name: str) -> str:
        return strip_accents(name).upper().replace(" ", "").replace("-", "")

    # 1) Match exato NOME-UF
    exact: List[dict] = []
    partial: List[dict] = []
    for m in cities:
        n = norm_city_name(m["name"])
        uf = (m.get("uf") or "").upper()
        key = f"{n}-{uf}"
        if cleaned == key or cleaned == n:
            exact.append(m)
        elif n in cleaned or cleaned in n or cleaned.startswith(n + "-"):
            # evita falso positivo curto (ex: "RIO" em vários nomes)
            if len(n) >= 4 or cleaned == n or cleaned.startswith(n + "-"):
                partial.append(m)

    if len(exact) == 1:
        return exact[0]["ibge_code"]
    if len(exact) > 1:
        # desempate por UF no input
        if "-" in cleaned:
            uf_in = cleaned.rsplit("-", 1)[-1]
            for m in exact:
                if (m.get("uf") or "").upper() == uf_in:
                    return m["ibge_code"]
        names = ", ".join(f"{m['name']}/{m['uf']}" for m in exact)
        raise ValueError(f"Vários municípios para '{raw}': {names}. Use NOME-UF.")

    if len(partial) == 1:
        return partial[0]["ibge_code"]
    if len(partial) > 1:
        names = ", ".join(f"{m['name']}/{m['uf']} ({m['ibge_code']})" for m in partial)
        raise ValueError(f"Ambíguo '{raw}': {names}. Use o código IBGE ou NOME-UF.")

    available = ", ".join(f"{m['name']}-{m['uf']}" for m in cities[:12])
    raise ValueError(
        f"Município '{raw}' não encontrado no catálogo. "
        f"Exemplos: {available}..."
    )
