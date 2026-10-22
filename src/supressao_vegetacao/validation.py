"""
Validação de contratos de entrada e horizontes temporais.
Aceita horizontes entre 1 e 20 anos (limite rígido da API v25).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .settings import (
    MAX_HORIZON_YEARS,
    MIN_HORIZON_YEARS,
    RECOMMENDED_HORIZONS,
    SUPPORTED_HORIZONS,
)


class ValidationError(ValueError):
    """Erro de validação de entrada do sistema."""


def validate_horizons(horizons: Optional[Sequence[int]] = None) -> List[int]:
    """
    Valida horizontes de projeção.

    - Se None/vazio → retorna os recomendados (3, 5, 8, 10, 20).
    - Aceita qualquer inteiro em [MIN_HORIZON_YEARS, MAX_HORIZON_YEARS]
      (ex.: 25, 28), além dos recomendados.
    - Remove duplicatas e ordena.
    """
    if not horizons:
        return list(RECOMMENDED_HORIZONS)

    cleaned: List[int] = []
    invalid: List[object] = []
    for h in horizons:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            invalid.append(h)
            continue
        if hi < MIN_HORIZON_YEARS or hi > MAX_HORIZON_YEARS:
            invalid.append(hi)
            continue
        if hi not in cleaned:
            cleaned.append(hi)

    if invalid:
        raise ValidationError(
            f"Horizontes inválidos detectados: {invalid}. "
            f"Use inteiros entre {MIN_HORIZON_YEARS} e {MAX_HORIZON_YEARS} anos "
            f"(recomendados: {list(RECOMMENDED_HORIZONS)})."
        )
    if not cleaned:
        raise ValidationError("Nenhum horizonte válido informado.")

    cleaned.sort()
    return cleaned


def validate_ibge_code(ibge_code: str) -> str:
    """Garante código IBGE municipal com 7 dígitos."""
    code = "".join(ch for ch in str(ibge_code or "") if ch.isdigit())
    if len(code) != 7:
        raise ValidationError(
            f"Código IBGE inválido: '{ibge_code}'. Esperado 7 dígitos (ex.: 4210100)."
        )
    return code
