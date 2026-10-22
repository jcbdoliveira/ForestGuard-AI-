"""Distribuição auditável de perda municipal em células espaciais.

A função distribui hectares por pesos de risco relativo, respeitando o limite de
floresta disponível por célula. Ela não cria observações raster: as capacidades
recebidas são tão confiáveis quanto a fonte que as forneceu.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


def allocate_weighted_loss(
    requested_ha: float,
    weights: Iterable[float],
    capacities_ha: Iterable[float],
) -> Dict[str, Any]:
    """Aloca perda proporcional aos pesos, redistribuindo sobras após atingir tetos.

    Args:
        requested_ha: perda total solicitada no horizonte, em hectares.
        weights: escores não negativos (não precisam somar 1); não são hectares.
        capacities_ha: área de floresta disponível em cada célula, em hectares.

    Retorna as alocações na mesma ordem das entradas e um balanço auditável.
    Se o pedido superar a capacidade total, o restante é reportado como não alocado.
    """
    requested = float(requested_ha)
    w = [float(x) for x in weights]
    caps = [float(x) for x in capacities_ha]
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("requested_ha deve ser um número finito não negativo")
    if len(w) != len(caps):
        raise ValueError("weights e capacities_ha devem ter o mesmo comprimento")
    if any(not math.isfinite(x) or x < 0 for x in w):
        raise ValueError("pesos devem ser finitos e não negativos")
    if any(not math.isfinite(x) or x < 0 for x in caps):
        raise ValueError("capacidades devem ser finitas e não negativas")

    allocations = [0.0] * len(caps)
    available = sum(caps)
    target = min(requested, available)
    remaining = target
    active = {i for i, cap in enumerate(caps) if cap > 1e-12}
    capped_cells = set()

    while remaining > 1e-9 and active:
        active_weight_sum = sum(w[i] for i in active)
        # Pesos nulos não podem travar a alocação: nesse caso divide-se igualmente
        # apenas entre as células que ainda têm capacidade.
        if active_weight_sum <= 1e-15:
            shares = {i: remaining / len(active) for i in active}
        else:
            shares = {i: remaining * w[i] / active_weight_sum for i in active}

        saturated = []
        for i in active:
            room = caps[i] - allocations[i]
            if shares[i] >= room - 1e-12:
                saturated.append(i)

        if not saturated:
            for i in active:
                allocations[i] += shares[i]
            remaining = 0.0
            break

        # Preenche primeiro todas as células que atingiriam o teto nesta rodada.
        for i in saturated:
            room = max(0.0, caps[i] - allocations[i])
            allocations[i] += room
            remaining -= room
            active.remove(i)
            capped_cells.add(i)
        remaining = max(0.0, remaining)

    allocated = sum(allocations)
    # Corrige um eventual resíduo numérico sem ultrapassar capacidade de nenhuma célula.
    residual = target - allocated
    if residual > 1e-8:
        for i in range(len(allocations)):
            room = caps[i] - allocations[i]
            delta = min(room, residual)
            if delta > 0:
                allocations[i] += delta
                residual -= delta
            if residual <= 1e-8:
                break
    allocated = sum(allocations)
    unallocated = max(0.0, requested - allocated)

    return {
        "allocations_ha": allocations,
        "requested_ha": requested,
        "allocated_ha": allocated,
        "unallocated_ha": unallocated,
        "available_forest_ha": available,
        "capped_cells_count": len(capped_cells),
        "within_capacity": all(a <= cap + 1e-8 for a, cap in zip(allocations, caps)),
        "balanced": abs(allocated + unallocated - requested) <= 1e-6,
    }
