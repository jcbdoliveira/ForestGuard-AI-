"""
Dynamic Neighborhood Feedback (v23)

Recalcula forest_neighbor_frac a cada ano da simulação anual e ajusta
os pesos de risco pelo efeito de borda.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_knn_neighbors(centroids: List[Tuple[float, float]], k: int = 8) -> List[List[int]]:
    """Constrói lista de índices dos k vizinhos mais próximos para cada célula (exclui self)."""
    n = len(centroids)
    if n == 0:
        return []
    coords = np.asarray(centroids, dtype=np.float64)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    kk = min(max(1, int(k)), max(1, n - 1))
    for i in range(n):
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        d[i] = np.inf
        if n == 1:
            neighbors[i] = []
            continue
        idx = np.argpartition(d, kk - 1)[:kk]
        idx = idx[np.argsort(d[idx])]
        neighbors[i] = [int(j) for j in idx if np.isfinite(d[j])]
    return neighbors


def recompute_weights_dynamic(
    base_weights: Dict[int, float],
    remaining: Dict[int, float],
    neighbors: List[List[int]],
    beta: float = 1.0,
    forest_threshold_ha: float = 1e-6,
) -> Dict[int, float]:
    """
    Loop de Feedback Dinâmico de Vizinhança (v23).

    Após desmatar células no ano t, recalcula forest_neighbor_frac como a
    fração de vizinhos que ainda possuem floresta remanescente > threshold.
    O peso de risco é então ajustado:

        weight_t+1 = base_weight * (1 + beta * (1 - forest_neighbor_frac))
    """
    new_weights: Dict[int, float] = {}
    for i, base in base_weights.items():
        neigh = neighbors[i] if i < len(neighbors) else []
        if not neigh:
            new_weights[i] = max(0.0, float(base))
            continue
        forested = sum(1 for j in neigh if remaining.get(j, 0.0) > forest_threshold_ha)
        frac = forested / float(len(neigh))
        factor = 1.0 + float(beta) * (1.0 - frac)
        new_weights[i] = max(0.0, float(base) * factor)
    return new_weights
