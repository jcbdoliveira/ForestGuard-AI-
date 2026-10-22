"""
Configurações globais — ForestGuard v4.1
Malha densa no polígono + horizontes livres (recomendados e estendidos).
"""

from dataclasses import dataclass
from typing import Tuple


# Horizontes recomendados (UI / defaults)
RECOMMENDED_HORIZONS: Tuple[int, ...] = (1, 3, 5, 8, 10, 20)

# Alias legado
SUPPORTED_HORIZONS: Tuple[int, ...] = RECOMMENDED_HORIZONS

# Limites rígidos da API v26: previsão 1–20 anos (acima disso → erro)
MIN_HORIZON_YEARS: int = 1
MAX_HORIZON_YEARS: int = 20

# Conjunto completo pré-carregado em memória no startup
PRELOAD_HORIZONS: Tuple[int, ...] = tuple(range(1, 21))  # 1..20


@dataclass(frozen=True)
class ModelSettings:
    model_version: str = "v27.1.0"
    supported_horizons: Tuple[int, ...] = RECOMMENDED_HORIZONS
    recommended_horizons: Tuple[int, ...] = RECOMMENDED_HORIZONS
    min_horizon_years: int = MIN_HORIZON_YEARS
    max_horizon_years: int = MAX_HORIZON_YEARS
    crs_storage: str = "EPSG:4674"
    crs_metric: str = "EPSG:31982"
    # Limiares do mapa polígono (0.25 / 0.50 / 0.75)
    major_threshold: float = 0.75
    high_threshold: float = 0.50
    moderate_threshold: float = 0.25
    critical_threshold: float = 0.75
    default_target_bbox_cells: int = 3600
    min_grid_side: int = 20
    max_grid_side: int = 100
    # Escala de decaimento da proximidade rodoviária (km). Parâmetro de engenharia
    # calibrado empiricamente — NÃO estimado pelo treino XGBoost atual.
    road_proximity_decay_km: float = 4.2
    # Fator heurístico de faixa de sensibilidade (NÃO é intervalo de confiança estatístico)
    sensitivity_band_factor: float = 0.25


settings = ModelSettings()
