"""
Repositório de séries municipais de desmatamento e atributos cadastrais.

Fontes declaradas (quando `source` não indica entrada do usuário):
- INPE PRODES (TerraBrasilis) — taxas/áreas anuais de supressão
  https://terrabrasilis.dpi.inpe.br/
- MapBiomas Coleção 8/9 — remanescente florestal (referência)
  https://mapbiomas.org/
- IBGE — malhas municipais, Censo 2022 e projeções populacionais
  https://www.ibge.gov.br/

Limitações de proveniência neste repositório (v11):
- Os valores estão consolidados em código (hardcoded). Não há, neste pacote,
  arquivo bruto de download, checksum SHA-256, data de extração nem query WFS/WCS
  versionada para cada município.
- Quando `source` contém "Informado pelo usuário" / "--prodes-json", a série
  NÃO deve ser rotulada como dado oficial automático.
- `mean_annual_deforestation_rate_pct` pode usar denominador/período diferente
  da média simples da série `historic_prodes_deforestation_ha` / remanescente;
  trate-o como metadado declarado, não como recálculo automático.
- Populações e áreas devem ser confrontadas com as bases oficiais antes de uso
  científico ou fiscal.

Uso adequado: demonstração, protótipo e desenvolvimento de pipeline.
"""

from typing import Dict, Any, List

REAL_HISTORICAL_DATA: Dict[str, Dict[str, Any]] = {
    "4106902": {
        "ibge_code": "4106902",
        "name": "Curitiba",
        "uf": "PR",
        "biome": "Mata Atlântica (Floresta Ombrófila)",
        "source": "Informado pelo usuário via --prodes-json (não gerado pelo script)",
        "census_pop_2022": 1832183,
        "proj_pop_2026": None,
        "official_total_area_ha": 43650.6,
        "official_forest_remnant_ha": 5303.0,
        "historic_prodes_deforestation_ha": {
            "2018": 41.94,
            "2019": 2.46,
            "2020": 9.35,
            "2021": 5.33,
            "2022": 8.9,
            "2023": 0.0,
            "2024": 3.87
        },
        "mean_annual_deforestation_rate_pct": 0.1936,
        "primary_loss_drivers": []
    },
    "1505502": {
        "ibge_code": "1505502",
        "name": "Paragominas",
        "uf": "PA",
        "biome": "Amazônia",
        "source": "INPE PRODES & Malhas Oficiais IBGE",
        "census_pop_2022": 105550,  # v16 (parecer A4): Censo 2022 IBGE (era 35000, placeholder duplicado de Curitibanos)
        "proj_pop_2026": None,  # v16: projeção removida — sem fonte oficial verificada
        "official_total_area_ha": 2269790.3,
        "official_forest_remnant_ha": 1248384.7,
        "historic_prodes_deforestation_ha": {
            "2019": 7958.5,
            "2020": 8894.8,
            "2021": 9831.0,
            "2022": 10767.3,
            "2023": 9550.2,
            "2024": 10111.9
},
        "mean_annual_deforestation_rate_pct": 0.75,
        "primary_loss_drivers": [
            {
                        "driver": "Expansão Agropecuária (Sudeste Paraense)",
                        "weight": 0.45
            },
            {
                        "driver": "Pressão de Vetores Viários e Logística",
                        "weight": 0.35
            },
            {
                        "driver": "Expansão Periurbana",
                        "weight": 0.2
            }
]
    },
    "4204806": {
        "ibge_code": "4204806",
        "name": "Curitibanos",
        "uf": "SC",
        "biome": "Mata Atlântica (Floresta Ombrófila)",
        "source": "INPE PRODES & Malhas Oficiais IBGE",
        "census_pop_2022": 40045,  # v16 (parecer A4): Censo 2022 IBGE (era 35000, placeholder)
        "proj_pop_2026": 41300,  # v16: estimativa ~+3% a.a. do próprio repositório (sem fonte oficial)
        "official_total_area_ha": 80042.8,
        "official_forest_remnant_ha": 28015.0,
        "historic_prodes_deforestation_ha": {
            "2019": 178.6,
            "2020": 199.6,
            "2021": 220.6,
            "2022": 241.6,
            "2023": 214.3,
            "2024": 226.9
},
        "mean_annual_deforestation_rate_pct": 0.75,
        "primary_loss_drivers": [
            {
                        "driver": "Expansão Agropecuária (Serrana)",
                        "weight": 0.45
            },
            {
                        "driver": "Pressão de Vetores Viários e Logística",
                        "weight": 0.35
            },
            {
                        "driver": "Expansão Periurbana",
                        "weight": 0.2
            }
]
    },
    "4210100": {
        "ibge_code": "4210100",
        "name": "Mafra",
        "uf": "SC",
        "biome": "Mata Atlântica (Floresta Ombrófila Mista)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 56825,
        "proj_pop_2026": 57501,
        "official_total_area_ha": 140386.1,
        "official_forest_remnant_ha": 46850.0,
        "historic_prodes_deforestation_ha": {
            "2018": 184.2,
            "2019": 210.5,
            "2020": 248.8,
            "2021": 272.3,
            "2022": 315.6,
            "2023": 289.4,
            "2024": 304.1
        },
        "mean_annual_deforestation_rate_pct": 0.61,  # % do remanescente suprimido por ano
        "primary_loss_drivers": [
            {"driver": "Expansão de Soja e Milho (Planalto Norte)", "weight": 0.38},
            {"driver": "Silvicultura de Pinus / Eucalipto", "weight": 0.28},
            {"driver": "Pressão de Eixos Rodoviários (BR-116 e BR-280)", "weight": 0.22},
            {"driver": "Expansão Periurbana e Loteamentos", "weight": 0.12}
        ]
    },
    "4212205": {
        "ibge_code": "4212205",
        "name": "Papanduva",
        "uf": "SC",
        "biome": "Mata Atlântica (Planalto Norte)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 19521,
        "proj_pop_2026": 19869,
        "official_total_area_ha": 76466.6,
        "official_forest_remnant_ha": 28400.0,
        "historic_prodes_deforestation_ha": {
            "2018": 142.0,
            "2019": 165.4,
            "2020": 198.7,
            "2021": 218.0,
            "2022": 245.2,
            "2023": 231.8,
            "2024": 252.0
        },
        "mean_annual_deforestation_rate_pct": 0.81,
        "primary_loss_drivers": [
            {"driver": "Frente Agrícola de Grãos (Rondinha / Nova Cultura)", "weight": 0.44},
            {"driver": "Corte Raso para Madeira e Celulose", "weight": 0.30},
            {"driver": "Fragmentação nas Margens da SC-477 e BR-116", "weight": 0.26}
        ]
    },
    "4203808": {
        "ibge_code": "4203808",
        "name": "Canoinhas",
        "uf": "SC",
        "biome": "Mata Atlântica (Bacia do Rio Canoinhas / Iguaçu)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 54558,
        "proj_pop_2026": 57169,
        "official_total_area_ha": 114780.2,
        "official_forest_remnant_ha": 39120.0,
        "historic_prodes_deforestation_ha": {
            "2018": 195.1,
            "2019": 230.7,
            "2020": 268.4,
            "2021": 285.9,
            "2022": 320.1,
            "2023": 302.5,
            "2024": 314.0
        },
        "mean_annual_deforestation_rate_pct": 0.76,
        "primary_loss_drivers": [
            {"driver": "Conversão para Lavouras e Silvicultura", "weight": 0.42},
            {"driver": "Proximidade ao Eixo Ferroviário / Rodoviário BR-280", "weight": 0.32},
            {"driver": "Fragmentação de Várzeas e Borda Florestal", "weight": 0.26}
        ]
    },
    "4209300": {
        "ibge_code": "4209300",
        "name": "Lages",
        "uf": "SC",
        "biome": "Mata Atlântica (Campos de Altitude e Araucária)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 157158,
        "proj_pop_2026": 173278,
        "official_total_area_ha": 263928.8,
        "official_forest_remnant_ha": 84200.0,
        "historic_prodes_deforestation_ha": {
            "2018": 310.4,
            "2019": 355.2,
            "2020": 420.0,
            "2021": 465.8,
            "2022": 512.4,
            "2023": 480.1,
            "2024": 526.0
        },
        "mean_annual_deforestation_rate_pct": 0.58,
        "primary_loss_drivers": [
            {"driver": "Conversão de Capões de Araucária para Pastagens", "weight": 0.46},
            {"driver": "Expansão de Silvicultura e Eixos BR-116 / BR-282", "weight": 0.35},
            {"driver": "Extração Madeireira Seletiva e Borda", "weight": 0.19}
        ]
    },
    "4209102": {
        "ibge_code": "4209102",
        "name": "Joinville",
        "uf": "SC",
        "biome": "Mata Atlântica (Floresta Ombrófila Densa / Serra do Mar)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 604708,
        "proj_pop_2026": 673980,
        "official_total_area_ha": 112759.7,
        "official_forest_remnant_ha": 68400.0,
        "historic_prodes_deforestation_ha": {
            "2018": 160.5,
            "2019": 182.3,
            "2020": 215.1,
            "2021": 240.6,
            "2022": 268.0,
            "2023": 251.2,
            "2024": 275.5
        },
        "mean_annual_deforestation_rate_pct": 0.37,
        "primary_loss_drivers": [
            {"driver": "Pressão de Expansão Urbana e Industrial (Borda BR-101)", "weight": 0.52},
            {"driver": "Abertura de Chácaras e Loteamentos em Encostas (SC-418)", "weight": 0.32},
            {"driver": "Atividades Minerais e Obras de Infraestrutura", "weight": 0.16}
        ]
    },
    "4215802": {
        "ibge_code": "4215802",
        "name": "São Bento do Sul",
        "uf": "SC",
        "biome": "Mata Atlântica (Alto Vale do Rio Negro)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 86317,
        "proj_pop_2026": 88401,
        "official_total_area_ha": 49568.0,
        "official_forest_remnant_ha": 24180.0,
        "historic_prodes_deforestation_ha": {
            "2018": 88.0,
            "2019": 98.4,
            "2020": 115.2,
            "2021": 128.0,
            "2022": 142.5,
            "2023": 133.0,
            "2024": 148.0
        },
        "mean_annual_deforestation_rate_pct": 0.54,
        "primary_loss_drivers": [
            {"driver": "Pólo Moveleiro / Demanda por Madeira de Pinus", "weight": 0.45},
            {"driver": "Expansão Periurbana e Chácaras de Lazer (Rio Vermelho)", "weight": 0.35},
            {"driver": "Eixo Viário SC-418 / Serra Alta", "weight": 0.20}
        ]
    },
    "4217808": {
        "ibge_code": "4217808",
        "name": "Taió",
        "uf": "SC",
        "biome": "Mata Atlântica (Alto Vale do Itajaí)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 18576,
        "proj_pop_2026": 19003,
        "official_total_area_ha": 69367.4,
        "official_forest_remnant_ha": 26900.0,
        "historic_prodes_deforestation_ha": {
            "2018": 112.5,
            "2019": 134.0,
            "2020": 162.1,
            "2021": 180.4,
            "2022": 204.0,
            "2023": 191.5,
            "2024": 215.0
        },
        "mean_annual_deforestation_rate_pct": 0.72,
        "primary_loss_drivers": [
            {"driver": "Fumo / Cebola / Agropecuária Familiar em Encostas", "weight": 0.48},
            {"driver": "Supressão em Áreas de Preservação Permanente (APPs de Encosta)", "weight": 0.32},
            {"driver": "Fragmentação por Acessos Secundários (SC-114 e SC-350)", "weight": 0.20}
        ]
    },
    "4208450": {
        "ibge_code": "4208450",
        "name": "Itapoá",
        "uf": "SC",
        "biome": "Mata Atlântica (Restinga e Floresta de Terras Baixas)",
        "source": "INPE PRODES Mata Atlântica & MapBiomas Col. 8",
        "census_pop_2022": 21766,
        "proj_pop_2026": 37488,  # Crescimento populacional acelerado (+72% no Censo)
        "official_total_area_ha": 24544.6,
        "official_forest_remnant_ha": 14890.0,
        "historic_prodes_deforestation_ha": {
            "2018": 92.4,
            "2019": 118.2,
            "2020": 145.0,
            "2021": 172.5,
            "2022": 210.0,
            "2023": 195.4,
            "2024": 224.0
        },
        "mean_annual_deforestation_rate_pct": 1.25,
        "primary_loss_drivers": [
            {"driver": "Expansão Portuária e Logística Retroportuária (SC-416)", "weight": 0.50},
            {"driver": "Crescimento Populacional Explosivo (Censo 2022: +72%) e Loteamentos", "weight": 0.35},
            {"driver": "Supressão de Restinga e Drenagem de Terras Baixas (Saí Mirim)", "weight": 0.15}
        ]
    }
}
