<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="public/logo_sctecB.png">
    <source media="(prefers-color-scheme: light)" srcset="public/logo_sctecW.png">
    <img alt="Logo SCTEC" src="public/logo_sctecW.png" width="200">
  </picture>
</p>

# ForestGuard AI v27.1

Sistema de apoio à decisão para estimar o risco relativo de supressão de vegetação em horizonte anual, combinando previsão de ML com Regressão Ridge + XGBoost, API REST em FastAPI e interface desktop.

> Aviso importante: os escores produzidos pelo sistema são de risco relativo e priorização territorial. Eles não representam desmatamento consumado, auto de infração, nem probabilidade calibrada para uso jurídico isolado.

---

## Visão geral

O ForestGuard foi pensado para apoiar órgãos ambientais, equipes técnicas e gestores públicos na priorização de áreas para investigação e fiscalização, quando a capacidade operacional é limitada.

O projeto combina:

- painel espacial por célula (dados municipais e de vizinhança);
- predição de ML com Regressão Ridge + XGBoost para priorização de risco e projeções temporais;
- classificador anual para `y_loss_1y` (perda de floresta de t0 para t0+1), com XGBoost como modelo principal;
- regressão Ridge para projeções temporais e avaliação de confiabilidade em séries históricas (`projection_core.py` e `reliability.py`);
- API de inferência com baixa latência e modelo em memória;
- geração de relatório PDF e dashboard para comunicação técnica;
- interface desktop para uso local e demonstração operacional.

Essa combinação é uma prática bastante desafiadora na área de ML, porque integra modelos com diferentes comportamentos, requisitos de regularização e objetivos de previsão em uma mesma solução analítica.

Município de referência no pacote: Mafra/SC — IBGE `4210100`.

---

## Problema de negócio

Em cenários de fiscalização e monitoramento territorial, nem sempre é possível inspecionar todo o território. O desafio é decidir onde olhar primeiro.

O ForestGuard busca responder essa pergunta por meio de um ranking espacial de risco relativo, permitindo focar esforço humano e recursos em áreas com maior chance de perda de cobertura vegetal no horizonte de 1 ano.

---

## Origem do modelo

Este projeto foi desenvolvido dentro do contexto do edital SCTEC como solução de análise preditiva espacial, com pipeline próprio e validação controlada para evitar vazamento de futuro.

O modelo principal para priorização espacial é um XGBoost (`binary:logistic`) aplicado ao alvo `y_loss_1y`. Em paralelo, o projeto também incorpora uma Regressão Ridge em componentes de projeção temporal e validação de confiabilidade, com foco em séries históricas municipais e benchmarking de desempenho.

| Item | Detalhe |
|------|---------|
| Alvo principal | `y_loss_1y` (variável binária de perda florestal em 1 ano) |
| Algoritmo principal | XGBoost (`binary:logistic`) |
| Modelo complementar | Regressão Ridge (regularização L2) em `projection_core.py` e `reliability.py` |
| Dados | Painel espacial municipal com features de vizinhança, relevo, pressão antrópica e atributos territoriais |
| Validação | Holdout temporal + walk-forward com embargo |
| Artefato | `data/spatial/models/4210100_annual/spatial_model.json` + `spatial_model_meta.json` |

Resumo do desempenho em holdout 2020:

| Métrica | Valor |
|---------|------:|
| ROC-AUC | ~0,85 |
| PR-AUC | ~0,59 |
| Recall@top 10% | ~0,64 |

Esses valores indicam bom poder de ordenação do risco, e não uma probabilidade calibrada para previsão pontual ou uso jurídico isolado.

---

## Stack tecnológica

| Camada | Tecnologias |
|--------|-------------|
| API | FastAPI, Uvicorn, Pydantic v2 |
| ML | XGBoost, scikit-learn, NumPy, Regressão Ridge |
| Relatórios | ReportLab, Pillow |
| Geoprocessamento | Shapely, PyProj |
| Desktop | Tkinter, tkintermapview |
| Empacotamento | `requirements.txt`, caminhos relativos |

Arquitetura da API:

```text
src/supressao_vegetacao/api/
  main.py        → ponto de entrada e lifespan
  schemas.py     → contratos Pydantic
  services.py    → carregamento do modelo e inferência
  routers.py     → APIRouter (/health, /predict)
  app.py         → FastAPI, rotas de negócio, cache e dashboard
  model_cache.py → modelo e análises em memória
```

---

## Requisitos

- Python 3.10+ (recomendado 3.11/3.12)
- Ambiente virtual recomendado
- Dependências listadas em `requirements.txt`

---

## Instalação e execução

### 1) Criar o ambiente virtual

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Definir o `PYTHONPATH`

```bash
# Linux / macOS
export PYTHONPATH=src

# Windows (PowerShell)
$env:PYTHONPATH="src"
```

### 3) Treinar ou atualizar o modelo espacial (opcional)

```bash
python scripts/train_spatial.py \
  --panel data/spatial/panels/4210100 \
  --target y_loss_1y \
  --model-dir data/spatial/models/4210100_annual
```

Artefatos esperados:

- `data/spatial/models/4210100_annual/spatial_model.json`
- `data/spatial/models/4210100_annual/spatial_model_meta.json`

### 4) Iniciar a API

```bash
python scripts/run_api.py
```

Também é possível executar manualmente com Uvicorn:

```bash
uvicorn supressao_vegetacao.api.main:app --host 0.0.0.0 --port 8080
```

Endpoints úteis:

- Swagger: http://127.0.0.1:8080/docs
- Health: http://127.0.0.1:8080/health

No startup, o modelo e as análises dos horizontes 1–20 podem ser pré-carregados em memória, dependendo dos artefatos disponíveis.

### 5) Interface desktop (opcional)

```bash
python scripts/run_desktop.py
```

---

## Documentação da API

### Endpoints principais

| Método | Rota | Função |
|--------|------|--------|
| `GET` | `/health` | Verifica saúde do serviço e se o modelo está em RAM |
| `POST` | `/v1/predict` | Inferência espacial em ranking de células |
| `POST` | `/predict` | Alias do endpoint principal exigido pelo edital |
| `GET` | `/v1/analyze/{ibge}/pdf` | Gera relatório PDF |
| `POST` | `/v1/simulate/annual` | Simulação agregada anual |
| `GET` | `/v1/dashboard/{ibge}` | Retorna payload do dashboard |
| `GET` | `/v1/dashboard/{ibge}/view` | Renderiza dashboard em HTML |

Documentação interativa: http://127.0.0.1:8080/docs

### Exemplo: `POST /v1/predict`

Request:

```http
POST /v1/predict
Content-Type: application/json
```

```json
{
  "ibge_code": "4210100",
  "years": 1,
  "top_n": 20,
  "target": "y_loss_1y"
}
```

Campos principais:

| Campo | Tipo | Restrições |
|-------|------|------------|
| `ibge_code` | string | 7 dígitos (`4210100`) |
| `years` | int | 1–20 |
| `top_n` | int | 1–500 |
| `target` | string opcional | `y_loss_1y` |

cURL:

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"ibge_code":"4210100","years":1,"top_n":10}'
```

Se o payload for inválido (por exemplo `years: 99`), a API responde com `HTTP 422`.

### Exemplo: `GET /health`

```bash
curl -s http://127.0.0.1:8080/health
```

Resposta esperada: `HTTP 200` com status de serviço ativo e indicação se o modelo está carregado em memória.

---

## Variáveis de ambiente

Algumas rotinas e scripts usam variáveis de ambiente locais para facilitar execução e pré-carregamento.

Exemplo:

```bash
# Município de referência para demo
export DEFAULT_DEMO_IBGE=4210100

# Pré-carregar modelos em memória no startup da API
export FORESTGUARD_PRELOAD_IBGE=4210100
```

O arquivo `.env.example` pode servir como base para configurações locais.

---

## Estrutura do repositório

```text
.
├── README.md
├── VERSION
├── requirements.txt
├── .env.example
├── data/
│   ├── spatial/
│   │   ├── panels/4210100/
│   │   └── models/4210100_annual/
│   └── boundaries/
├── public/
│   ├── logo_sctecB.png
│   ├── logo_sctecW.png
│   └── quadrinhos.jpg
├── scripts/
│   ├── run_api.py
│   ├── run_desktop.py
│   ├── train_spatial.py
│   └── ...
└── src/supressao_vegetacao/
    ├── api/
    ├── desktop/
    ├── inference/
    ├── spatial/
    ├── reporting/
    └── ...
```

Não utilize caminhos absolutos do tipo `C:\Users\...` em comandos de execução do projeto.

---

## Link do vídeo / apresentação

Ainda não foi incluído um link público definitivo. Este espaço pode ser preenchido com a URL do vídeo de apresentação ou demo do projeto.

---

## História do projeto

Certa noite, enquanto trabalhava, minha filha de oito anos me perguntou: “o que tanto você digitava e trabalhava no computador?”. Tentei explicar de várias formas, mas ela sempre me olhava com a mesma expressão de quem não entendeu nada.

Então resolvi desenhar uma história em quadrinhos. Mostrei para ela e, para minha surpresa, ela disse: “Está fazendo um radar para não derrubar árvore”.

Foi uma forma simples e inspiradora de resumir a ideia central do projeto, e por isso decidi compartilhar.

<p align="center">
  <picture>
    <img alt="História em quadrinhos do ForestGuard" src="public/quadrinhos.jpg" width="600">
  </picture>
</p>
