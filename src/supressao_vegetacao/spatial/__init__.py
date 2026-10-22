"""Pipeline espacial: painel célula×ano, treino XGBoost e métricas reais (AUC/PR-AUC)."""

def __getattr__(name):
    if name in {"load_panel_csv", "temporal_split"}:
        from .panel import load_panel_csv, temporal_split
        return {"load_panel_csv": load_panel_csv, "temporal_split": temporal_split}[name]
    if name in {"train_spatial_model", "evaluate_binary"}:
        from .train_spatial import train_spatial_model, evaluate_binary
        return {"train_spatial_model": train_spatial_model, "evaluate_binary": evaluate_binary}[name]
    raise AttributeError(name)

__all__ = [
    "load_panel_csv",
    "temporal_split",
    "train_spatial_model",
    "evaluate_binary",
]
