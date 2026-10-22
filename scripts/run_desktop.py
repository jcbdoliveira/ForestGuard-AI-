#!/usr/bin/env python3
"""
Script de Inicialização da Aplicação Gráfica Desktop Tkinter.
Uso: python scripts/run_desktop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supressao_vegetacao.desktop.app import run_desktop_app


def main():
    print("[Desktop] Iniciando interface gráfica Tkinter...")
    print("[Desktop] Padrão: município MAFRA-SC (4210100) | alvo anual y_loss_1y | horizontes 1–20 anos")
    print("[Desktop] Digite MAFRA-SC e clique em Consultar, ou pressione Enter.")
    run_desktop_app()


if __name__ == "__main__":
    main()
