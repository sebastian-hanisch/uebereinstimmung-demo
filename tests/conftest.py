"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def four_points():
    """Vier Wochen: Prognose f = (1, 2, 3, 4), Ist a = (2, 1, 4, 3). Beide Mittel 2,5, Varianzen (durch n) je 1,25, Kovarianz 0,75: r = 0,6, CCC = 2 · 0,75 / 2,5 = 0,6.
    Differenzen d = f − a = (−1, 1, −1, 1): mittlere Abweichung 0, RMSE = MAE = 1, Standardabweichung (n − 1) = √(4/3) = 1,1547, Grenzen ±1,96 · 1,1547 = ±2,2632;
    alle vier liegen innerhalb ±1."""
    return np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 4.0, 3.0])


@pytest.fixture
def three_weeks():
    """Ist a = (80, 100, 120): Mittel 100, Varianz (durch n) 800/3 = 266,67. Prognose f = 1,3 · a = (104, 130, 156): Mittel 130, Varianz 1,69 · 266,67 = 450,67, Kovarianz 346,67,
    r = 1, CCC = 2 · 346,67 / (450,67 + 266,67 + 30²) = 693,33 / 1 617,33 = 0,4287. Prognose a + 10: CCC = 2 · 266,67 / (2 · 266,67 + 100) = 533,33 / 633,33 = 0,8421."""
    return np.array([80.0, 100.0, 120.0])
