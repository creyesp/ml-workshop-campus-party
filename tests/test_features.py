"""
Pruebas unitarias para el módulo de ingeniería de características.
"""

import numpy as np
from src.features.build import MostCommonCategories


def test_most_common_categories_reduction():
    """
    Verifica que MostCommonCategories identifique correctamente las categorías
    más comunes y agrupe las poco comunes bajo la etiqueta 'other'.
    """
    # 'A' aparece 4 veces (57%), 'B' aparece 2 veces (28%), 'C' y 'D' aparecen 1 vez cada una (14%)
    # Frecuencias acumuladas ordenadas: A (57%), B (85%), C (99%), D (113%)
    X = np.array(
        [["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["C"], ["D"]], dtype=object
    )

    # Con un umbral de 0.8, A y B deberían conservarse, mientras C y D deberían agruparse
    transformer = MostCommonCategories(thr=0.8)
    transformer.fit(X)

    # Verificar que el fit guardó las categorías correctas para la columna 0
    common = transformer.common_categories_[0]
    assert "A" in common
    assert "B" in common
    assert "C" not in common
    assert "D" not in common

    # Transformar y verificar mapeo
    X_trans = transformer.transform(X)
    assert X_trans[0, 0] == "A"
    assert X_trans[4, 0] == "B"
    assert X_trans[6, 0] == "other"
    assert X_trans[7, 0] == "other"
