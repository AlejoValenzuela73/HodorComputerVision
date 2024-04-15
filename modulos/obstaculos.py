"""
Proyecto de visión computacional para detección de obstáculos mediante YOLOV8 para robot autonomo

-   Módulo de cálculo de ángulos y distancias de obstáculos

Este módulo contiene funciones para calcular ángulos y distancias de obstáculos.

Autores: 
    -   Gutierrez Emiliano
    -   Valenzuela Alejo

Fecha de creación: 3-2-24
Última modificación: 14-3-24
Versión: 1.2.2

Funciones:
- obstaculos: Calcula el ángulo y la distancia de un obstáculo desde un punto de referencia.
"""

import numpy as np

def obstaculos(x2, y2):
    # x1, y1 son los puntos del centro inferior de la imagen (x1, y1 = 640, 720)
    # x2, y2 son los puntos del centro inferior del obstaculo
    
    x1, y1 = 640, 720
    
    vector1 = [x2 - x1, y2 - y1]  # Vector formado por (x1, y1) y (x2, y2)
    vector2 = [1280 - x1, 720 - y1]  # Vector formado por (x2, y2) y (1280, 720)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2)
    theta = np.arccos(cos_theta)

    angulo_grados = np.degrees(theta)

    distancia_cm = f_interp(y2).tolist()

    return angulo_grados, distancia_cm
