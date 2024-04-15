"""
Proyecto de visión computacional para detección de obstáculos mediante YOLOV8 para robot autonomo

-   Módulo de interpolación

Este módulo contiene la definicion de funcion de distancia y variables relacionadas con la interpolación de datos para la estimacion de angulo y distancia de con la camara.

Autores: 
    -   Gutierrez Emiliano
    -   Valenzuela Alejo

Fecha de creación: 3-2-24
Última modificación: 14-3-24
Versión: 1.2.2

Funciones:
- f_interp: Función de interpolación de datos.
"""

import numpy as np
from scipy.interpolate import interp1d

# Puntos dados (px, distance_cm)
puntos = np.array([(343, 324.5), (370, 264.5), (413, 204.5), (493, 144.5), (570, 114.5), (690, 84.5), (720, 77.3)])

# Separamos las coordenadas x e y de los puntos
x = puntos[:, 0]
y = puntos[:, 1]

# Interpolamos una función para aproximar distancias
f_interp = interp1d(x, y, kind='cubic')

# Creamos los puntos para la interpolación
x_interp = np.linspace(min(x), max(x), 100)
y_interp = f_interp(x_interp)
