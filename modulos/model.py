"""
Proyecto de visión computacional para detección de obstáculos mediante YOLOV8 para robot autonomo

-   Módulo de modelo de detección de obstáculos

Autores: 
    -   Gutierrez Emiliano
    -   Valenzuela Alejo

Fecha de creación: 3-2-24
Última modificación: 14-3-24
Versión: 1.2.2

Funciones:
- detectar_obstaculos: Detecta obstáculos en una imagen utilizando un modelo entrenado de yolov8n.
"""

import cv2
import os
from ultralytics import YOLO
from obstaculos import obstaculos


def detectar_obstaculos(ruta_imagen, model):
    frame = cv2.imread(ruta_imagen)
    image_height, image_width = frame.shape[:2]
    results = model.track(frame, conf=0.5, iou=0.5, show=False)

    datos = []
    distancia_obstaculo = []
    angulo_obstaculos = []
    id_class_obstaculo = []

    for result in results:
        for r in result.boxes.data.tolist():
            try:
                x1, y1, x2, y2, id, score, class_id = r
            except:
                x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            id = int(id)

            if y2 > 343 and y2 <= 720:
                # Calculamos las coordenadas del punto en el borde inferior del centro del objeto
                w, h = (x2 - x1), (y2 - y1)
                cx, cy = x1 + w // 2, y2

                angulo_grados, distancia = obstaculos(cx, cy)
                datos += [[distancia, angulo_grados]]
                distancia_obstaculo.append(distancia)
                angulo_obstaculos.append(angulo_grados)
                class_id = int(class_id)
                id_class_obstaculo.append(class_id)

    return distancia_obstaculo, angulo_obstaculos, id_class_obstaculo
