
import cv2
from ultralytics import YOLO
import random
import math
import os

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

######################################

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

def obstaculos(x2, y2):
    # x1, y1 son los puntos del centro inferior de la imagen (x1, y1 = 640, 0)
    x1, y1 = 640, 0

    vector1 = (x2 - x1, y2 - y1)  # Vector formado por (x1, y1) y (x2, y2)
    vector2 = (1280 - x1, 0 - y1)  # Vector formado por (x2, y2) y (1280, 0)

    producto_punto = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitud_vector1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitud_vector2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    coseno_angulo = producto_punto / (magnitud_vector1 * magnitud_vector2)
    angulo_radianes = math.acos(coseno_angulo)
    angulo_grados = math.degrees(angulo_radianes)

    # Asegurar que el ángulo esté en el rango de 0 a 180 grados
    if angulo_grados > 180:
        angulo_grados = 360 - angulo_grados

    distancia_cm = f_interp(y2).tolist()

    return angulo_grados, distancia_cm
########################################

# Modelo e imagenes a color
model = YOLO('../yolov8n_train_20-3-24.pt')
source = '../dataset_frames'
#source = '../bin_dataset_2'

# Modelo e imagenes a escala de grises
# model = YOLO('../best_grayDataset.pt')
# source = '../gray_dataset_frames'

# Obtener la lista de nombres de archivos de la carpeta
nombres_archivos = sorted(os.listdir(source))

classNames = ["pata mesa", "tacho basura", "pata silla", "caja"]

# Creo la matriz de colores de 100 combinaciones de colores diferentes para las boxes de cada obstaculo
# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classNames))]


clases = {
    0: 'pata mesa',
    1: 'tacho basura',
    2: 'pata silla',
    3: 'caja'
}

# Tamaño de la imagen de referencia (1280 x 720)
image_width = 1280
image_height = 720

apAng = 35  #grados
altura_camara = 27.6
distancia_inferior = 24.5+60

while True:
    # ret, frame = cap.read()
    # Iterar sobre los nombres de archivos y leer las imágenes
    for nombre_archivo in nombres_archivos:
        ruta_imagen = os.path.join(source, nombre_archivo)
        frame = cv2.imread(ruta_imagen)
        image_height, image_width = frame.shape[:2]
        # print(f'image_height: {frame} ,image_width: {image_width}')

        # Tomo las detecciones de objetos del video
        # results = model.track(frame, conf=0.5)
        # results = model(frame, conf=0.2)
        results = model.track(frame, conf=0.5, iou=0.5, show=False)
        datos = []
        distancia_obstaculo = []
        angulo_obstaculos = []
        id_class_obstaculo = []
        for result in results:
            for r in result.boxes.data.tolist():
                print(r)
                # x1, y1, x2, y2, id, score, class_id = r
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
                    print(f'y1: {y1}, y2: {y2}')
                    # Calcular las coordenadas del punto en el borde inferior del centro del objeto
                    center_x = (x1 + x2) / 2
                    bottom_center_y = y2
                    # Convertir las coordenadas a píxeles en una imagen de 1280 x 720
                    # pixel_x = int(center_x * image_width)
                    # pixel_y = int(bottom_center_y * image_height)
                    # Dibujar un punto en el centro del borde inferior del objeto
                    # cv2.circle(frame, (pixel_x, pixel_y), radius=5, color=(0, 0, 255), thickness=-1)

                    # Calculo las coordenadas del punto central en cada bounding box
                    w, h = (x2 - x1), (y2 - y1)
                    # cx, cy = x1 + w // 2, y1 + h // 2
                    cx, cy = x1 + w // 2, y2
                    # distancia = distance(cx, 0, cx, cy)
                    # distancia = f_interp(y2)
                    angulo_grados, distancia = obstaculos(cx, cy)
                    datos += [[distancia, angulo_grados]]
                    distancia_obstaculo.append(distancia)
                    angulo_obstaculos.append(angulo_grados)

                    class_id = int(class_id)
                    id_class_obstaculo.append(class_id)

                    color = colors[class_id]
                    # Tomo el nombre de la clase
                    class_name = classNames[int(class_id)]
                    # Calculo el nivel de la prediccion
                    conf = math.ceil((score*100))/100
                    # Armo un texto para mostrar sobre el recuadro
                    # label=f'{class_name},{conf}'
                    label = f'{class_name}, conf: {conf}'

                    # Dibujo el recuadro con un color diferente para cada deteccion
                    # cv2.rectangle(frame, (x1,y1), (x2,y2), (colors[id % len(colors)]), 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Calculo el largo del texto a mostrar
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0] - 90, y1 - t_size[1]

                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

                    # Dibujo el rectangulo ajustado al largo del texto
                    # cv2.rectangle(frame,(x1,y1),c2,(colors[id]),-1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)

                    start_point = (640, 720)
                    end_point = (cx, cy)
                    thickness = 2
                    # color = (colors[id % len(colors)])

                    # Dibujo una línea desde el punto de inicio al punto final
                    # cv2.arrowedLine(frame, start_point, end_point, (colors[id % len(colors)]), thickness)
                    cv2.arrowedLine(frame, start_point, end_point, color, thickness)
                    label_dist = f'distancia: {distancia:.2f} cm, angulo: {angulo_grados:.1f}'

                    # Coloco el texto
                    cv2.putText(frame, label, (x1, y1-2), 0, 0.5, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, label_dist, (cx, cy-4), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        print('Los datos distancia y angulo son:')
        print(datos)
        print('Los datos de distancia son:')
        print(distancia_obstaculo)
        print('Los datos de angulo son:')
        print(angulo_obstaculos)

        # Convertir ángulos a radianes para el gráfico polar
        angulos_radianes = np.radians(angulo_obstaculos)

        # Mostrar las imágenes en un solo gráfico
        fig, (ax, ax2) = plt.subplots(1, 2)  # Crear una figura con dos subgráficos

        # Graficar en un mapa polar
        ax.plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, projection='polar')
        for i, (distancia, angulo_radianes) in enumerate(zip(distancia_obstaculo, angulos_radianes)):
            ax.plot(angulo_radianes, distancia, marker='o', label=classNames[id_class_obstaculo[i]])
        ax.set_title('Mapa Polar de Distancias y Ángulos')

        ax.set_thetamin(0)  # Límite mínimo del eje theta (30 grados)
        ax.set_thetamax(180)  # Límite máximo del eje theta (60 grados)
        ax.plt.legend('obstaculos de HODOR')
        #plt.show()
        #cv2.imshow("frame", frame)
        cv2.imwrite('./results/' + nombre_archivo, frame)

        # Mostrar la segunda imagen en el segundo subgráfico
        ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        plt.show()

        if cv2.waitKey(1) == 27:
            break
    break