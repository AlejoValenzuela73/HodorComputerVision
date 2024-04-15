"""
Proyecto de visión computacional para detección de obstáculos mediante YOLOV8 para robot autonomo

Autores: 
    -   Gutierrez Emiliano
    -   Valenzuela Alejo

Fecha de creación: 3-2-24
Última modificación: 14-3-24
Versión: 1.2.2

"""

import cv2
from ultralytics import YOLO
import random
import math
import os

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#modo = 1    # Solo muestro la imagen con los resultados y guardo la imagen
modo = 2    # Solo muestro el mapa en coord. polares y guardo la imagen
#modo = 3    # muestro ambos y guardo ambos
save = 'yes'    #yes/no para guardar los datos

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


# Modelo e imagenes a color
model = YOLO('trainedYOLO_color.pt')
source = './dataset_frames'

# Modelo e imagenes a escala de grises
# model = YOLO('trainedYOLO_gray.pt')
# source = './gray_dataset_frames'

# Obtener la lista de nombres de archivos de la carpeta
nombres_archivos = sorted(os.listdir(source))
# nombres_archivos = source

classNames = ["pata mesa", "tacho basura", "pata silla", "caja"]

# Creo la matriz de colores de 100 combinaciones de colores diferentes para las boxes de cada obstaculo
colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255)]
# Rojo: (255, 0, 0), Azul: (0, 0, 255), Verde: (0, 255, 0), Amarillo: (0, 255, 255)

clases = {
    0: 'pata mesa',
    1: 'tacho basura',
    2: 'pata silla',
    3: 'caja'
}

# Tamaño de la imagen de referencia (1280 x 720)
image_width = 1280
image_height = 720

apAng = 35  # grados
altura_camara = 27.6
distancia_inferior = 24.5 + 60

while True:
    # ret, frame = cap.read()   (para el caso de camara)
    
    # En caso de trabajar con imagenes guardadas, iteramo sobre los nombres de los archivos para leer las imágenes
    for nombre_archivo in nombres_archivos:
        ruta_imagen = os.path.join(source, nombre_archivo)
        frame = cv2.imread(ruta_imagen)
        image_height, image_width = frame.shape[:2]
        results = model.track(frame, conf=0.5, iou=0.5, show=False)

        datos = []
        distancia_obstaculo = []
        angulo_obstaculos = []
        id_class_obstaculo = []

        for result in results:
            for r in result.boxes.data.tolist():
                print(r)
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
                    # Calculamos las coordenadas del punto en el borde inferior del centro del objeto
                    w, h = (x2 - x1), (y2 - y1)
                    cx, cy = x1 + w // 2, y2

                    angulo_grados, distancia = obstaculos(cx, cy)
                    datos += [[distancia, angulo_grados]]
                    distancia_obstaculo.append(distancia)
                    angulo_obstaculos.append(angulo_grados)
                    class_id = int(class_id)
                    id_class_obstaculo.append(class_id)

                    # Lo que sigue en el codigo se encarga simplemente de mostrar los datos.
                    color = colors[class_id]

                    # Tomo el nombre de la clase
                    class_name = classNames[int(class_id)]

                    # Calculo la confianza de la prediccion
                    conf = math.ceil((score * 100)) / 100

                    # DEFINO LOS LABELS A MOSTRAR
                    # label = f'{class_name}, conf: {conf}'
                    label = f'{class_name}'
                    label_conf = f'conf: {conf}'
                    label_dist = f'd:{distancia:.2f} cm'
                    label_ang = f'phi:{angulo_grados:.1f} grad'
                    print(f'{label}, x1:{x1},x2:{x2},y1:{y1},y2:{y2}, {label_conf}, {label_dist}, {label_ang}')

                    start_point = (640, 720)
                    end_point = (cx, cy)
                    thickness = 2

                    # Calculo el largo del texto a mostrar
                    largo_label = max(len(label), len(label_conf), len(label_dist), len(label_ang))
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                    print(t_size)

                    c3 = x1 + largo_label * 8 + 3, y1 + 4 * t_size[1]  # Coordenada para recuadro de datos
                    cv2.rectangle(frame, (x1, y1), c3, color, -1, cv2.LINE_AA)  # Rectangulo de datos

                    y_text_distance = 4
                    # text_color = (255, 255, 255)
                    text_color = (0, 0, 0)

                    # Dibujo el recuadro con un color diferente para cada clase
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


                    # Dibujo una línea desde el punto de inicio al punto final
                    cv2.arrowedLine(frame, start_point, end_point, color, thickness)

                    # Coloco el texto
                    cv2.putText(frame, label, (x1, y1 + 3 * y_text_distance), 0, 0.5, text_color, thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(frame, label_conf, (x1, y1 + 9 * y_text_distance), 0, 0.5, text_color, thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(frame, label_dist, (x1, y1 + 15 * y_text_distance), 0, 0.5, text_color, thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(frame, label_ang, (x1, y1 + 21 * y_text_distance), 0, 0.5, text_color, thickness=1,
                                lineType=cv2.LINE_AA)

        print('Los datos distancia y angulo son:')
        print(datos)
        print('Los datos de distancia son:')
        print(distancia_obstaculo)
        print('Los datos de angulo son:')
        print(angulo_obstaculos)
        print('las clases detectadas son:')
        print(id_class_obstaculo)

        if modo == 1 or modo == 3:
            cv2.imshow("frame", frame)  #En caso de ejecutar en google collab, comentar esta linea
            # cv2_imshow(frame)         #En caso de ejecutar en google collab, descomentar esta linea
            if save == 'yes':
                cv2.imwrite('./results/' + nombre_archivo, frame)

        if modo == 2 or modo == 3:
            #   Graficamos el mapa de datos en coordenadas polares
            angulos_radianes = np.radians(angulo_obstaculos)

            unique_list = []
            plt.figure(figsize=(5, 5))
            ax = plt.subplot(111, projection='polar')
            for i, (distancia, angulo_radianes, id_clase) in enumerate(
                    zip(distancia_obstaculo, angulos_radianes, id_class_obstaculo)):
                color = 'blue' if id_clase == 0 else 'red' if id_clase == 1 else 'green' if id_clase == 2 else 'orange'
                if id_clase not in unique_list:
                    unique_list.append(id_clase)
                    ax.plot(angulo_radianes, distancia, marker='o', color=color, label=classNames[id_clase])
                    plt.legend()
                else:
                    ax.plot(angulo_radianes, distancia, marker='o', color=color)  # , label= classNames[id_clase])

            ax.set_thetamin(0)  # Límite mínimo del eje theta (30 grados)
            ax.set_thetamax(180)  # Límite máximo del eje theta (60 grados)

            if save == 'yes':
                # Ruta donde guardar las imagenes
                ruta_carpeta = './mapa/'

                # Guardar la figura en la carpeta especificada
                plt.savefig(ruta_carpeta + nombre_archivo)
            plt.show()


        if cv2.waitKey(1) == 27:
            break
    break