# Proyecto de automatizacion con vision artificial
# Integrantes:
#   - Gutierrez Emiliano
#   - Valenzuela Alejo

import cv2
from ultralytics import YOLO
import random
import math
import os

def distance(x1,y1,x2,y2):
    # x1, y1 son los puntos del centro inferior de la imagen
    # x2, y2 son los puntos del centro inferior del recuadro del obstaculo (punto que se dibuja)

    #Calculo el modulo entre puntos, y lo asumo como la altura
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    #Calculo el beta con el modulo
    beta = -35 / 360 * (360-dist) + 35

    #Con beta calculo la distancia aproximada
    return -25.71 * (beta - 90) / beta

# Modelo e imagenes a color
#model = YOLO('trainedYOLO_color.pt')
#source = '/dataset_frames'

# Modelo e imagenes a escala de grises
model = YOLO('trainedYOLO_gray.pt')
source = '/gray_dataset_frames'


# Obtener la lista de nombres de archivos de la carpeta
nombres_archivos = sorted(os.listdir(source))

# Creo la matriz de colores de 100 combinaciones de colores diferentes para las boxes de cada obstaculo
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

classNames = ["pata mesa","tacho basura","pata silla","caja"]

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

while True:
    #ret, frame = cap.read()
    # Itero sobre los nombres de archivos y leer las imágenes
    for nombre_archivo in nombres_archivos:
        ruta_imagen = os.path.join(source, nombre_archivo)
        frame = cv2.imread(ruta_imagen)

        #Tomo las detecciones de objetos del video
        #results = model.track(frame, conf=0.5)
        #results = model(frame, conf=0.2)
        results = model.track(frame, conf=0.2, iou=0.5, show=False)
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

                # Calculo las coordenadas del punto en el borde inferior del centro del objeto
                center_x = (x1 + x2) / 2
                bottom_center_y = y2

                # Calculo las coordenadas del punto
                w, h = (x2 - x1), (y2 - y1)        
                cx, cy = x1 + w // 2, y2

                # Calculo la distancia
                distancia = distance(cx, 0, cx, cy)

                class_id = int(class_id)
                #Tomo el nombre de la clase
                class_name=classNames[int(class_id)]
                #Calculo el nivel de la prediccion
                conf=math.ceil((score*100))/100

                # Muestro los datos
                #Armo un texto para mostrar sobre el recuadro
                #label=f'{class_name},{conf}'
                label = f'{class_name}, conf: {conf}'

                #Dibujo el recuadro con un color diferente para cada deteccion
                cv2.rectangle(frame, (x1,y1), (x2,y2), (colors[id % len(colors)]), 3)

                #Calculo el largo del texto a mostrar
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2=x1+t_size[0]-90,y1 - t_size[1]

                cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

                #Dibujo el rectangulo ajustado al largo del texto
                cv2.rectangle(frame,(x1,y1),c2,(colors[id]),-1,cv2.LINE_AA)

                start_point = (640, 720)
                end_point = (cx, cy)
                thickness = 2
                color = (colors[id % len(colors)])

                # Dibujo una línea desde el punto de inicio al punto final
                cv2.arrowedLine(frame, start_point, end_point, (colors[id % len(colors)]), thickness)
                label_dist = f'distancia: {distancia:.2f}'

                #Coloco el texto
                cv2.putText(frame, label, (x1, y1-2), 0, 0.5, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, label_dist, (cx, cy), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("frame", frame)
        cv2.imwrite('./results/'+nombre_archivo, frame)
        if cv2.waitKey(1) == 27:
            break
    break