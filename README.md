# Hodor Computer Vision

Este repositorio pertenece al proyecto de automatizacion del robot HODOR, que fue propuesto como trabajo final.

## Integrantes 

- [Gutierrez Emiliano](https://github.com/emigutierr)

- [Valenzuela Alejo](https://github.com/AlejoValenzuela73)

## Proyecto:

El proyecto consiste en la deteccion de obstaculos empleando solamente una camara, para lo cual se realiza una automatizacion con yolo para la deteccion de un conjunto determinado de obstaculos.
El trabajo desarrollado se dividio en:

- Armado del dataset.

- Entrenamiento de la red neuronal basado en el modelo yolov8n.

- Procesamiento de datos para poder estimar las distancias.

## Hipotesis simplificativas

- Se asume que la apertura angular de la camara es lineal en todo su dominio

- Se asume que solamente se tendran los 4 obstaculos previstos en la sala, de manera que nada mas formara parte del dominio de obstaculos. Dichos obstaculos son:

        Pata mesa
        tacho basura
        Pata silla
        caja

- Se asume que los obstaculos siempre estaran apoyados sobre el piso.

- Se asume que el robot cuenta con el tiempo suficiente para realizar el procesamiento y tomar los datos de los obstaculos.

- Se asume que el horizonte se encuentra en el infinito, de manera que todos los obstaculos estaran siempre en la parte inferior de la imagen.

- Se asume que la altura de la camara sera siempre fija a `27.6 cm`.

## Corner Cases

- En caso de tener un obstaculo que se choque con el borde inferior de la imagen (py = 0), entonces se asume que esta en una distancia entre `0 cm` y `84.3 cm`

- Si bien se comunicaran todos los obstaculos detectados, solamente nos interesa comunicar con detalle el obstaculo mas cercano, pero en caso de que hayan varios a la misma distancia, dejaremos que el grupo de movimiento sea el encargado de decidir que hacer con el movimiento del robot.

## Funcionamiento

En este caso, el sistema se encarga de detectar las clases de objetos del mundo de obstaculos definidos, y luego se procesa la informacion referida a la distancia y angulo a la que se encuentra dicho obstaculo para ser comunicado a la etapa de movimiento.

<a href="https://github.com/AlejoValenzuela73/HodorComputerVision"> <img src="./HODOR_detection.gif"> </a>