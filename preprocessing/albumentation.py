#Read me
#Algoritmo de Albumentations para tranformaciones de imagenes etiquetadas con bounding boxex de formato YOLO
#Se puede modificar el pipeline en función de las transformaciones que se deseen aplicar. Para mayor información revisar: https://albumentations.ai/docs/2-core-concepts/transforms/
#Si se desea usar otra base de datos y otro target de imagenes se puede modificar el main

import albumentations as A
import cv2
import os
from glob import glob

#Flujo de transformaciónes que se aplicarán: Pipeline
#No se estan tomando transformaciones agresivas para no dañar los bounding boxes ni provocar sesgos
def get_Aumentation_Pipeline():
    pipeline = A.Compose([
        #El argumento p simboliza la probabilidad que la transformación ocurra
        A.HorizontalFlip(p=0.5),
        #Invertir horizontalmente la imagen (espejo)
        #Si bien el parametro se usa para evitar sobreentrenamiento de una orientación, al ser lunares, el invertir la imagen se puede tomar como un lunar nuevo
        A.RandomBrightnessContrast(p=0.3),
        #Cambia el brillo y el contraste aleatoriamente
        #Simulará diferentes condiciones de iluminación. Útil para dar mayor robustez frente a este factor que se controla de forma física
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        #Se aplicará traslación(5%), escalado(+-10%) o rotación(15°)
        #Simulará diferentes encuadros de captura. Útil para dar mayor robustez a fallo de encuadre del usuario
        #La gran mayoría de lunares en la base de datos se encuentra al centro, este parámetro ayudará a dar mayor cantidad de lunares en otras posiciones
        A.RandomGamma(p=0.2),
        #Ajusta la correción gamma de la imagen (brillo capturado por el dispositivo)
        #Mejorará la capacidad del modelo frente a diferentes curvas de brillo de diferentes capturadores
        #No se sabe con que dispositivo fueron capturadas las imagenes de la base de dato, sumado a que ninguna se ha realizado con la camara que se empleará. El parámetro ayudará a controlar mejor esa diferencia de captura
        A.RGBShift(p=0.2),
        #Cambiará aleatoriamente los canales RGB  (un poco más rojizo o un poco más verdoso por ejemplo)
        #Será un intento de variación del tono de piel
        #Dará mayor robustez frente a la ilumanación ambiental como el balance de blancos diferente que puede tener la camara a utilizar
        A.GaussianBlur(p=0.1),
        #Aplicará un leve desenfoque usando un filtro gaussiano.
        #Dará robustez frente a fallos del usuario al momento de enfocar la imagen al lunar de elección

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return pipeline

#Función de lectura de etiquetas YOLO
#labelPath es la ruta al archivo .txt que contiene una etiqueta del bouding box de una imagen
#formato YOLO: <clase> <x_centro> <y_centro> <ancho> <alto>
def read_Yolo_Label(labelPath):
    bboxes = []         #Lista de almacen de coordenadas de cada box (float)
    labels = []         #Lista de almacen de las clases de cada caja (int)
    with open(labelPath, 'r') as f:
        #Lectura del archivo
        for line in f.readlines():
            #Iteración linea por linea (hay imagenes con varias bouding boxes)
            cls, xc, yc, w, h = map(float, line.strip().split())    #División de la linea en las 5 variables de interes en float
            bboxes.append([xc, yc, w, h])
            labels.append(int(cls))
    return bboxes, labels

#Función de guardado de resultados
def save_Data(img, bboxes, labels, outputImgPath, outputLblPath):
    cv2.imwrite(outputImgPath, img)     #Guardado de imagen transformada
    #Guardado de la etiqueta y todos los vectores en formato YOLO
    with open(outputLblPath, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            line = f'{label} ' + " ".join(f'{coord:.6f}' for coord in bbox)
            f.write(line + '\n')

#Función de aplicación de aumentos
def augment_Dataset(imgDir, lblDir, outputImgDir, outputLblDir, targetCount):
    #Creación de las carpetas caso no existan
    os.makedirs(outputImgDir, exist_ok=True)
    os.makedirs(outputLblDir, exist_ok=True)
    #Obtención de la ruta de las imagenes originales 
    imagePaths = sorted(glob(os.path.join(imgDir, "*.jpg")))
    currentCount = len(imagePaths)      #imagenes total de la carpeta (punto de partida)
    #Invocación de los parámetros de transformación
    aug = get_Aumentation_Pipeline()      #objeto de tipo A.Compose (secuencia de transformaciones encadenadas)
    #Bucle de generación de nuevas imagenes hasta alcanzar el objetivo
    i = 0       #Cantidad de imagenes modificadas
    while i + currentCount < targetCount:
        #Bucle de procesamiento de cada imagen original
        for imgPath in imagePaths:
            #Creación de la ruta de archivo de las etiquetas de cada imagen
            base = os.path.basename(imgPath)        #img0001.jpg
            name = os.path.splitext(base)[0]        #img0001
            labelPath = os.path.join(lblDir, f'{name}.txt')
            #Lectura de imagen y etiqueta
            img = cv2.imread(imgPath)
            bboxes, labels = read_Yolo_Label(labelPath)
            #Transformación
            #Se devolverá un diccionario con:
            #transformed["image"] -> Nueva imagen aumentada
            #transformed["bboxes"] -> Nuevas coordenadas
            #transformed["class_labels"] -> Etiquetas sincronizadas
            transformed = aug(image=img, bboxes=bboxes, class_labels=labels)
            #Guardado de la nueva información
            newImgName = f'{name}_aug{i}.jpg'
            newLblName = f'{name}_aug{i}.txt'
            save_Data(transformed["image"], transformed["bboxes"], transformed["class_labels"],
                     os.path.join(outputImgDir, newImgName), os.path.join(outputLblDir, newLblName))
            #Control del bucle
            i += 1
            if i % 100 == 0:
                 print(f'{i} imagenes aumentadas...')
            if i + currentCount >= targetCount:
                break
    print(f'{i} imagenes aumentadas hasta llegar a {targetCount} imagenes')

#Main
#Clase cancer
augment_Dataset(
    imgDir="HAM10000_Dataset3/images/train/cancer",
    lblDir="HAM10000_Dataset3/labels/train/cancer",
    outputImgDir="Dataset3_Transformed/augmented/cancer/images",
    outputLblDir="Dataset3_Transformed/augmented/cancer/labels",
    targetCount=10000
)
#Clase noCancer 
augment_Dataset(
    imgDir="HAM10000_Dataset3/images/train/noCancer",
    lblDir="HAM10000_Dataset3/labels/train/noCancer",
    outputImgDir="Dataset3_Transformed/augmented/noCancer/images",
    outputLblDir="Dataset3_Transformed/augmented/noCancer/labels",
    targetCount=10000
)
