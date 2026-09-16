El programa LUNAR esta diseñado para la clasificación binaria de cáncer mediante fotografías. Se basa en el algoritmo YOLO y la base de datos HAM10000. Se diseñó para ser utilizado en un Raspeberry Pi 5 conectado a una cámara.

El presente repositorio es solo un compilado de lo realizado en el curso "Proyecto de Biodiseño 1" en la UPCH. Se recomienda revisar los códigos antes de su implementación o uso, debido a que se han movido carpetas del repositorio local, por tanto algunos paths puede que esten rotos.

### Librerias clave
- [Ambumentations](https://github.com/albumentations-team/albumentations): Permite realizar data augmentation. Compatible con los bounding boxes de YOLO
- [EigenCam](https://github.com/rigvedrs/YOLO-26-CAM): Mapa de calor de zonas de mayor interés del modelo dándole una capa extra de interpretación al usuario.
 
# Aplication

Programa final lanzado en el Raspberry Pi 5. El código ha sido modicado para poder subirlo a github; las credenciales como lógica de sincronización con la nube de la aplicación  se encuentra desactivada

# Preprocessing

## randomizador3.py

Toma la información de `dataset/HAM10000.xlsx` para generar de manera aleatoria los 3 grupos de entrenamiento: `train` - `val` - `externalval`, tanto de las imágenes como los labels.

Se recomienda descargar el dataset HAM10000 (Por cuestiones de memoria no se encuentra en el repositorio), colocar las imágenes en la carpeta Dataset y ejecutar este script

## Albumentation.py

Script basado en la librería Albumentation para realizar data augmentation. Se realiza un aumento aproximado de 10000 a 25000 imágenes. Conserva los bounding boxes.

Este script cuenta con un pipeline de filtros que puede ser modificado en función de las necesidades que se requieran

# Dataset

El dataset utilizado fue el HAM10000. El presente `.xlsx` es el original del dataset con una columna agregada llamada `vector`. Esta columna define los vectores de los bounding boxes del algoritmo YOLO.

Las imágenes del dataset no se incluyen en el repositorio. Para utilizar el código, estas deben ser colocadas en la carpeta:

`dataset/Images/`

# Models

[14/09/2026] El modelo se encuentra en el Raspberry Pi 5, no cuenta por el momento con la copia para subirla, de obtenerla se actualizará esta sección.
