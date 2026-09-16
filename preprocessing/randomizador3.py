import random
import shutil
import os
import pandas as pd

# Dataset
df = pd.read_excel("HAM10000.xlsx")

# Rutas de acceso
sourceImg = os.path.join("HAM10000", "HAM10000_images")
basePath = "HAM10000_Dataset3"
trainImg = os.path.join(basePath, "images", "train")
valImg = os.path.join(basePath, "images", "val")
testImg = os.path.join(basePath, "images", "externalTest")
trainTxt = os.path.join(basePath, "labels", "train")
valTxt = os.path.join(basePath, "labels", "val")
testTxt = os.path.join(basePath, "labels", "externalTest")

# Crear subcarpetas necesarias
for path in [trainImg, valImg, testImg, trainTxt, valTxt, testTxt]:
    os.makedirs(os.path.join(path, "cancer"), exist_ok=True)
    os.makedirs(os.path.join(path, "noCancer"), exist_ok=True)

# Preparar información del dataset
listaInfo = list(zip(df['image_id'], df['Vector']))
cancer_images = [item for item in listaInfo if item[1].split()[0] == '0']
noCancer_images = [item for item in listaInfo if item[1].split()[0] == '1']

def dividir_datos(lista):
    random.shuffle(lista)
    total = len(lista)
    train = lista[:int(0.7 * total)]
    val = lista[int(0.7 * total):int(0.95 * total)]
    test = lista[int(0.95 * total):]
    return train, val, test

# Dividir los datos por clase
train_cancer, val_cancer, test_cancer = dividir_datos(cancer_images)
train_noCancer, val_noCancer, test_noCancer = dividir_datos(noCancer_images)

# Combinación final
conjuntos = [
    (train_cancer, "train", "cancer"),
    (val_cancer, "val", "cancer"),
    (test_cancer, "externalTest", "cancer"),
    (train_noCancer, "train", "noCancer"),
    (val_noCancer, "val", "noCancer"),
    (test_noCancer, "externalTest", "noCancer")
]

# Recorrido y copiado
for lista, conjunto, clase in conjuntos:
    imgPath = os.path.join(basePath, "images", conjunto, clase)
    txtPath = os.path.join(basePath, "labels", conjunto, clase)

    for id, vec in lista:
        # Guardar etiqueta
        txt_file = os.path.join(txtPath, f"{id}.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(str(vec))

        # Copiar imagen
        src_img = os.path.join(sourceImg, f"{id}.jpg")
        dst_img = os.path.join(imgPath, f"{id}.jpg")
        try:
            shutil.copy(src_img, dst_img)
            print(f"Imagen {id}.jpg copiada en {conjunto}/{clase}")
        except Exception as e:
            print(f"Error al copiar {id}.jpg: {e}")

print("Proceso finalizado correctamente ✅")
