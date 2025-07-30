'''
Script en Python.
En este script entrenamos un modelo de clasificación de imágenes utilizando 
    MobileNetV2 a través de TensorFlow Hub, con un conjunto de datos de imágenes
    de cucharas, cuchillos y tenedores, que hemos obtenido por internet nosotros
    mismos.
    MobileNetV2 es una red neuronal convolución ya entrenada creada por Google y
    que se puede utilizar. La utilizamos sin la última capa de salida, que será la
    que incorporemos nosotros, según nuestra conveniencia, en este caso para la 
    clasificación de las imágenes anteriormente descritas, y congelaremos los pesos
    y sesgos creados por esta red para que no lo tengamos que entrenar nosotros, y
    sólamente entrenaremos las últimas capas de las imágenes.
1º Obtención de las imágenes:
    Hay una extensión en Google llamada 'Download All Images' que descarga en un 
    archivo .zip todas las imágenes que hemos visializado en el navegador. Posteriormente
    lo descomprimimos y tenemos todas las imágenes de lo que queramos.
    Funciona de la siguiente manera, escribimos en el buscador lo que queremos obtener,
    en el apartado de imágenes vamos bajando para que aparezcan más imágenes, y cuando
    creamos que ya tenemos suficientes, pulsamos sobre el botón de la extensión y se
    descargan todas las imágenes.

2º Preparar el Dataset:
    Se crean en este caso 3 carpetas, cada una con las imágenes correspondientes, y se
    descomprimen los archivos .zip. Habrá que hacer una limpia de imágenes porque seguro
    que NO TODAS las imágenes son válidas.

3º Aumento de Datos:
    Utilizamos aumento de datos para tener mayor volumen de imágenes de entrenamiento,
    simulando variaciones de las imágenes originales. Además establecemos un límite del
    uso de los datos entre entrenamiento y validación (80% - 20%).

4º Generadores de Datos:
    Lee las imágenes del disco, las procesa según el aumento de datos, las redimensiona
    a 224x224 y las entrega por lotes de 32 con etiquetas categóricas (one-hot).

5º Transfer Learning con MobileNetV2:
    Cargamos un modelo preentrenado MobileNetV2 sin la capa final, que sirve como
    extractor de características. Al poner 'trainable=False', se congela sus pesos
    para no entrenarlos de nuevo, lo cual es ideal con pocos datos. El 'input_shape'
    esta establecido al tamaño de la imagen y 3 canales, que son a color.

6º Construcción del Modelo:
    El modelo lo componen, la capa de la red neuronal descargada creada anteriormente.
    También añadimos una capa 'Dense' para clasificar las tres clases. La activación
    'softmax' transforma las salidas en probabilidades.

7º Compilación y Entrenamiento:
    Se usa 'adam' como optimizador, y 'categorical_crossentropy' como función de
    pérdida para clasificación multiclase.
    En el entrenamiento utilizamos 'samples' para usar el número total de imágenes en
    cada subconjunto y 'batch_size' que es el número de imágenes procesadas en cada
    paso, con esto hacemos una división entera para calcular cuántos lotes hay por
    época. Esto asegura que el modelo entrene con todo el conjunto una vez por época,
    en bloques de 32 imágenes.

8º Predicción y Visualización:
    Primero obtenemos un lote del conjunto de validación y hacemos la predicción.
    Posteriormente usamos 'argmax' para seleccionar la clase con mayor probabilidad, y
    mostramos la imagen junto con la clase predicha y el porcentaje de confianza.
'''

import tensorflow_hub as hub
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from tf_keras import layers, models

# Preparar el dataset
# Estructura de carpetas:
# .Pocos_datos_entrenamiento/content/dataset/
#   |-- cuchara/
#   |-- cuchillo/
#   |-- tenedor/

dataset_path = './Pocos_datos_entrenamiento/content/dataset'

# Aumento de datos
datagen = ImageDataGenerator(
    rescale=1. / 255, # Normaliza los pixeles a rango [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # 80% entrenamiento, 20% validación    
)

# Generamos los datos.
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Cargar MobileNetV2 sin capa de clasificación final
feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=(224, 224, 3),
    trainable=False # Congelamos pesos del modelo base
)

# Crear el modelo
model = models.Sequential([
    feature_extractor_layer,
    layers.Dense(3, activation='softmax') # 3 clases: cucharas, cuchillos, tenedores
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Entrenamiento
steps_por_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history = model.fit(
    train_generator,
    epochs=10,
    batch_size=32,
    validation_data=validation_generator,
    steps_per_epoch=steps_por_epoch,
    validation_steps=validation_steps
)

# Predicción
# Tomamos una imagen del conjunto de validación
sample_images, sample_labels = next(validation_generator)
preds = model.predict(sample_images)

# Mostrar predicción de la primera imagen
import matplotlib.pyplot as plt

plt.imshow(sample_images[0])
plt.axis('off')

# Diccionario para hacer el código más escalable
clases = {0: 'cuchara', 1: 'cuchillo', 2: 'tenedor'}

# Obtener la clase predicha con mayor probabilidad.
predic = np.argmax(preds[0])
confianza = np.max(preds[0])
cadena = clases.get(predic, 'desconocido')

# Mostrar el título con la predicción y la confianza de la predicción.
plt.title(f'Predicción: {cadena} - {confianza: .2f} %')
plt.show()

# Grabar el modelo.
#model.save('C:/Users/katal/Documents/Python/Videocursos/Pocos_datos_entrenamiento/clasificador_cocina.h5')
#print('modelo guardado')

# Gráficas de precisión
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_los = history.history['val_loss']

rango_epocas = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_los, label='Precisión Pruebas')
plt.legend(loc='upper right')
plt.title('Precisión de entrenamiento y pruebas')
plt.show()
