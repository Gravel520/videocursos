'''
Script en Python.
Vamos a utilizar el Dropout, y el aumento de datos para entrenar una red
neuronal convolucional, para la clasificación de perros y gatos.
Una vez descargados los datos con 'tensorflow_dataset', los tenemos que
redimensionar a un tamaño de 100 x 100 pixeles, y cambiar de color a 
escala de grises, para que el entrenamiento sea más rápido.

Posteriormente separaremos los datos en dos variables, una para las imágenes
y otra para las etiquetas. Después las convertiremos a un array con 'numpy' y
normalizaremos el valor de las imágenes entre 0 y 1, dividiéndo los valores
entre 255.

Para poder utilizar el aumento de datos, tenemos que instalar el módulo 
'ImageDataGenerator' de 'tensorflow.keras.preprocessing.image', con el cual
podremos modificar, aleatoriamente, el aspecto de las imágenes, utilizando los
siguientes parámetros:
    * rotation_range - rota la imagen en un rango, en grados, que nosotros dispongamos.
    * width_shift_range - Desplaza horizontalmente hasta el % del ancho de la imagen.
    * height_shift_range - Desplaza verticalmente hasta el % de la altura de la imagen.
    * shear_range - Aplica una transformación de cizallamiento del rango en grados.
    * zoom_range - Aplica un zoom aleatorio entre el rango de acercamiento y alejamiento
                    en porcentaje que dispongamos.
    * horizontal_flip - Invierte horizontalmente la imagen, como un espejo.
    * vertical_flip - Invierte verticalmente la imagen, de arriba abajo.
¿Por qué usar esto?
    - Evita el sobreajuste: El modelo no memoriza las imágenes, sino que aprende a
    reconocer patrones generales.
    - Simula variaciones reales: Como si las imágenes fueran tomadas desde diferentes
    ángulos o posiciones.
    - Aumenta el tamaño efectivo del dataset sin necesidad de recolectar más datos.

Ya podemos definir una red neuronal convolucional, que es ideal para tareas de clasificación
de imágenes.
La arquitectura del modelo es la siguiente:
    - Capa convolucional. Extrae 32 mapas de características usando filtros de 3x3. La imagen
                            de entrada tiene 1 canal en escala de grises.
    - Capa Submuestreo. Reduce la dimensión espacial, el tamaño de la imagen, a la mitad.
    - Capa convolucional. Extrae 64 características más complejas, con activación 'relu'.
    - Capa Submuestreo. Reduce la dimensión espacial, el tamaño de la imagen, a la mitad.
    - Capa convolucional. Extrae 128 características más complejas, con activación 'relu'.
    - Capa Submuestreo. Reduce la dimensión espacial, el tamaño de la imagen, a la mitad.
    - Regularización (Dropout). Apaga aleatoriamente el 50% de las neuronas durante el
                                entrenamiento para evitar sobreajuste.
    - Aplanamiento (Flatten). Convierte los mapas de características en un vector plano
                                para alimentar a las capas densas.
    - Dense. Capa totalmente conectada, aprende combinaciones no lineales de las 
                características extraídas.
    - Capa de salida (Dense). Devuelve una probabilidad entre 0 y 1 para clasficación
                                binaria (sigmoid). Nos dice 0 gato o 1 perro.

Compilamos el modelo.
Para este tipo de entrenamiento, tenemos que asignar nosotros la cantidad de datos que
vamos a utilizar, tanto para el entrenamiento como para la validación. 85 % para el 
entrenamiento y 15 % para la validación.

Después creamos un generador de datos para entrenar el modelo, creando lotes de imágenes
aumentadas en tiempo real. Así evitamos el sobreajuste al mostrarle al modelo imágenes
ligeramente diferentes en cada época. Reducimos la necesidad de tener un dataset enorme. Y
aprovechamos la memoria al no tener que cargar todo el dataset a la vez, sino por lotes.

Ya podemos entrenar nuestro modelo. Lo hacemos durante 100 épocas y utilizamos 'TensorBoard'
para visualizar el proceso de entrenamiento.
Los parámetros utilizados con 'fit' son los siguientes:
    - data_gen_entrenamiento. Generador de imágenes aumentadas para entrenamiento. Los datos
                                creados en el párrafo anterior.
    - epochs. Entrena el modelo durante las vueltas completas sobre los datos.
    - batch_size. Tamaño del lote, ya está definido en el generador.
    - validation_data. Datos para evaluar el modelo después de cada época.
    - steps_per_epoch. Número de lotes que se procesan por época. Se calcula dividiendo el
                        total de imágenes entre el tamaño del lote.
    - validation_steps. Igual que el anterior, pero con los datos de validación.
    - callbacks. Ejecuta el callback de TensorBoard en cada época para guardar métricas.

Lo último será guardar el modelo para poder utilizarlo posteriormente, y no tener que volver
a entrenar el modelo.
'''

# Importamos las librerías.
import tensorflow as tf
import tensorflow_datasets as tfds

# Descargamos el set de datos de perros y gatos.
datos, metadatos = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Mostramos las primeras 5 imágenes con sus etiquetas.
tfds.as_dataframe(datos['train'].take(5), metadatos)

# Mostramos un gráfico con las 25 primeras imágenes. Le cambiamos su tamaño a 100x100
#   y a escala de grises en lugar de a color.
import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(20, 20))
TAMAÑO_IMG = 100

for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
    # Convierte el tensor de imagen a un array Numpy, y la redimensiona.
    imagen = cv2.resize(imagen.numpy(), (TAMAÑO_IMG, TAMAÑO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    plt.subplots(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen, cmap='gray')

# Creamos una lista para usarla como base en el entrenamiento del modelo.
datos_entrenamiento = []

# Mediante un bucle redimensionamos y configuramos las imágenes para que sean
#   uniformes y válidas para el entrenamiento.
for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (TAMAÑO_IMG, TAMAÑO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    # Reconfiguramos las dimensiones, para añadir el canal de escala de grises (1),
    #   y que sea compatible con redes neuronales convolucionales.
    imagen = imagen.reshape(TAMAÑO_IMG, TAMAÑO_IMG, 1)
    datos_entrenamiento.append([imagen, etiqueta])

# Creamos dos listas (X) para las imágenes, y (Y) para las etiquetas.
X = []
Y = []

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    Y.append(etiqueta)

# Convertimos las listas en arrays y normalizamos las imágenes.
import numpy as np

X = np.array(X).astype(np.float32) / 255
Y = np.array(Y)

# Importamos la librería para utilizar el aumento de datos.
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Creamos una instancia y configuramos el generador de aumento de datos, con los
#   distintos parámetros.
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

# Calcula estadísticas internas, como la media, si usas ciertos parámetros, pero
#   en este caso, al no usarlas, no es estrictamente necesario, se puede omitir.
datagen.fit(X)

# Creamos un gráfico para mostrar las primeras 10 imágenes después del aumento de datos.
plt.figure(figsize=(20, 8))
for imagen, etiqueta in datagen.flow(X, Y, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplots(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i].reshape(100, 100), cmap='gray')
    break

# Creamos el modelo con aumento de datos.
from keras.models import Sequential
import keras.layers

modeloCNN_AD = Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilamos el modelo.
modeloCNN_AD.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Tenemos que definir el porcentaje de datos que queremos para entrenamiento y para la
#   validación. Utilizamos un 85% para el entrenamiento (19700), y un 15% para validación
#   (3562)
X_entrenamiento = X[:19700]
X_validacion = X[19700:]

Y_entrenamiento = Y[:19700]
Y_validacion = Y[19700:]

# Genera un stream de datos de entrenamiento en tiempo real, aplicando todas las 
#   transformaciones que definimos en la instancia del aumento de datos. Se generarán
#   en lotes de 32 imágenes con sus etiquetas.
data_gen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)

# Creamos la instancia para la recolección de las métricas.
from keras.callbacks import TensorBoard

tensorboardCNN_AD = TensorBoard(log_dir='logs/cnn_AD')

# Entrenamos el modelo.
modeloCNN_AD.fit(
    data_gen_entrenamiento,
    epochs=100,
    batch_size=32,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32))),
    callbacks=[tensorboardCNN_AD]    
)

# Por último grabamos el modelo para poder utilizarlo en otras ocasiones.
modeloCNN_AD.save('C:/Users/katal/Documents/Python/Videocursos/Clasificador_perros_gatos/perros_gatos_CNN_AD.keras')
