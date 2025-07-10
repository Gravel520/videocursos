'''
Script en Python.
Con este script podremos hacer una clasificación después de recibir
de entrada una imagen de una prenda de ropa, el modelo nos dirá a 
que categoría pertenece; pantalon, sandalia, blusa, etc...

Este problema ya no es de regresión, sino de clasificación, y nos
enfrentamos a los siguientes problemas:
    - Clasificación. Las neuronas de salida serán la misma cantidad
        que tipos de, en este caso prendas, existan en la clasificación.
    - Tipo de entrada. Recibiremos imágenes, serán convertidas a escala
        de grises, y cada pixel de esa imagen será un valor entre 0 y
        255, que son los valores entre negro total y blanco total.
    - Red neuronal convolucional. Son las que se utilizan para los
        problemas de clasificación, pero en este caso no lo haremos.
    - Capas ocultas. Le damos a la red la oportunidad de utilizar más
        sesgos y pesos, al usar más cantidad de neuronas en cada capa.
    - Funciones de activación. Las capas ocultas nos permiten tener más
        matices pero seguirán siendo lineales, estás funciones, usadas
        en la salida que da la neurona cambia su valor, haciendolo
        más manejable. Una de las más utilizadas es la función ReLu.
'''

# Con estas líneas de código desaparecen los mensaje informativos
#   al usar las librerías Keras y Tensorflow. 
# Después hay que importar TensorFlow y silenciar sus logs.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importemos las librerías.
from keras.models import Sequential
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import logging

# Silenciar logs de TensorFlow.
tf.get_logger().setLevel(logging.ERROR)

# Descargar set de datos de Fahion MNIST de Zalando.
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# Imprimir los metadatos para ver que trae el set.
#print(metadatos)

# Obtenemos en variables separadas los datos de entrenamiento (60k) y
#   prueba (10k).
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

# Etiquetas de las 10 categorias posibles.
nombres_clases = metadatos.features['label'].names
#print(nombres_clases)

# Función de normalización para los datos (Pasar de 0-255 a 0-1).
# Hace que la red aprenda mejor y más rápido.
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 # Aqui lo pasa de 0-255 a 0-1
    return imagenes, etiquetas

# Normalizar los datos de entrenamiento y pruebas con la función anterior
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

# Agregar a cache (usar memoria en lugar de disco, entrenamiento más
#   rápido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

# Mostrar una imagen de los datos de pruebas, de momento mostremos
#   la primera
for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28, 28)) # Redimensionar, cosas de tensores.

import matplotlib.pyplot as plt

# Dibujar
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
#plt.show()
plt.close('all')

# Dibujar mas
plt.figure(figsize=(10, 10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombres_clases[etiqueta])
#plt.show()
plt.close('all')

# Crear el modelo
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28,1)), # 1 - blanco y negro
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(10, activation='softmax') # Para redes de clasificación
])

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Los numeros de datos en entrenamiento y pruebas (60k y 10k)
num_ej_entrenamiento = metadatos.splits['train'].num_examples
num_ej_pruebas = metadatos.splits['test'].num_examples

print(num_ej_entrenamiento)
print(num_ej_pruebas)

# El trabajo por lotes permite que entrenamientos con gra cantidad de datos
#   se haga de manera más eficiente.
TAMAÑO_LOTE = 20

# Shuffle y repeat hacen que los datos esten mezclados de manera
#   aleatoria para que la red no se vaya a aprender el orden de las cosas
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMAÑO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMAÑO_LOTE)

import math

# Entrenar el modelo
print('Comienza el entrenamiento...')
historial = modelo.fit(
    datos_entrenamiento, 
    epochs=5,
    steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMAÑO_LOTE),
    verbose=False
)
print('Fín del entrenamiento.')

# Ver la función de pérdida
plt.xlabel('# Vuelta')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])
plt.show()

# Pintar una cuadrícula con varias predicciones, y marcar si fue correcta
#   (azul) o incorrecta (roja)
import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)
    print(predicciones, etiquetas_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(
        nombres_clases[etiqueta_prediccion],
        100 * np.max(arr_predicciones),
        nombres_clases[etiqueta_real]),
        color = color)
        
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color = '#777777')
    plt.ylim([0, 1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')

filas = 5
columnas = 4
num_imagenes = filas * columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)    

plt.show()

# Probar una imagen suelta
imagen = imagenes_prueba[4] # Al ser la variable imagenes_prueba solo tiene lo que se le puso en el bloque anterior
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print('Predicción: ' + nombres_clases[np.argmax(prediccion[0])])

# Exportación del modelo a .keras
modelo.save('C:/Users/katal/Documents/Python/Videocursos/Clasificador_imagenes/modelo_imagenes_ropa.keras')
print('Modelo guardado...')