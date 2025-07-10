'''
Script en Python.
Vamos a entrenar una red neuronal convolucional con aumento de datos
y dropout.
También se utiliza la técnica 'one-hot encoding' con el módulo 'to_categorical':
    es una forma de representar categorías como vectores binarios. En lugar de usar
    un único número para indicar una clase (por ejemplo, 3 para la clase 3), se crea
    un vector de ceros con un solo 1 en la posición correspondiente a la clase. Por
    ejemplo:
        Si tienes 10 clases (como dígitos del 0 al 9), en vez de 3, se representa
        como : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ¿Para qué sirve en redes neuronales?
        * Las redes neuronales suelen tener una capa de salida con tantas neuronas
        como clases, y se espera que solo una neurona 'active' indicando la clase
        predicha.
        * Así se facilita el cálculo de la función de pérdida como 'categorical
        crossentropy', que compara vectores y no solo números.
'''

# Importamos las librerías.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

# Cargar los datos de MNIST
# Aquí lo hago de otra manera porque es más simple para poder usar el modulo de 
# aumento de datos de Keras de esta manera
(X_entrenamiento, Y_entrenamiento), (X_pruebas, Y_pruebas) = mnist.load_data()

# Colocar los datos en la forma correcta (1, 28, 28, 1)
X_entrenamiento = X_entrenamiento.reshape(X_entrenamiento.shape[0], 28, 28, 1)
X_pruebas = X_pruebas.reshape(X_pruebas.shape[0], 28, 28, 1)

# Hacer 'one-hot encodin' de los resultados (e.g. en lugar de tener como resultado una
#   sola neurona, tendre 10 donde solo el resultado correcto sera 1 y el resto 0)
Y_entrenamiento = to_categorical(Y_entrenamiento)
Y_pruebas = to_categorical(Y_pruebas)

# Convertir a flotante y normalizar para que aprenda mejor la red
X_entrenamiento = X_entrenamiento.astype('float32') / 255
X_pruebas = X_pruebas.astype('float32') / 255

# Código para mostrar imágenes del set.
filas = 2
columnas = 8
num = filas * columnas
imagenes = X_entrenamiento[0:num]
etiquetas = Y_entrenamiento[0:num]
fig, axes = plt.subplots(filas, columnas, figsize=(1.5*columnas, 2*filas))
for i in range(num):
    ax = axes[i//columnas, i%columnas]
    ax.imshow(imagenes[i].reshape(28, 28), cmap='gray_r')
    ax.set_title('Label: {}'.format(np.argmax(etiquetas[i])))
plt.tight_layout()
#plt.show()
plt.close('all')

# Aumento de datos
# Variables para controlar las transformaciones que se harán en el aumento de datos
#   utilizando ImageDataGeneratos de Keras
rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
#rango_inclinacion = 15
rango_acercamiento = [0.5, 1.5]

datagen = ImageDataGenerator(
    rotation_range= rango_rotacion,
    width_shift_range= mov_ancho,
    height_shift_range= mov_alto,
    zoom_range= rango_acercamiento,
    #shear_range= rango_inclinacion
)

datagen.fit(X_entrenamiento)

# Código para mostrar imágenes del set.
filas = 4
columnas = 8
num = filas * columnas
print('ANTES: \n')
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5+columnas, 2*filas))
for i in range(num):
    ax = axes1[i//columnas, i%columnas]
    ax.imshow(X_entrenamiento[i].reshape(28,28), cmap='gray_r')
    ax.set_title('label: {}'.format(np.argmax(Y_entrenamiento[i])))
plt.tight_layout()
plt.show()
print('DESPUÉS:\n')
fig2, axes2= plt.subplots(filas, columnas, figsize=(1.5*columnas, 2*filas))
for X, Y in datagen.flow(X_entrenamiento,Y_entrenamiento.reshape(Y_entrenamiento.shape[0], 10),batch_size=num,shuffle=False):
    for i in range(0, num):
        ax = axes2[i//columnas, i%columnas]
        ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
        ax.set_title('label: {}'.format(int(np.argmax(Y[i]))))
    break
plt.tight_layout()
plt.show()

# Modelo
modelo = Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilación
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Los datos para entrenar saldrán del datagen, de manera que sean generados con
#   las transformaciones que indicamos
data_gen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)

TAMAÑO_LOTE = 32

# Entrenar la red.
print('Entrenando el modelo...')
epocas = 60
history = modelo.fit(
    data_gen_entrenamiento,
    epochs=epocas,
    batch_size=TAMAÑO_LOTE,
    validation_data=(X_pruebas, Y_pruebas),
    steps_per_epoch=int(np.ceil(60000 / float(TAMAÑO_LOTE))),
    validation_steps=int(np.ceil(10000 / float(TAMAÑO_LOTE)))
)

print('Modelo entrenado!')

# Exportar el modelo al explorador.
modelo.save('C:/Users/katal/Documents/Python/Videocursos/Redes_neuronales_convolucionales/modelo_digitos_aumento_datos.keras')
print('Modelo guardado!')
