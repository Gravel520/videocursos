'''
Script en Python.
Construiremos una pequeña red neuronal convolucional (CNN) que
aprenda sus propios filtros para identificar patrones en imágenes.
Usaremos el conjunto de datos MNIST, que contiene imágenes de dígitos
escritos a mano. Lo que nos interesa es ver cómo los filtros se ajustan
automáticamente durante el entrenamiento.
'''

# importamos las librerías.
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# Cargar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar imágenes y redimensionar a (28,28,1)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Codificar etiquetas
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Construir el modelo CNN
modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar y entrenar
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_test, y_test))

# Evaluar
test_loss, test_acc = modelo.evaluate(x_test, y_test)
print(f'Precisión en test: {test_acc:.2f}')

# Exportación del modelo a .keras
modelo.save('C:/Users/katal/Documents/Python/Videocursos/Redes_neuronales_convolucionales/modelo_digitos.keras')
print('Modelo guardado!')

'''
Ahora vamos a espiar cómo luce el 'cerebro visual' de nuestra red neuronal
después de haber aprendido a reconocer patrones.
Visualización de los filtros aprendidos en la primera capa coonvolucional.
Estos filtros han sido ajustados por el modelo durante el entrenamiento para 
detectar patrones últiles para clasificar dígitos. Vamos a extraerlos y mostrarlos.
'''

import matplotlib.pyplot as plt
import numpy as np

# Obtener los filtros de la primera capa convolucional.
filtros, bias = modelo.layers[0].get_weights()

# Normalizar los valores para visualización.
filtros = (filtros - filtros.min()) / (filtros.max() - filtros.min())

# Número de filtros
num_filtros = filtros.shape[-1]

# Mostrar los filtros
fig, axes = plt.subplots(4, 8, figsize=(12, 6)) # 32 filtros en total
for i in range(num_filtros):
    fila = i // 8
    col = i % 8
    ax = axes[fila, col]
    ax.imshow(filtros[:, :, 0, i], cmap='gray') # Mostrar el canal único (blanco y negro)
    ax.axis('off')
    ax.set_title(f'Filtro {i}')

plt.tight_layout()
plt.show()

'''
Que se verá.
    * Cada imagen pequeña representa un filtro que la red aprendió.
    * Muchos se parecen a detectores de líneas, curvas, bordes... ¡como los de 
        visión biológica!
    * A medida que el modelo aprende más, los filtros se especializan más en 
        patrones relevantes.
'''

'''
Script en Python.
Podemos visualizar cómo responden estos filtros a una imagen concreta (por ejemplo,
el dígito '4'), para ver qué partes activan cada filtro.
'''

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model

# Seleccionamos una imagen del conjunto de prueba
imagen = x_test[4].reshape(1, 28, 28, 1) # Imagen del dígito '4'

# Creamos un modelo que devuelve las salidas de la primera capa convolucional
modelo_intermedio = Model(inputs=modelo.inputs, outputs=modelo.layers[0].output)

# Obtenemos las activaciones
activaciones = modelo_intermedio.predict(imagen)

# Número de filtros.
num_filtros = activaciones.shape[-1]

# Visualizamos los mapas de activación.
fig, axes = plt.subplots(4, 8, figsize=(12, 6)) # 32 filtros
for i in range(num_filtros):
    fila = i // 8
    col = i % 8
    ax = axes[fila, col]
    ax.imshow(activaciones[0, :, :, i], cmap='viridis')
    ax.axis('off')
    ax.set_title(f'Filtro {i}')

plt.tight_layout()
plt.show()

