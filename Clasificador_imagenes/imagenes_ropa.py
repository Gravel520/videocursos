'''
Scripts en Python.
En este script, vamos a utilizar la red neuronal creada anteriormente para
clasificar imágenes de ropa, y poder predecir a que clase pertenece.

Importamos el módulo encargado de cargar el modelo del formato 
oficial recomendado desde TensorFlow 2.13, el nuevo favorito de Keras.
Guarda todo en un sólo archivo: arquitectura, pesos, configuración
de entrenamiento, etc, además es más rápido al predecir.

En un pequeño gráfico presentaremos la prenda seleccionada o elegida al
azar, junto con su gráfica de acierto.

'''
# Con estas líneas de código desaparecen los mensaje informativos
#   al usar las librerías Keras y Tensorflow. 
# Después hay que importar TensorFlow y silenciar sus logs.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importamos las librerías.
import numpy as np
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import matplotlib.pyplot as plt
from keras.models import load_model # Módulo para cargar el archivo .keras.

# Silenciar logs de TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Descargamos el set de datos de fashion MNIST de Zalando.
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
datos_prueba = datos['test']
nombre_clases = metadatos.features['label'].names

# Definimos las funciones reutilizables.
# Función para cargar el modelo.
def cargar_modelo(ruta):
    return load_model(ruta)

# Función para generar las predicciones. Se genera una predicción por
#   cada vuelta de iteración.
'''
Con 'shuffle' hacemos una mezcla aleatorio del conjunto de datos.
Después sacamos una imagen y su etiqueta correspondiente del dataset anterior.
    Con 'take' tomamos sólo un ejemplo, es un dataset que contienen un elemento.
    Con 'iter' convertimos ese dataset en un iterador para poder recorrerlo
        manualmente.
    Con 'next' extraemos el primer, y único, elemento del iterador, que es la
        tupla (imagen, etiqueta), y se lo asignamos a las variables.
Luego preparamos la imagen y la etiqueta para la predicción.
    Con 'numpy' convertimos el tensor de TensorFlow en un array Numpy.
    Reorganizamos la forma del array para que el modelo se pueda procesar:
        1. Se refiere al tamaño del lote.
        28, 28. Tamaño original del aimagen del Fashion MNIST.
        1. Canal del color, en este caso escala de grises.
    Con la etiqueta, primero la convertimos de tensor a array, y después lo
        metemos dentro de una lista con un solo elemento (1,)
Por último, hacemos la predicción y retornamos los datos.
'''
def generar_predicciones(modelo):
    datos_mezclados = datos_prueba.shuffle(buffer_size=100)
    imagen, etiqueta = next(iter(datos_mezclados.take(1)))
    imagen = imagen.numpy().reshape(1, 28, 28, 1)
    etiqueta = np.array([etiqueta.numpy()])
    prediccion = modelo.predict(imagen)
    return prediccion, etiqueta, imagen

# Función para graficar una imagen aleatorio mediante índice
def graficar_imagen(arr_predicciones, etiquetas_reales, imagenes):    
    imagen = imagenes.reshape((28, 28))
    etiqueta_real = int(etiquetas_reales[0])
    plt.imshow(imagen, cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    color = 'blue' if etiqueta_prediccion == etiqueta_real else 'red'

    plt.grid(False)
    plt.xlabel('{} {:2.0f}% ({})'.format(
        nombre_clases[etiqueta_prediccion],
        100 * np.max(arr_predicciones),
        nombre_clases[etiqueta_real]),
        color = color)
    
# Función para graficar la predicción mediante una barra.
def graficar_valor_arreglo(arr_predicciones, etiqueta_real):
    etiqueta_real = int(etiqueta_real[0])
    arr_predicciones = np.array(arr_predicciones)
    plt.grid(False)
    grafica = plt.bar(range(10), arr_predicciones[0], color = '#777777')
    plt.ylim([0, 1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')

# Función principal del script.
def main():
    modelo = cargar_modelo('C:/Users/katal/Documents/Python/Videocursos/Clasificador_imagenes/modelo_imagenes_ropa.keras')    

    plt.figure(figsize=(20, 10))
    for i in range(5):
        predicciones, etiquetas_prueba, imagenes_prueba = generar_predicciones(modelo)
        plt.subplot(2, 5, i+1)
        graficar_imagen(predicciones, etiquetas_prueba, imagenes_prueba)
        plt.subplot(2, 5, i+6)
        graficar_valor_arreglo(predicciones, etiquetas_prueba)

    # Con estas dos líneas de código, colocamos el grafico desde la parte superior
    #   izquierda de la pantalla, pero no es una función de matplotlib.
    manager = plt.get_current_fig_manager()
    manager.window.geometry('+0+0')
    
    plt.show()


# Arrancamos el scripts.
if __name__ == '__main__':
    main()