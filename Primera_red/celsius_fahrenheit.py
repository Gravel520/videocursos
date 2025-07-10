'''
Script en Python.
En este scripts, vamos a utilizar la red neuronal creada anteriormente
para pasar de celsius a fahrenheit.
La red ya fué entrenada y los parámetros ajustados para que funcionase
perfectamente, así que una vez operativa, podremos usarla.
Esta red esta entrenada y guardados sus parámetros y pesos en dos
archivos, que cargaremos y así no hara falta volver a entrenar la
red neuronal.
'''

# Con estas líneas de código desaparecen los mensaje informativos
#   al usar las librerías Keras y Tensorflow. 
# Después hay que importar TensorFlow y silenciar sus logs.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Importamos las librerías.
import numpy as np
import sys
import logging
from keras.models import model_from_json
'''
Importamos el módulo encargado de cargar el modelo del formato 
oficial recomendado desde TensorFlow 2.13, el nuevo favorito de Keras.
Guarda todo en un sólo archivo: arquitectura, pesos, configuración
de entrenamiento, etc, además es más rápido al predecir.
'''
from keras.models import load_model
import tensorflow as tf

# Silenciar logs de TensorFlow
tf.get_logger().setLevel(logging.ERROR)

json_file = open('C:/Users/katal/Documents/Python/Videocursos/Primera_red/modelo_celsius.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# Cargar pesos al nuevo modelo
loaded_model.load_weights('C:/Users/katal/Documents/Python/Videocursos/Primera_red/modelo_celsius.weights.h5')
print('Cargado modelo desde disco.')

# Compilar modelo cargado y listo para usar.
cel = 100
predicciones = loaded_model.predict(np.array([[cel]]))
predicciones = round(predicciones[0].item())
print(f'El resultado es {str(predicciones)} grados fahrenheit\n')

# Cargar modelo completo desde archivo .keras.
modelo_cargado = load_model('C:/Users/katal/Documents/Python/Videocursos/Primera_red/modelo_celsius_keras.keras')
predic_keras = modelo_cargado.predict(np.array([[cel]]))
predic_keras = round(predic_keras[0].item())
print(f'El resultado es {str(predic_keras)} grados fahrenheit,\ncargado desde el modelo .keras.')
