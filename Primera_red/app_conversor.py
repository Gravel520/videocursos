'''
Script en Python.
Con este script, vamos a hacer una aplicación de consola en la que nos pedirá
los grados en celsius y la red neuronal lo convertirá a grados fahrenheit.

Hemos entrenado una red neuronal con Keras y TensorFlow, en el archivo
'primera_red_neuronal.py', grabando el modelo entrenado en el nuevo
formato .keras. Es el formato oficial recomendado desde TensorFlow 2.13
en adelante. Guarda todo: arquitectura, pesos, configuración de 
entrenamiento, etc, en un solo archivo. Es muy simple de usar y compatible
con futuras versiones de Keras.

Utilizaremos un bucle infinito, y para detenerlo tendremos que introducir
la palabra 'fin'.
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
import platform
from keras.models import load_model # Módulo para cargar el archivo .keras.

# Silenciar logs de TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Definimos las funciones reutilizables.
# Función para cargar el modelo.
def cargar_modelo(ruta):
    return load_model(ruta)

# Función para convertir la cantidad con la red neuronal.
def convertir_red(modelo, cel):
    predic = modelo.predict(np.array([[cel]]))
    return round(predic[0].item())

# Función para convertir la cantidad con la formula matemática.
def conversor_matematico(cel):
    return round((cel * 1.8) + 32)

# Función para limpiar la consola
def limpiar_consola():
    comando = 'cls' if platform.system() == 'Windows' else 'clear'
    os.system(comando)

# Función principal.
def main():
    modelo = cargar_modelo('C:/Users/katal/Documents/Python/Videocursos/Primera_red/modelo_celsius_keras.keras')

    while True:
        limpiar_consola()
        cel = input('🌡️ Introduce grados Celsius (o escribe "fin" para salir):\n➡ ')

        # Creamos la interrupción del bucle.
        if cel.lower() == 'fin':
            print('👋 ¡Hasta la próxima!')
            break

        # Control de errores, por valor distinto de digito o 'fin'.
        try:
            cel = float(cel)
        except ValueError:
            print('⚠ Valor no válido. Intenta con un número.')
            input('Pulsa ↩ para continuar...')
            continue

        # Convertimos la cantidad con la red y la formula.
        pred_red = convertir_red(modelo, cel)
        pred_math = conversor_matematico(cel)

        # Presentamos los resultados.
        print(f'''
🔎 Resultado para {cel} ℃:

🧠 Red neuronal         → {pred_red} ℉
📐 Conversión directa   → {pred_math} ℉
''')
        input('Presiona ↩ para continuar...')

# Arrancamos el scripts.
if __name__ == '__main__':
    main()
