'''
Scripts en Python.
Esto código es una aplicación en Python que usa una red neuronal convolucional (CNN)
con aumento de datos, para clasificar en tiempo real si la imagen de una cámara 
muestra un perro o un gato.

Las librerías destacadas son las siguientes:
    cv2 - OpenCV para capturar imágenes desde la cámara y procesarlas.
    PIL - Convierte imágenes para mostrarlas en la interfaz.

Las imágenes las capturamos desde un dispositivo movil a través de su cámara. En este
    dispositivo tendremos que tener instalado la app 'DroidCam', que nos presenta
    una dirección https desde donde captura la imagen en la aplicación de Python.

El modelo de red neuronal fué entrenado en otro script 'entrenamiento_clasi_perros_gatos.py'.
'''

# Importamos las librerías.
import numpy as np
import cv2
import sys
from tkinter import *
from keras.models import load_model
from PIL import Image, ImageTk


# Cargamos el modelo.
modelo = load_model('C:/Users/katal/Documents/Python/Videocursos/Clasificador_perros_gatos/perros_gatos_CNN_AD.keras')

# Función para preprocesar la imagen.
'''
* Convertimos la imagen a una escala de grises, ya que el modelo fue entrenado así.
* Redimensionamos la imagen al tamaño del entreno, 100x100 pixeles.
* Añadimos el canal del color, en este caso será escala de grises (100, 100, 1).
* Normalizamos los valores del array de la imagen entre [0, 1].
* Añadimos la dimensión batch.

Con esto modificamos la imagen recibida por la cámara para ajustarla a la imagen
    utilizada en el entrenamiento.
'''
def preprocesar(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, (100, 100))
    imagen = np.expand_dims(imagen, axis=-1)
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

# Libera los recursos si se cierra la ventana.
def onClosing():
    root.quit()
    cap.release()
    print('Cámara desconectada')
    root.destroy()

# Función principal.
'''
Capturamos un fotograma desde la cámara, esto daría 'ret' como verdadero, y
    frame como el fotograma. Este frame será el que utilicemos para hacer la
    predicción, después de preprocesar la imagen.
    * Preprocesamos la imagen con la función correspondiente.
    * Preparamos la imagen capturada de la cámara para que pueda mostrarse
        correctamente en la interfaz Tkinter.
        * Convertimos el fotograma de OpenCV a RGBA. La A representa la transparencia.
        * Convertimos la imagen, de una matriz NumPy a un objeto de tipo 
            PIL.Image, para poder mostrarla en TKinter.
        * Redimensionamos la imagen a 400x400. Esta función mantiene la relación
            de aspecto original.
    * Hacemos la predicción y lo guardamos en una variable.
    * Convertimos el valor de la predicción en un float y redondeamos su valor
        con 2 decimales.
    * Se decide si es gato o perro con un umbral de 0.5.
    * Se actualiza el texto en la etiqueta.
    * Actualizamos la imagen en la interfaz gráfica.
        * Convertimos la imagen PIL, en un formato compatible con Tkinter.
            ImageTk.PhotoImage, es necesario para mostrar imágenes en Label u
            otros widgets de TKinter.
        * Actualizamos la etiqueta con la nueva imagen.
        * Almacenamos la imagen como una referencia dentro del objeto Label.
            TKinter elimina imágenes sin referencia, así que esta línea evita que
            la imagen desaparezca por el recolector de basura de Python.
        * Indica que se llame a la función 'callback()' después de 30 milisegundos.
            Esto genera un bucle de captura y predicción en tiempo real a 33 fps.
            Así se mantiene actualizada tanto la imagen como el resultado del 
                clasificador.
'''
def callback():
    # Capturamos el fotograma.
    ret, frame = cap.read()

    if ret: # Si captura verdadera.
        # Llamamos a la función preprocesar.
        imagen_preprocesada = preprocesar(frame)
        # Preparamos el frame para mostrarlo en TKinter.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(img)
        img.thumbnail((400, 400))
        # Hacemos la predicción y obtenemos el resultado.
        prediccion = modelo.predict(imagen_preprocesada)
        confianza = float(prediccion[0][0])
        confianza_redondeada = round(confianza, 2)
        # Se comprueba el resultado.
        resultado = '🐱 Gato' if confianza <= .5 else '🐶 Perro'
        resultado_label.config(text=f'Resultado: {resultado}-{confianza_redondeada}')
        # Actualizamos la imagen en la interfaz gráfica.
        tkimage = ImageTk.PhotoImage(img)
        label.configure(image = tkimage)
        label.image = tkimage
        root.after(30, callback)

    else: # Si captura es falsa, cerramos la aplicación.
        onClosing()

# Inicializamos la cámara IP, desde una app, en este caso DroidCam.
url = 'https://192.168.0.13:4343/video'
cap = cv2.VideoCapture(url)

# Comprobamos que la cámara se ha abierto.
if cap.isOpened():
    print('Cámara inicializada')
else:
    sys.exit('Cámara desconctada')

# Creamos la interfaz gráfica con Tkinter.
'''
Configuramos qué debe hacer el programa cuando el usuario intenta cerrar la ventana.
    protocol - Llama a una función del sistema Tkinter para definir comportamientos 
                personalizados.
    WM_DELETE_WINDOW - Es el nombre del evento que se lanza al cerrar la ventana
                        principal.
    onClosing - Es la función definida anteriormente para liberar recursos y evitar
                que el programa se quede colgado si la camara sigue abierta, y así
                recursos como la conexión de video o el modelo de predicción 
                podrían no cerrarse correctamente.
'''
root = Tk()
root.protocol('WM_DELETE_WINDOW', onClosing)
root.title('Clasificador Perros y Gatos')

# Añadimos las etiquetas.
'''
La primera etiqueta sería para contener la imagen capturada por la camara, estaría
    en una cuadrícula, en la primera fila y la primera columna, y con esa separación
    desde los bordes.
La segunda contendrá el texto referente al resultado de la predicción, por eso 
    definimos el tipo y el tamaño de letra del mensaje. Estará situada en la segunda
    fila y en la primera columna, con una separación en (y) desde el borde.
'''
label = Label(root)
label.grid(row=0, column=0, padx=5, pady=5)
resultado_label = Label(root, font=('Arial', 16))
resultado_label.grid(row=1, column=0, pady=10)

# Lanzamiento de la app.
# Llamamos a la función 'callback' cada 30 milisegundos.
root.after(30, callback)
root.mainloop()
