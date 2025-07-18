'''
Scripts en Python.

'''

# Importamos las librerías
import numpy as np
import cv2
import sys
from tkinter import *
from keras.models import load_model
from PIL import Image, ImageTk


# Cargamos el modelo
modelo = load_model('C:/Users/katal/Documents/Python/Videocursos/Clasificador_perros_gatos/perros_gatos_CNN_AD.keras')

# Función para preprocesar la imagen
def preprocesar(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, (100, 100))
    imagen = np.expand_dims(imagen, axis=-1)
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

def onClossing():
    root.quit()
    cap.release()
    print('Cámara desconectada')
    root.destroy()

def callback():
    ret, frame = cap.read()

    if ret:
        imagen_preprocesada = preprocesar(frame)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(img)
        img.thumbnail((400, 400))

        
        prediccion = modelo.predict(imagen_preprocesada)
        confianza = float(prediccion[0][0])
        confianza_redondeada = round(confianza, 2)
        
        resultado = '🐱 Gato' if confianza <= .5 else '🐶 Perro'
        resultado_label.config(text=f'Resultado: {resultado}-{confianza_redondeada}')

        tkimage = ImageTk.PhotoImage(img)
        label.configure(image = tkimage)
        label.image = tkimage
        root.after(1, callback)
    else:
        onClossing()

url = 'http://192.168.0.21:4747/video'
cap = cv2.VideoCapture(url)

if cap.isOpened():
    print('Cámara inicializada')
else:
    sys.exit('Cámara desconctada')

root = Tk()
root.protocol('WM_DELETE_WINDOW', onClossing)
root.title('Clasificador Perros y Gatos')

label = Label(root)
label.grid(row=0, column=0, padx=5, pady=5)
resultado_label = Label(root, font=('Arial', 16))
resultado_label.grid(row=1, column=0, pady=10)

root.after(1, callback)

root.mainloop()
