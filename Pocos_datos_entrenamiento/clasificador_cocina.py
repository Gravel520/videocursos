'''
Script en Python.

'''

import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Cargar modelo
model = load_model('C:/Users/katal/Documents/Python/Videocursos/Pocos_datos_entrenamiento/clasificador_cocina.h5')

# Diccionario de clases.
clases = {0: 'cuchara', 1: 'cuchillo', 2: 'tenedor'}

# Dirección de DroidCam.
URL_CAM = 'https://192.168.0.13:4343/video'

def predecir(frame):
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    clase = np.argmax(pred)
    confianza = np.max(pred)
    nombre = clases.get(clase, 'desconocido')
    return nombre, confianza

def capturar():
    cap = cv2.VideoCapture(URL_CAM)
    ret, frame = cap.read()
    if ret:
        nombre, conf = predecir(frame)
        resultado.set(f'{nombre} ({conf:.2f} confianza)')
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.config(image=img_tk)
        panel.image = img_tk
    cap.release()

# Interfaza Tkinter
root = tk.Tk()
root.title('Clasificación con modelo .h5')

panel = tk.Label(root)
panel.pack(pady=10)

resultado = tk.StringVar()
tk.Label(root, textvariable=resultado, font=('Arial', 14)).pack()

tk.Button(root, text='📷 Capturar imagen', command=capturar, font=('Arial', 12)).pack(pady=10)

root.mainloop()

