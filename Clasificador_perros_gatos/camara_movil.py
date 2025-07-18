'''
Scripts en Python.
Con este scripts podemos visualizar la cámara del movil en la pantalla
del ordenador, a través de una etiqueta. Hay que instalar en el movil
el programa 'DroidCam', e introducir la dirección ip que presenta el
programa en la variable 'url'
'''

from tkinter import *
from PIL import Image, ImageTk
import cv2
import sys

def onClossing():
    root.quit()
    cap.release()
    print('Cámara desconectada')
    root.destroy()

def callback():
    ret, frame = cap.read()

    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((400, 400))
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
root.title('Visión Artificial')

label = Label(root)
label.grid(row=0, column=0, padx=5, pady=5)

root.after(1, callback)

root.mainloop()