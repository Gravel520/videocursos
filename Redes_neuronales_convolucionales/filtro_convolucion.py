'''
Script en Python.
Simularemos cómo una capa de convolución detecta bordes simples en una imagen
real. Usaremos OpenCV y matplotlib para ilustrarlo.
Ejemplo: detectar bordes en una imagen con un filtro de convolución.
'''

# importamos las librerías.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises.
imagen = cv2.imread('C:/Users/katal/Documents/Python/Videocursos/Redes_neuronales_convolucionales/gato.jpg', cv2.IMREAD_GRAYSCALE)

# Definir varios filtros (kernel) para detectar bordes horizontales,
#   verticales y diagonales.
filtros ={
    'horizontal': np.array([[-1,-1,-1],
                   [2,2,2],
                   [-1,-1,-1]]),
    'vertical': np.array([[-1,2,-1],
                   [-1,2,-1],
                   [-1,-1,-1]]),
    'diagonal':np.array([[2,-1,-1],
                   [-1,2,-1],
                   [-1,-1,2]])
                   }

# Aplicar los filtros de convolución
resultados = {}
for nombre, filtro in filtros.items():
    resultados[nombre] = cv2.filter2D(imagen, -1, filtro)

# Mostrar imagen original vs imagen con borde detectado
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

for i, (nombre, img_filtrada) in enumerate(resultados.items(), start=2):
    plt.subplot(1,4,i)
    plt.title(nombre.capitalize())
    plt.imshow(img_filtrada, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
