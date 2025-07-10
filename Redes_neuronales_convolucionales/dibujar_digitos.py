'''
Script en Python.
Vamos a construir una pizarra de dibujo digital con PySide6. Esta estructura te
permitirá dibujar con el mouse, borrar, y capturar lo que has hecho para usarlo
en una red neuronal.
Contamos con dos predicciones, una con una red neuronal secuencial, y la otra
con una red neuronal convolucional con aumento de datos y dropout, que esta
mejor entrenada.
'''

# Importamos las librerías.
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QDialog
from PySide6.QtGui import QPainter, QPixmap, QPen, QColor, QMouseEvent
from PySide6.QtCore import Qt, QPoint
from PIL import Image, ImageQt
from keras.models import load_model
import sys
import numpy as np

class ZonaDibujo(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280) # Tamaño grande, pero escalable a 28x28 luego
        self.imagen = QPixmap(self.size())
        self.imagen.fill(Qt.white)
        self.dibujando = False
        self.ultimo_punto = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.imagen)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dibujando = True
            self.ultimo_punto = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dibujando:
            painter = QPainter(self.imagen)
            pen = QPen(QColor(0, 0, 0), 18, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            punto_nuevo = event.position().toPoint()
            painter.drawLine(self.ultimo_punto, punto_nuevo)
            self.ultimo_punto = punto_nuevo
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dibujando = False

    def limpiar(self):
        self.imagen.fill(Qt.white)
        self.update()

    def obtener_pixmap(self):
        return self.imagen
    
class VentanaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dibuja un dígito')
        self.zona_dibujo = ZonaDibujo()
        self.boton_limpiar = QPushButton('Limpiar')
        self.boton_predecir = QPushButton('Predecir')
        self.resultado = QLabel('')
        self.imagen_mostrada = QLabel()
        self.imagen_mostrada.setFixedSize(28, 28)

        layout = QVBoxLayout()
        layout.addWidget(self.zona_dibujo)
        layout.addWidget(self.boton_predecir)
        layout.addWidget(self.boton_limpiar)
        layout.addWidget(self.resultado)
        layout.addWidget(self.imagen_mostrada)

        self.setLayout(layout)

        self.boton_limpiar.clicked.connect(self.zona_dibujo.limpiar)
        self.boton_predecir.clicked.connect(self.predecir)

    def predecir(self):        
        pixmap = self.zona_dibujo.obtener_pixmap()

        # Convertir Qpixmap en array numpy
        imagen_qt = pixmap.toImage()
        imagen_bytes = imagen_qt.bits().tobytes()
        img_np = np.frombuffer(imagen_bytes, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), 4))

        # Convertir a escala de grises con PIL
        # 'L' = escala de grises
        img_pil = Image.fromarray(img_np).convert('L')

        # Redimensionar a 28x28
        img_pil = img_pil.resize((28, 28), Image.Resampling.NEAREST)

        # Convertir a array numpy y normalizar
        img_array = np.array(img_pil).astype('float32') / 255.0
        img_array = img_array.reshape((1, 28, 28, 1))

        img_array = 1.0 - img_array

        # Cargar el modelo
        modelo_aumento = load_model('C:/Users/katal/Documents/Python/Videocursos/Redes_neuronales_convolucionales/modelo_digitos_aumento_datos.keras')
        modelo = load_model('C:/Users/katal/Documents/Python/Videocursos/Redes_neuronales_convolucionales/modelo_digitos.keras')

        # Mostrar la imagen redimensionada antes de la predicción
        img_pil_mostrar = img_pil
        mg_qt = img_pil_mostrar.toqpixmap() if hasattr(img_pil_mostrar, 'toqpixmap') else QPixmap.fromImage(ImageQt(img_pil_mostrar))
        self.imagen_mostrada.setPixmap(mg_qt)

        import matplotlib.pyplot as plt
        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title('Imagen enviada al modelo')
        plt.axis('off')
        #plt.show()
        plt.close('all')

        # Realizar las predicciones
        prediccion = modelo.predict(img_array)
        digito = np.argmax(prediccion)
        prediccion_aumento = modelo_aumento.predict(img_array)
        digito_aumento = np.argmax(prediccion_aumento)

        # Mostrar el resultado
        self.resultado.setText(f'Resultado: {digito}    -   Resultado Aumento: {digito_aumento}')

    def mostrar_en_ventana(self, pixmap):
        ventana_imagen = QDialog()
        ventana_imagen.setWindowTitle('Dibujo capturado')

        label = QLabel()
        label.setPixmap(pixmap.scaled(280, 280, Qt.KeepAspectRatio))

        layout = QVBoxLayout()
        layout.addWidget(label)
        ventana_imagen.setLayout(layout)
        ventana_imagen.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())
