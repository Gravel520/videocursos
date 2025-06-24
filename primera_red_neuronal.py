'''
Scripts en Python.
En este scripts vamos a utilizar la programación de aprendizaje automático.
El ejemplo que vamos a utilizar es una conversión de grados celsius a 
fahrenheit. 
Se podría hacer mediante programación regular, creando una
función que recibiera el código de los grados en celsius, y después de
hacer una pequeña operación matemática (F = C * 1.8 + 32), devolveríamos
la cantidad en grados fahrenheit.
En aprendizaje automática no se conoce el algoritmo u operación matemática
que se usará para hacer la conversión. En lugar de esto, tenemos una lista
con los datos de grados en celsius (entrada), y otra lista con los grados
en fahrenheit (salida), la conversión se hará mediante una red neuronal.

Las capas de tipo 'Densa', son las que tienen conexsión desde una neurona
    a todas las capas. También le decimos las neuronas de salida y las
    neuronas de entrada.
    Usamos un modelo de capa 'secuencial', que es más sencillo.

Para compilar el modelo, definimos sólo dos parámetros; el primero es el
    optimizador, que le dira al modelo como ajustar los pesos y sesgos
    de manera eficiente para ir aprendiendo y no desaprendiendo, el valor
    de este parámetro le indicará como irá afinando ese aprendizaje. En el
    otro parámetro especificamos la función de pérdida.

Después entrenamos el modelo. Para ello tenemos que definir que valores
    son las entradas, el arreglo de los celsius, y cuales deben ser los
    valores de salida, el arreglo de los fahrenheid. También tenemos
    que especificar cuantas vueltas tendrá que dar para conseguir los
    resultados esperados, cuantas menos vueltas de menos entrenará, pero
    si damos demasiadas se eternizará, hay que buscar una media.
    El último parámetro es para que vaya imprimiendo el entrenamiento.

Finalmente podemos comprobar los valores que utilizó la red neuronal
    para llegar a la conversión correcta de unos grados a otros. El 
    primer valor es el peso, y el segundo el sesgo.

'''

import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt

# Creamos los arreglos que corresponden a los grados celsius y fahrenheit.
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

'''
# Creamos la red neuronal de tipo 'Densa', con una capa de entrada o
#   otra de salida.
capa = keras.layers.Dense(units=1, input_shape=[1])
modelo = keras.Sequential([capa])
'''

# Creamos la red neuronal de tipo 'Densa', con una capa de entrada, con
#   una neurona, dos capas intermedias, ocultas, con tres neuronas
#   cada una, y finalmente una capa de salida con una neurona.
oculta1 = keras.layers.Dense(units=3, input_shape=[1]) # Capa entrada y oculta1
oculta2 = keras.layers.Dense(units=3) # Capa oculta2
salida = keras.layers.Dense(units=1) # Capa salida
modelo = keras.Sequential([oculta1, oculta2, salida])

# Preparamos al modelo para ser entrenado.
modelo.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamos el modelo.
print('Comenzando entrenamiento...')
historial = modelo.fit(celsius, fahrenheit, epochs=300, verbose=False)
print('Modelo entrenado!')

# Graficamos la función de perdida del modelo.
plt.xlabel('# Vueltas')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])
plt.show()

# Hacemos una predicción.
print('Hagamos una predicción!')
x = np.array([[100]])
resultado = modelo.predict(x)
print(f'El resultado es {str(resultado)} grados fahrenheit')

# Presentamos los valores del sesgo y el peso.
print('\nVariables internas del modelo')
#print(capa.get_weights())

