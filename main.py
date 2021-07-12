#Generador de  cm a metros con solo 7 entrada y una neurona 
import tensorflow as tf
import numpy as np

cm = np.array([1, -10, 0, 8, 15, 222, 380], dtype=float)
mtr = np.array([0.01, -0.1, 0.08, 0.15, 2.22, 72, 3.8], dtype=float)

capa = tf.keras.layers.Dense(units= 1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Comenzando Entrenamiento..")
historial =  modelo.fit(cm, mtr, epochs=1000, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("PREDICCION")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " Metros")