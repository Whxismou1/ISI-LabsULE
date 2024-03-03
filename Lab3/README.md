# Laboratorio 3

En este laboratorio se abordan los siguientes ejercicios:

## [Ejercicio 1](Lab3_1.py): Operaciones Matriciales
   Realiza las siguientes operaciones matriciales:

   a. Crea la siguiente matriz y almacénala en la variable `Matrix1`:
   ```
   [[4, -2, 7],[9, 4, 1],[5, -1, 5]]
   ```
   
b. Calcula su transpuesta y almacénala en la variable `Matrix2`.

c. Calcula el producto elemento a elemento de `Matrix1` y `Matrix2`.

d. Calcula el producto de las matrices `Matrix1` y `Matrix2` y almacénalo en la variable `prodM1M2`.

e. Calcula el producto de las matrices `Matrix2` y `Matrix1` y almacénalo en la variable `prodM2M1`.

f. Almacena en una matriz 2D llamada `mat_corners` las esquinas de `Matrix1` en una sola línea de código.

g. Calcula el máximo de cada fila de `Matrix1` y almacénalo en `vec_max`. Además, calcula el máximo global de `Matrix1`.

h. Calcula el mínimo de cada columna de `Matrix1` y almacénalo en `vec_min`. Además, calcula el mínimo global de `Matrix1`.

i. Calcula el producto matricial de `vec_min` y `vec_max` (en ese orden), de modo que el resultado sea una matriz de forma (3, 3).

j. Calcula la suma de los elementos de la primera y tercera columna de `Matrix1` y almacénalos en una variable llamada `mat_sum`. Hazlo en solo una línea de código.


### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab3_1.py` utilizando el intérprete de Python.


## [Ejercicio 2](Lab3_2.py): Indexación Booleana

Genera una matriz cuadrada con 400 puntos en la que cada elemento sea un valor aleatorio en el intervalo [0, 3). Imprime en la pantalla:

a. La matriz original.

b. Las coordenadas de los elementos cuyo valor está entre 1 y 2.

c. Las coordenadas de los elementos que son menores que 1 o mayores que 2.

d. Redondea la matriz generada y luego imprime las coordenadas de los valores que son diferentes de 1 en la matriz redondeada.

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab3_2.py` utilizando el intérprete de Python.




## [Ejercicio 3](Lab3_3.py): Distancia entre Elementos

Genera una matriz con dimensiones 10x4 cuyos elementos sean valores aleatorios en el intervalo [-10, 10). Luego, considerando que cada fila es un punto en un espacio 4D, realiza lo siguiente:

a. Crea una matriz de distancia euclidiana. Es una matriz con el mismo número de filas y columnas que el número de puntos considerados (tenemos 10 puntos, por lo que las dimensiones en este caso son 10x10). Un elemento dij en dicha matriz es la distancia euclidiana entre los puntos xi y xj (es decir, las filas i y j en nuestra matriz 10x4).

b. Indica la distancia entre los puntos, cuando dicha distancia sea menor que 10, y los números de los puntos (es decir, las filas donde se encuentran los puntos) con el mensaje: "La distancia euclidiana entre los vectores i y j es X", cambiando i, j y X por los números de los puntos y la distancia.

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab3_3.py` utilizando el intérprete de Python.



## [Ejercicio 4](Lab3_4.py): Optimización de Código

Escribe una función que calcule los cuadrados de los N primeros números naturales (donde N es el parámetro de entrada de la función) de tres formas diferentes:

a. Almacenando los valores en una lista.

b. Almacenando los valores en un arreglo unidimensional de NumPy previamente asignado (por ejemplo, creándolo con `np.zeros()`), pero calculando el cuadrado de cada elemento en un bucle for.

c. Vectorizando la operación.

Calcula, utilizando `timeit`, el tiempo promedio necesario para calcular los cuadrados con cada método y muestra esos tiempos en la pantalla. Prueba con valores grandes de N (por ejemplo, 100000). ¿Cuáles son los tiempos?

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab3_4.py` utilizando el intérprete de Python.



---
Este README.md proporciona una guía completa para entender y ejecutar las operaciones requeridas en el Laboratorio 4. Cada sección describe detalladamente las operaciones a realizar y cómo ejecutar el código asociado a cada una.
