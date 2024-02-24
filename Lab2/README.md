# Laboratorio 2

En este laboratorio se abordan los siguientes ejercicios:

## [Ejercicio 1](Exercise1/Lab2_1.py): Juego de Adivinanzas
1. Programa un juego que:
   a. Pida al usuario su nombre y un número entero N.
   b. Luego, el programa genera un número entre 1 y N, y le pide al usuario que lo adivine insertándolo en la pantalla. Dependiendo del número máximo, N, el usuario tendrá un número máximo de intentos.
   c. Si el número insertado por el usuario no es el que generó el programa, le dirá al usuario si el número es mayor o menor, y el número de intentos restantes.
   d. Si la adivinanza del usuario es correcta, entonces recibirá varios puntos, que se mostrarán al usuario.
Ten en cuenta lo siguiente:
   - El número máximo de intentos que tiene un usuario para adivinar el número es:
     𝑡𝑟𝑖𝑎𝑙𝑠 = ⌈𝑁 ⌉
   - El número máximo de puntos por usuario se guardará en un archivo. Por lo tanto, si el usuario obtiene varios puntos más altos que su puntaje más alto, se guardan.
   - El programa debe decirle al usuario al principio del juego cuál ha sido su puntaje más alto (si lo hubiera).
   - El programa debe dar la posibilidad al usuario de ver los puntajes más altos de todos los usuarios.
   - Sea f el número de errores antes de que el usuario adivine el número. Entonces, los puntos que recibe son:
     𝑝𝑜𝑖𝑛𝑡𝑠 = 𝑁
                   2𝑓
   - Al final de un juego, el usuario puede elegir si jugar de nuevo o salir.

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab2_1.py` utilizando el intérprete de Python.


## [Ejercicio 2](Exercise2/Lab2_2.py): Clase Vector3D
2. Escribe una clase llamada Vector3D que permita:
   - Inicializar un vector con algunas coordenadas dadas.
   - Cambiar el valor de las coordenadas.
   - Mostrar las coordenadas en la pantalla.
   - Agregar un objeto de vector dado al vector actual.
   - Restar el vector menos un objeto de vector dado.
   - Multiplicar el vector por un escalar dado.
   - Devolver el módulo del vector.
   - Almacenar los valores del vector en un archivo txt.
   - Almacenar los valores del vector en un archivo pickle.

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab2_2.py`  utilizando el intérprete de Python.

## [Ejercicio 3](Exercise3/Lab2_3.py): Calculadora de Operaciones Matemáticas
3. Escribe un programa que permita al usuario:
   - Agregar un número arbitrario de valores.
   - Restar dos valores.
   - Multiplicar un número arbitrario de valores.
   - Dividir dos valores.
   - Calcular el valor de un número elevado a otro.
   - Calcular el logaritmo natural de un número.
Cada operación debe ser una función. Luego, en el programa principal (ver Sección 3.4), deja que el usuario elija qué operación hacer, solicita al usuario los valores y finalmente muestra el resultado.
El proceso se repite hasta que el usuario elija salir del programa.

### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab2_3.py` utilizando el intérprete de Python.

## [Ejercicio 4](Exercise4/Lab2_4.py): Acceso a un Archivo en Internet
4. Escribe un programa que acceda a un archivo en Internet utilizando su URL (puede ser un sitio web con texto) y muestre en la pantalla el número de palabras que contiene, o un mensaje que muestre que la URL no existe, si hay un error al acceder al archivo.
Para descargar un archivo de Internet, verifica la función urlopen(), del módulo urllib.request.
Comprueba el programa con la URL: https://www.gutenberg.org/cache/epub/1184/pg1184.txt. Debería tener 464025 palabras.


### Instrucciones de Ejecución
Para ejecutar el programa, sigue estos pasos:
1. Abre una terminal.
2. Navega hasta la carpeta del Laboratorio 2.
3. Ejecuta el archivo `Lab2_4.py` utilizando el intérprete de Python.
