import numpy as np
import matplotlib.pyplot as plt

numRows = 3

# Crear una figura con 3 subgráficos
fig, ax = plt.subplots(numRows, 1, figsize=(10, 6))

angulo = np.linspace(0, 20, 1000)

# Configurar los ejes según las especificaciones para todas las figuras
for i in range(numRows):
    ax[i].set_xlim(0, 20)
    ax[i].set_ylim(-1.0, 1.0)
    ax[i].set_xticks(np.arange(0, 21, 2.5))
    ax[i].set_yticks(np.arange(-1.0, 1.25, 0.5))


# Calcular el coseno de los ángulos
cosContinued = np.cos(angulo)
cosDiscontinuo = np.cos(angulo + (np.pi/4))

# Calcular la tangente
tanContinuo = np.tan(angulo) * 0.05
tanDiscontinuo = np.tan(angulo + (np.pi/4)) * 0.05

# Calcular el seno de los ángulos
sin = np.sin(angulo)
sinDiscontinuo = np.sin(angulo + (np.pi/4))


for i in range(numRows):
    if i == 0:
        # coseno
        # Graficar la función continua en rojo
        ax[0].plot(angulo, cosContinued, color='red', label='cos(α)')
        # Graficar la función discontinua en negro y con línea discontinua
        ax[0].plot(angulo, cosDiscontinuo, color='black',
                   linestyle='--', label='cos(α + π/4)')
        # Etiquetas y título
        ax[0].set_title('Cosine function')
        # Mostrar la leyenda

        ax[0].legend(loc="lower right")

    elif i == 1:
        # seno
        # Graficar la función continua en rojo
        ax[1].plot(angulo, sin, color='red', label='sin(α)')
        # Graficar la función discontinua en negro y con línea discontinua
        ax[1].plot(angulo, sinDiscontinuo, color='black',
                   linestyle='--', label='sin(α + π/4)')
        # Etiquetas y título
        ax[1].set_title('Sine function')
        # Mostrar la leyenda
        ax[1].legend(loc="upper right")
    elif i == 2:
        # Tangente
        # Graficar la función continua en rojo
        ax[2].plot(angulo, tanContinuo, color='red', label='tan(α)')
        # Graficar la función discontinua en negro y con línea discontinua
        ax[2].plot(angulo, tanDiscontinuo, color='black',
                   linestyle='--', label='tan(α + π/4)')
        # Etiquetas y título
        ax[2].set_title('Tangent function')
        # Mostrar la leyenda
        ax[2].legend(loc="upper right")

# Mostrar el gráfico
#rect=[0.1, 0.1, 0.9, 0.9]

plt.tight_layout()
plt.show()
