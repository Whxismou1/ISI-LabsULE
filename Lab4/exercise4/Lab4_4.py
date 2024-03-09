from matplotlib import patches
import pandas as pd
import matplotlib.pyplot as plt


def draw3DPlot(x, y, z, shape, dfIris):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Setosa': 'purple', 'Versicolor': 'blue', 'Virginica': 'orange'}

    for i in range(len(x)):
        variety = dfIris['variety'][i]

        ax.scatter(x[i], y[i], z[i], s=100, marker='o', alpha=0.5,
                   label=variety, edgecolors='k', c=colors[variety])

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    ax.set_title('Scatter plot of the iris dataset')

    legends = []
    for variety, color in colors.items():
        # legends.append(patches.Circle(color=color, label=variety))
        circle = patches.Circle((0, 0), radius=1, color=color, label=variety)
        legends.append(circle)
        
    ax.legend(handles=legends)

    plt.show()


def main():
    ruta = "./Lab4/exercise4/iris.csv"
    irisDF = pd.read_csv(ruta)
    x = irisDF['sepal_length']
    y = irisDF['sepal_width']
    z = irisDF['petal_length']
    shape = irisDF['petal_width']

    draw3DPlot(x, y, z, shape, irisDF)


if __name__ == "__main__":
    main()
