from matplotlib import pyplot as plt


def calculateCenterOfMass(particles):
    totalMass = 0
    xCenterOfMass = 0
    yCenterOfMass = 0
    for particle in particles:
        totalMass += particle['mass']
        xCenterOfMass += particle['x'] * particle['mass']
        yCenterOfMass += particle['y'] * particle['mass']
    xCenterOfMass /= totalMass
    yCenterOfMass /= totalMass
    return {'x': xCenterOfMass, 'y': yCenterOfMass, 'mass': totalMass}


def drawParticles(centerOfMass, particles):
    plt.figure(figsize=(8, 6))
    for p in particles:
        # , label=f"Mass: {p['mass']}")
        plt.scatter(p['x'], p['y'], s=100, color='blue')
        plt.text(p['x']+0.05, p['y'],
                 f"{float(p['mass'])}", fontsize=7.5, color='black')
    # label=f"Center of Mass")
    plt.scatter(centerOfMass['x'], centerOfMass['y'], marker='^', color='red')
    plt.text(centerOfMass['x']+0.05, centerOfMass['y'],
             f"{float(centerOfMass['mass'])}", fontsize=7.5, color='black')
    plt.grid(True)
    plt.show()


def main():
    numParticles = int(input("Insert the number of particles: "))
    # particlesCoords = [{'x': 0, 'y': 0, 'mass': 1}, {'x': 1, 'y': 1, 'mass': 5}, {'x': 2, 'y': 0, 'mass': 15}]
    particlesCoords = []
    for i in range(numParticles):
        x = float(input("Particle " + str(i + 1) + ". Position x: "))
        y = float(input("Particle " + str(i + 1) + ". Position y: "))
        mass = float(input("Particle " + str(i + 1) + ". Mass: "))
        particlesCoords.append({'x': x, 'y': y, 'mass': mass})

    coordsCenterOfMass = calculateCenterOfMass(particlesCoords)

    drawParticles(coordsCenterOfMass, particlesCoords)


if __name__ == "__main__":
    main()
