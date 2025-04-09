import numpy as np

def QSHO(fitness, bounds, num_particles=10, iterations=50):
    best_global = None
    best_score = float('inf')

    particles = np.random.uniform(bounds[:,0], bounds[:,1], (num_particles, len(bounds)))
    
    for _ in range(iterations):
        for i, particle in enumerate(particles):
            score = fitness(particle)
            if score < best_score:
                best_global = particle
                best_score = score

            delta = np.random.rand() - 0.5
            particles[i] = best_global + delta * (bounds[:,1] - bounds[:,0]) * np.random.rand()

    return best_global
