# Optimization algorithms
# Code heavily adapted from  on samples from
# "O'Reilly: Programming collective intelligence"
import math
import numpy as np
import time
import random
from mimicry import Mimic
from IPython.core.debugger import Tracer


def rhc(P, iter=200):
    """Randomized hill climbing with multiple iterations."""
    # Initialize
    best_sol = []
    best_fitness = float("-Inf")

    # Main loop
    for i in range(iter):
        # Random starting point
        start = P.random_solution()

        # Get local optima
        sol = hc(P, sol=start)

        # Better than current best solution?
        if P.fitf(sol) > best_fitness:
            best_sol = sol
            best_fitness = P.fitf(sol)
    return best_sol


def hc(P, sol=None):
    """Hill climbing with either a provided or random initial point."""
    # Default
    if sol is None:
        sol = P.random_solution()

    # Main loop
    while 1:
        # Get list of neighbors
        neighbors = P.neighbors(sol)
 
        # See what the best solution amongst the neighbors is
        current = P.fitf(sol)
        scores = map(P.fitf, neighbors)
        best = max(scores)
        best_neighbors = [n for i, n in enumerate(neighbors)
                          if scores[i] == best]

        # If there's no improvement, then we've reached the top
        if best <= current:
            break
        else:
            # Randomly choose among best neighbors
            i = random.randint(0, len(best_neighbors)-1)
            sol = best_neighbors[i]
    return sol


def rsa(P, iter=200, T=10000.0, cool=0.97):
    """Random restart simulated annealing, with -iter- restarts, -T temperature,
    -cool- cooling rate."""
    # Initialize
    best_sol = []
    best_fitness = float("-Inf")

    # Main loop
    for i in range(iter):
        # Get local optima
        sol = sa(P, T=T, cool=cool)

        # Better than current best solution?
        if P.fitf(sol) > best_fitness:
            best_sol = sol
            best_fitness = P.fitf(sol)
    return best_sol
        
    
def sa(P, T=10000.0, cool=0.97):
    """Single run simulated annealing with -T- temperature, -cool- cooling rate."""
    # Initialize with a random, feasible solution    
    vec = P.random_solution()

    while T > 0.1:
        # Choose a random neighbor
        neighbors = P.neighbors(vec)
        vecb = neighbors[np.random.randint(len(neighbors))]

        # Calculate the current cost and the new cost
        ea = P.fitf(vec)
        eb = P.fitf(vecb)
        log_p = (eb-ea)/T
        
        # Is it better, or does it make the probability cutoff?
        if (eb > ea) or (math.log(np.random.rand()) < log_p):
            vec = vecb

        # Decrease the temperature
        T = T*cool
   
    # Finish optimization with hill climbing
    vec = hc(P, vec)
     
    return vec


def ga(P, popsize=100, mutprob=0.3, elite=0.3, maxiter=100):
    """Genetic algorithm with -popsize- population size, -mutprob- mutation
    probability, -elite- elitism threshold, -maxiter- iterations. Only one 
    offspring is generated per crossover. """
     # Mutation Operation
    def mutate(vec):
        # Switching to a random neighbor is like mutation
        neighbors = P.neighbors(vec)
        i = random.randint(0, len(neighbors)-1)
        return neighbors[i] 

    # Crossover Operation
    def crossover(r1, r2):
        if hasattr(P, 'crossover'):
            # Use problem specific crossover, if it exists
            return P.crossover(r1, r2)
        else:
            i = random.randint(1, len(P.domain)-2)
        return r1[0:i]+r2[i:]

    # Build the initial population
    pop=[]
    for i in range(popsize):
        vec = P.random_solution()
        pop.append(vec)

    # How many winners from each generation?
    topelite=int(elite*popsize)

    # Main loop
    for i in range(maxiter):
        scores = [(P.fitf(v), v) for v in pop] 
        scores.sort(key=lambda x:x[0], reverse=True)
        ranked = [v for (s, v) in scores]
        # Start with the pure winners
        pop = ranked[0:topelite]
        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random() < mutprob:
                # Mutation
                c = random.randint(0, topelite)
                mutated = mutate(ranked[c])
                pop.append(mutated)
            else:
                # Single offspring crossover
                c1=random.randint(0, topelite)
                c2=random.randint(0, topelite)
                crossed = crossover(ranked[c1], ranked[c2])
                pop.append(crossed)
    
    # Bring elite individuals to their local optima via hill climbing
    pop = [hc(P, v) for v in pop]

    # Return the best individual
    scores = [(P.fitf(v), v) for v in pop] 
    scores.sort(key=lambda x:x[0], reverse=True)
    ranked = [v for (s, v) in scores]
    return ranked[0]


def mimic(P, n_iter=100, samples=50, percentile=0.9):
    """MIMIC algorithm with -n_iter_ iterations, -samples- # of samples, and
    -percentile- percentile cutoff."""
    mmc = Mimic(domain=P.domain, fitness_function=P.fitf, samples=samples,
                percentile=percentile)
    for i in range(n_iter):
        print 'i: {}'.format(i)
        sols = mmc.fit()
        print 'best fitness: {}'.format(max(map(P.fitf, sols)))
    
    # Bring valid samples to their local optima via hill climbing
    vsols = [hc(P, sol) for sol in sols if P.valid(sol)]

    # Return the best individual
    scores = [(P.fitf(v), v) for v in vsols] 
    scores.sort(key=lambda x:x[0], reverse=True)
    ranked = [v for (s, v) in scores]
    return sols[0]
