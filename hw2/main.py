# Script to analyze performance of optimization algorithms on several problems

import matplotlib.pyplot as plt
import numpy as np
import pickle
from myopt import *  # Optimization algorithms
from probs import *  # Problems to optimize
from time import time


class Stats():
    """Simple contaner for holding stats."""
    def __init__(self, sols, fits, runt):
        self.sols = sols
        self.fits = fits
        self.runt = runt


def get_stats(prob=None, alg=None, iter=3):
    """ Get performance statistics for an optimization algorithm
    on a particular problem. """ 
    P = prob

    # Get performance statistics
    sols = []  # Solutions
    fits = []  # Fitness scores
    runt = []  # Runtimes
    for i in range(iter):
        start = time()
        sol =  alg(P)
        runt.append(time() - start)
        sols.append(sol)
        fits.append(P.fitf(sol))

    print "Median fitness: {}".format(np.median(fits))
    print "Mean fitness: {}".format(np.mean(fits))
    print "Min fitness: {}".format(np.min(fits))
    print "Max fitness: {}".format(np.max(fits))
    print "Mean runtime: {}".format(np.mean(runt))

    # Return solutions, fitness scores, and runtimes
    return Stats(sols, fits, runt)

    
def compare_algorithms(prob=None, algorithms=None, iterations=None, 
                       labels=None):
    """Collects performance statistics for a set of algorithms on a particular
    problem. Algorithms are applied for a set number of iterations. """
    P = prob
    # Get performance statistics
    sols = []
    fits = []
    runt = []
    for alg, iter, label in zip(algorithms, iterations, labels):
        print ' '.join(['#'*10, label, '#'*10])
        alg_stats = get_stats(prob=P, alg=alg, iter=iter)
        sols.append(alg_stats.sols) 
        fits.append(alg_stats.fits)
        runt.append(alg_stats.runt)
        print ''

    # Return solutions, fitness scores, and runtimes for each algorithm.
    return sols, fits, runt, labels


def analyze_four_peaks():    
    """Analyzes the performance of all the algorithms on the Four Peaks
     problem. """
    with open('pickles/four_peaks.pickle') as f:
        sols, fits, runt, labels = pickle.load(f)    

    # Plot fitness scores of different algorithms 
    plt.figure()
    plt.boxplot(fits, labels=labels)
    plt.ylim(90,200)
    plt.title('Four Peaks - Distribution of fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('Four Peaks - Fitness scores.png')

    # Plot run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Four Peaks - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('Four Peaks - Runtimes.png')


def analyze_knapsack():    
    """Analyzes the performance of all the algorithms on the Knapsack
     problem. """
    # Load performance statistics
    with open('pickles/knapsack.pickle') as f:
        sols, fits, runt, labels = pickle.load(f)    

    # Compare fitness scores of different algorithms 
    plt.figure()
    plt.boxplot(fits, labels=labels)
    plt.title('Knapsack - Distribution of fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('Knapsack - Fitness scores.png')

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Knapsack - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('Knapsack - Runtimes.png')


def analyze_k_colors():    
    """Analyzes the performance of all the algorithms on the K Colors
     problem. """
    # Load performance statistics
    with open('pickles/kcolors.pickle') as f:
        sols, fits, runt, labels = pickle.load(f)    

    # Compare fitness scores of different algorithms 
    plt.figure()
    plt.boxplot(fits, labels=labels)
    plt.title('K colors - Distribution of fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('K colors - Fitness scores.png')

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('K colors - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('K colors - Runtimes.png')


def analyze_neural_network():    
    """Analyzes the performance of all the algorithms on the K Colors
     problem. """
    # Load performance statistics
    with open('pickles/ann.pickle') as f:
        sols, fits, runt, labels = pickle.load(f)    

    # Calculate test fitness scores
    
    # Compare training fitness scores of different algorithms 
    plt.figure()
    plt.boxplot(fits, labels=labels)
    plt.title('Neural network - Distribution of fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('Neural network - Fitness scores.png')

    # Compare test fitness scores of different algorithms
    plt.figure()

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Neural network - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('Neural network - Runtimes.png')


def demo_problem(prob=None):
    """Demo performance of all algorithms on a single problem. 
    Only a single iteration."""
    algorithms = [hc, sa, ga, mimic]
    labels = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'MIMIC']
    for alg, label in zip(algorithms, labels):
        print ' '.join(['#'*10, label, '#'*10])
        alg_stats = get_stats(prob=prob, alg=alg, iter=1)
        print ''

        
def demo_alg(opt_alg=None):
    """Demo for a single optimization algorithm on multiple problems."""
    print '########## 4-peaks problem ##########'
    P = FourPeaks()
    sol = opt_alg(P)
    print "solution: {}\nfitness: {}".format(sol, P.fitf(sol))
    print ''

    print '########## 6-peaks problem ##########'
    P = SixPeaks()
    sol = opt_alg(P)
    print "solution: {}\nfitness: {}".format(sol, P.fitf(sol))
    print ''

    print '########## K-colors problem ##########'
    P = KColors()
    sol = opt_alg(P)
    print "solution: {}\nfitness: {}".format(sol, P.fitf(sol))
    print ''
    
    print '########## Traveling salesman problem ##########'
    P = TSP()
    sol = opt_alg(P)
    print "solution: {}\nfitness: {}".format(sol, P.fitf(sol))
    print ''

    print '########## Knapsack problem ##########'
    P = Knapsack()
    sol = opt_alg(P)
    print "solution: {}\nfitness: {}".format(sol, P.fitf(sol))
    print ''


def main():
    # Part 1: Get performance data for all 4 optimization algorithms on
    # all problems except ANN weight selection. Pickle results.
    algs = [hc, sa, ga, mimic]
    iter = [200, 200, 20, 10]
    lbls = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm',
            'MIMIC']
    # Four peaks
    fs = FourPeaks()
    sols, fits, runt, labels = compare_algorithms(prob=fs,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('four_peaks.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels), f)

    # Knapsack
    ks = Knapsack()
    sols, fits, runt, labels = compare_algorithms(prob=ks,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('knapsack.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels), f)

    # K colors
    kc = KColors()
    sols, fits, runt, labels = compare_algorithms(prob=kc,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('kcolors.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels), f)
    
    # Part 2: Get performance data for HC, SA, and GA optimization
    # algorithms on the ANN weight selection problem. Pickle results.
    algs = [hc, sa, gc]
    iter = [200, 200, 20]
    lbls = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
    
    ann = ANN()
    sols, fits, runt, labels = compare_algorithms(prob=ann,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('ann.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels, ann), f)

if __name__ == '__main__':
    main()
