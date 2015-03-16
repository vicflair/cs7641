# Script to analyze performance of optimization algorithms on several problems

import matplotlib.pyplot as plt
import numpy as np
import pickle
from myopt import *  # Optimization algorithms
from probs import *  # Problems to optimize
from time import time
from functools import partial

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
        print 'iter: {}'.format(i)
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
    plt.savefig('plots/Four Peaks - Fitness scores.png')

    # Plot run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Four Peaks - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('plots/Four Peaks - Runtimes.png')

    # Print training fitness statistics
    print '#'*10 + ' Training fitness ' + '#'*10
    for fit, lbl in zip(fits, labels):
        print lbl
        print '# Runs: {}'.format(len(fit))
        print 'Min: {}'.format(min(fit))
        print 'Max: {}'.format(max(fit))
        print 'Mean: {}'.format(np.mean(fit))
        print 'Median: {}'.format(np.median(fit))
        print ''

    # Print run time statistics
    print '#'*10 + ' Run times ' + '#'*10
    for run, lbl in zip(runt, labels):
        print lbl
        print '# Runs: {}'.format(len(run))
        print 'Min: {}'.format(min(run))
        print 'Max: {}'.format(max(run))
        print 'Mean: {}'.format(np.mean(run))
        print 'Median: {}'.format(np.median(run))
        print ''


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
    plt.savefig('plots/Knapsack - Fitness scores.png')

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Knapsack - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('plots/Knapsack - Runtimes.png')

    # Print training fitness statistics
    print '#'*10 + ' Training fitness ' + '#'*10
    for fit, lbl in zip(fits, labels):
        print lbl
        print '# Runs: {}'.format(len(fit))
        print 'Min: {}'.format(min(fit))
        print 'Max: {}'.format(max(fit))
        print 'Mean: {}'.format(np.mean(fit))
        print 'Median: {}'.format(np.median(fit))
        print ''

    # Print run time statistics
    print '#'*10 + ' Run times ' + '#'*10
    for run, lbl in zip(runt, labels):
        print lbl
        print '# Runs: {}'.format(len(run))
        print 'Min: {}'.format(min(run))
        print 'Max: {}'.format(max(run))
        print 'Mean: {}'.format(np.mean(run))
        print 'Median: {}'.format(np.median(run))
        print ''


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
    plt.savefig('plots/K colors - Fitness scores.png')

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('K colors - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('plots/K colors - Runtimes.png')

    # Print training fitness statistics
    print '#'*10 + ' Training fitness ' + '#'*10
    for fit, lbl in zip(fits, labels):
        print lbl
        print '# Runs: {}'.format(len(fit))
        print 'Min: {}'.format(min(fit))
        print 'Max: {}'.format(max(fit))
        print 'Mean: {}'.format(np.mean(fit))
        print 'Median: {}'.format(np.median(fit))
        print ''

    # Print run time statistics
    print '#'*10 + ' Run times ' + '#'*10
    for run, lbl in zip(runt, labels):
        print lbl
        print '# Runs: {}'.format(len(run))
        print 'Min: {}'.format(min(run))
        print 'Max: {}'.format(max(run))
        print 'Mean: {}'.format(np.mean(run))
        print 'Median: {}'.format(np.median(run))
        print ''


def analyze_neural_network():    
    """Analyzes the performance of all the algorithms on the K Colors
     problem. """
    # Load performance statistics
    with open('pickles/ann.pickle') as f:
        sols, fits, runt, labels, ann = pickle.load(f)    

    # Calculate and plot test fitness scores
    tfits = [map(lambda x: ann.fitf(x, train=False), sol) for sol in sols]
    plt.figure()
    plt.boxplot(tfits, labels=labels)
    plt.title('Neural network - Distribution of test fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('plots/Neural network - Test set fitness scores.png')
    
    # Plot training fitness scores of different algorithms 
    plt.figure()
    plt.boxplot(fits, labels=labels)
    plt.title('Neural network - Distribution of training fitness scores by algorithm')
    plt.ylabel('Fitness score')
    plt.savefig('plots/Neural network - Training set fitness scores.png')

    # Compare run times of different algorithms
    plt.figure()
    plt.boxplot(runt, labels=labels)
    plt.title('Neural network - Distribution of runtimes by algorithm')
    plt.ylabel('Runtime [secs]')
    plt.savefig('plots/Neural network - Runtimes.png')

    # Test backpropagation
    ann.network.randomize()
    for i in range(200):
        ann.train_network()
        x = ann.network.params
        print 'iter: {}'.format(i)
        print 'backprop training fitness: {}'.format(ann.fitf(x))
        print 'backprop test fitness: {}'.format(ann.fitf(x, train=False))

    # Print test fitness statistics
    print '#'*10 + ' Test fitness ' + '#'*10
    for tfit, lbl in zip(tfits, labels):
        print lbl
        print '# Runs: {}'.format(len(tfit))
        print 'Min: {}'.format(min(tfit))
        print 'Max: {}'.format(max(tfit))
        print 'Mean: {}'.format(np.mean(tfit))
        print 'Median: {}'.format(np.median(tfit))
        print ''

    # Print training fitness statistics
    print '#'*10 + ' Training fitness ' + '#'*10
    for fit, lbl in zip(fits, labels):
        print lbl
        print '# Runs: {}'.format(len(fit))
        print 'Min: {}'.format(min(fit))
        print 'Max: {}'.format(max(fit))
        print 'Mean: {}'.format(np.mean(fit))
        print 'Median: {}'.format(np.median(fit))
        print ''

    # Print run time statistics
    print '#'*10 + ' Run times ' + '#'*10
    for run, lbl in zip(runt, labels):
        print lbl
        print '# Runs: {}'.format(len(run))
        print 'Min: {}'.format(min(run))
        print 'Max: {}'.format(max(run))
        print 'Mean: {}'.format(np.mean(run))
        print 'Median: {}'.format(np.median(run))
        print ''


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
    iter = [200, 200, 20, 5]
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
    print 'KColors'
    kc = KColors()
    sols, fits, runt, labels = compare_algorithms(prob=kc,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('kcolors.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels), f)
    
    # Part 2: Get performance data for HC, SA, and GA optimization
    # algorithms on the ANN weight selection problem. Pickle results.
    algs = [hc, sa, ga]
    iter = [200, 200, 40]
    lbls = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
    
    ann = ANN()
    sols, fits, runt, labels = compare_algorithms(prob=ann,
        algorithms=algs, iterations=iter, labels=lbls)
    with open('ann.pickle', 'w') as f:
        pickle.dump((sols, fits, runt, labels, ann), f)

    # Part 3: Plot and print results
    analyze_four_peaks()
    analyze_knapsack()
    analyze_k_colors()
    analyze_neural_network()

if __name__ == '__main__':
    main()
