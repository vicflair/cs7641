# From O'Reilly: Programming collective intelligence
import math
import networkx
import numpy as np
import time
import random
from itertools import combinations
from itertools import groupby
from itertools import izip
from IPython.core.debugger import Tracer
from scipy.stats import bernoulli


class FourPeaks():
    def __init__(self, N=100, R=100, T=10):
        self.N = N
        self.domain = [(0,1)]*N
        self.R = R
        self.T = T

    def random_solution(self):
        return [random.randint(0,1) for i in range(self.N)]
        
    def neighbors(self, x):
        # Create list of neighboring solutions
        neighbors = []
        for j in range(len(self.domain)):
            # Flip bits
            if x[j] == 1:
                neighbors.append(x[0:j] + [0] + x[j+1:])
            else:
                neighbors.append(x[0:j] + [1] + x[j+1:])
        return neighbors

    def valid(self, x):
        return set(x) == {0, 1}

    def o(self, x):
        if x[0] == 0:
            return 0
        else:
            ones = (list(it) for contains1, it in groupby(x, lambda x:x==1))
            return len(ones.next())

    def z(self, x):
        if x[-1] == 1:
            return 0
        else:
            x = reversed(x)
            zeros = (list(it) for contains0, it in groupby(x, lambda x:x==0))
            return len(zeros.next())

    def fitf(self, x):
        if (self.o(x) > self.T) & (self.z(x) > self.T):
            reward = self.R
        else:
            reward = 0
        return max(self.o(x), self.z(x)) + reward


class SixPeaks(FourPeaks):
    def fitf(self, x):
        y = x[::-1]
        if ((self.o(x) > self.T) and (self.z(x) > self.T)) or \
            ((self.o(y) > self.T) and (self.z(y) > self.T)):
            reward = self.R
        else:
            reward = 0
        return max(self.o(x), self.z(x), self.o(y), self.z(y)) + reward


class KColors():
    def __init__(self, N=25, K=4):
        self.N = N
        self.K = K
        self.domain = [(0, K-1)]*N
        g = networkx.gnp_random_graph(N, (2*K-0.)/N)
        while not networkx.is_connected(g):
            g = networkx.gnp_random_graph(N, (2*K-0.)/N)
        self.graph = g

    def random_solution(self):
        return [random.randint(0, self.K-1)
                for i in range(len(self.domain))]

    def neighbors(self, x):
        neighbors = []
        for i in range(len(self.domain)):
            for j in range(self.K):
                if j != x[i]:
                    neighbors.append(x[0:i] + [j] + x[i+1:])
        return neighbors

    def fitf(self, x):
        return -sum([x[a] == x[b] for a, b in self.graph.edges()])
        

class TSP():
    def __init__(self, N=25):
        self.N = N
        self.domain = [(0, N-1)]*N
        m = np.random.randint(1, N*3, size=(N,N))
        self.dist = (m + m.T + np.diag(m.diagonal()))/2

    def random_solution(self):
        return list(np.random.permutation(range(self.N)))

    def neighbors(self, x):
        # All possible pairwise swaps
        pairs = combinations(x, r=2)
        # All possible permutations of Hamming distance 1 away
        neighbors = []
        for p in pairs:
            y = x[:]
            y[p[0]], y[p[1]] = y[p[1]], y[p[0]]
            neighbors.append(y)
        return neighbors

    def valid(self, x):
        distinct_cities = (len(set(x)) == len(x))
        all_cities = (set(x) == set(range(self.N)))

        if all_cities & distinct_cities:
            return True
        else:
            return False

    def fitf(self, x):
        fitness = sum([self.dist[first, second]
                       for first, second in izip(x, x[1:] +  [x[0]])])
        return fitness


class Knapsack():
    def __init__(self, N=50, max_weight=100, alpha=5, P=0.2):
        self.domain = [(0, 1)]*N
        self.N = N
        self.max_weight = max_weight
        self.alpha = alpha
        self.P = P
        self.weights = np.random.randint(1, max_weight/alpha, size=N)
        self.values = np.random.randint(1, max_weight, size=N)

    def random_solution(self):
        sol = []
        while not self.valid(sol):
            sol = list(bernoulli.rvs(self.P, size=self.N))
        return sol

    def get_items(self, x):
        return [i for i, e in enumerate(x) if e != 0]

    def total_weight(self, x):
        return sum([self.weights[i] for i in self.get_items(x)])

    def total_value(self, x):
        return sum([self.values[i] for i in self.get_items(x)])

    def valid(self, x):
        in_domain = (set(x) == {0,1})
        len_N = (len(x) == self.N)
        if in_domain and len_N and self.total_weight(x) <= self.max_weight:
            return True
        else:
            return False 

    def neighbors(self, x):
        # Create list of neighboring solutions
        neighbors = []
        for j in range(len(self.domain)):
            # Flip bits
            if x[j] == 1:
                neighbors.append(x[0:j] + [0] + x[j+1:])
            else:
                neighbors.append(x[0:j] + [1] + x[j+1:])
        neighbors = filter(lambda x: self.valid(x), neighbors)
        return neighbors
        
    def fitf(self, x):
       return self.total_value(x)
    

class SquareFunc():
    def __init__(self):
        self.domain = [(-100, 100)]*10
        self.C = [random.randint(*self.domain[i])
                  for i in range(len(self.domain))]

    def random_solution(self):
        return [np.random.randint(*self.domain[i]) 
                for i in range(len(self.domain))]
 
    def neighbors(self, x):
        neighbors = []
        for j in range(len(self.domain)):
            if x[j] > self.domain[j][0]:
                neighbors.append(x[0:j] + [x[j]-1] + x[j+1:])
            if x[j] < self.domain[j][1]:
                neighbors.append(x[0:j] + [x[j]+1] + x[j+1:])
        return neighbors

    def valid(self, x):
        in_domain = [self.domain[j][0] < x[j] < self.domain[j][1]
                     for j in range(len(self.domain))]
        return reduce(bool.__and__, in_domain)

    def fitf(self, x):
        square_errors = [(x[i] - self.C[i])**2 for i in range(len(self.domain))] 
        return -sum(square_errors)


def rhc(P, sol=[]):
    # If not provided, create a feasible, random solution
    if not sol:
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
            print "{} ==> {}".format(current, best)
    return sol


def sa(P, T=10000.0, cool=0.95):
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
   
    # Finish optimization with random hill climbing
    vec = rhc(P, vec)
     
    return vec


def ga(P, popsize=100, step=1, mutprob=0.2, elite=0.2, maxiter=100):
     # Mutation Operation
    def mutate(vec):
        # Switching to a random neighbor is like mutation
        neighbors = P.neighbors(vec)
        i = random.randint(0, len(neighbors)-1)
        return neighbors[i] 

    # Crossover Operation
    def crossover(r1,r2):
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
                # Crossover
                c1=random.randint(0, topelite)
                c2=random.randint(0, topelite)
                crossed = crossover(ranked[c1], ranked[c2])
                pop.append(crossed)
    return rhc(P, ranked[0])


def mimic(P, n_iter=50, samples=50, percentile=0.9):
    mmc = Mimic(domain=P.domain, fitness_function= P.fitf, samples=samples,
                percentile=percentile)
    for i in range(n_iter):
        sols = mmc.fit()
    return sols[0]


def demo_problem(prob=SquareFunc):
    P = prob()

    print '#'*10 + ' Random hill climbing ' + '#'*10
    sol = rhc(P)
    print "solution: {}, fitness {}".format(sol, P.fitf(sol))
    print ''
    
    print '#'*10 + ' Simulated annealing ' + '#'*10
    sol = sa(P)
    print "solution: {}, fitness {}".format(sol, P.fitf(sol))
    print '' 

    print '#'*10 + ' Genetic algorithm ' + '#'*10
    sol = ga(P)
    print "solution: {}, fitness {}".format(sol, P.fitf(sol))
    print '' 


def demo():
    print '########## Random hill climbing ##########'
    demo_alg(opt_alg=rhc)
    print ''
    print '########## Simulated annealing  ##########'
    demo_alg(opt_alg=sa)


def demo_alg(opt_alg=rhc):
    print '########## Square function ##########'
    P = SquareFunc()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))

    print '########## 4-peaks problem ##########'
    P = FourPeaks()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))

    print '########## 6-peaks problem ##########'
    P = SixPeaks()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))

    print '########## K-colors problem ##########'
    P = KColors()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))
    
    print '########## Traveling salesman problem ##########'
    P = TSP()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))

    print '########## Knapsack problem ##########'
    P = Knapsack()
    sol = opt_alg(P)
    print "solution: {}\nfitness {}".format(sol, P.fitf(sol))

