# Optimization problems 
import csv
import networkx
import numpy as np
import random
from geopy.distance import great_circle
from itertools import combinations
from itertools import groupby
from itertools import izip
from IPython.core.debugger import Tracer
from scipy.stats import bernoulli
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.neuralnets import NNclassifier
from sklearn.metrics import accuracy_score


class FourPeaks():
    """Four Peaks problem."""
    def __init__(self, N=100, R=100, T=10):
        """Initialize with N-length bit string, R reward, and T threshold."""
        self.N = N
        self.domain = [(0,1)]*N
        self.R = R
        self.T = T
        self.name = "Four Peaks"

    def random_solution(self):
        """Generate random feasible solution."""
        return [random.randint(0,1) for i in range(self.N)]
        
    def neighbors(self, x):
        """Generate neighbors to solution x."""
        x = list(x)
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
        """Check for valid solution."""
        return set(x) == {0, 1}

    def o(self, x):
        """Number of leading ones"""
        if x[0] == 0:
            return 0
        else:
            ones = (list(it) for contains1, it in groupby(x, lambda x:x==1))
            return len(ones.next())

    def z(self, x):
        """Number of leading zeros."""
        if x[-1] == 1:
            return 0
        else:
            x = reversed(x)
            zeros = (list(it) for contains0, it in groupby(x, lambda x:x==0))
            return len(zeros.next())

    def fitf(self, x):
        """Fitness function"""
        if (self.o(x) > self.T) & (self.z(x) > self.T):
            reward = self.R
        else:
            reward = 0
        return max(self.o(x), self.z(x)) + reward


class SixPeaks(FourPeaks):
    """Six Peaks problem."""
    def __init__(self, N=100, R=100, T=10):
        """Initialize with N-length bit string, R reward, and T threshold."""
        self.N = N
        self.domain = [(0,1)]*N
        self.R = R
        self.T = T
        self.name = "Six Peaks"

    def fitf(self, x):
        """Fitness function."""
        y = x[::-1]
        if ((self.o(x) > self.T) and (self.z(x) > self.T)) or \
            ((self.o(y) > self.T) and (self.z(y) > self.T)):
            reward = self.R
        else:
            reward = 0
        return max(self.o(x), self.z(x), self.o(y), self.z(y)) + reward


class KColors():
    """K Colors problem."""
    def __init__(self, N=25, K=2):
        """Initialize with N vertex graph network, and 2**K colors."""
        self.N = N
        self.K = K
        self.domain = [(0, 1)]*N*(2**K)
        g = networkx.gnp_random_graph(N, (2**(K+1)-0.)/N)
        while not networkx.is_connected(g):
            g = networkx.gnp_random_graph(N, (2**(K+1)-0.)/N)
        self.graph = g
        self.name = "K Colors"

    def random_solution(self):
        """Generate random feasible solution."""
        return [random.randint(0, 1)
                for i in range(len(self.domain))]

    def neighbors(self, x):
        """Generate list of neighbors to solution x."""
        neighbors = []
        for i in range(len(self.domain)):
            if x[i] == 0:
                neighbors.append(x[0:i] + [1] + x[i+1:])
            else:
                neighbors.append(x[0:i] + [0] + x[i+1:])
        return neighbors

    def crossover(self, x, y):
        """Single point crossover which preserves binary encodings"""
        i = random.randint(1, self.N-2)
        return x[0:self.K*i] + y[self.K*i:]

    def bin2dec(self, x):
        """Transform bit string to node index (decimal) representation"""
        return [2*a + b for a,b in zip(x[0::2], x[1::2])]

    def fitf(self, x):
        """Fitness function."""
        y = self.bin2dec(x)
        return -sum([y[a] == y[b] for a, b in self.graph.edges()])
        

class TSP():
    """Travelling salesman problem."""
    def __init__(self):
        """Initialize with 50 European cities."""
        self.cities, self.dist = self.european_cities()
        self.N = len(self.cities)
        self.domain = [(0, self.N-1)] * self.N
        self.name = "Travelling Salesman"

    def european_cities(self):
        """Load list of European cities and intercity distances."""
        def geo_dist(x, y):
            """Calculate Euclidean distance from latitude-longitude coords."""
            return (x-y).dot(x-y)**0.5

        # Load European cities data
        with open('data/cities.csv') as f:
            data = csv.reader(f, delimiter=',')
            cities = [x for i, x in enumerate(data) if i != 0]
        
        # City names
        names = [x[3] for x in cities]

        # Latitude-longitude
        latlon = [np.array([float(x[0]), float(x[1])])
                  for x in cities]

        # Get between-city "great circle" distance matrix
        N = len(cities)
        dist = np.zeros([N, N])
        for i, a in enumerate(cities):
            for j, b in enumerate(cities):
                if j > i:
                    dist[i,j] = great_circle(latlon[i], latlon[j]).miles
        dist += dist.T 
        return names, dist
        
        
    def random_solution(self):
        """Generate random feasible solution."""
        return list(np.random.permutation(range(self.N)))

    def neighbors(self, x):
        """Generate list of feasible neighbors to solution x."""
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
        """Check whether solution x is valid."""
        distinct_cities = (len(set(x)) == len(x))
        all_cities = (set(x) == set(range(self.N)))

        if all_cities & distinct_cities:
            return True
        else:
            return False

    def crossover(self, x, y):
        """ Partially-mapped crossover (PMX), from Goldber and Lingle 1985
        'Alleles, Loci, and the Traveling Salesman Problem'
        """
        # Pick crossover point
        p = random.randint(1, len(x)-1)
        
        # Displace x's alleles with y's
        offspring = list(x)
        for i in range(p):
            j = offspring.index(y[i])
            offspring[i], offspring[j] = y[i], offspring[i]
        return offspring

    def fitf(self, x):
        """Fitness function."""
        x = map(int, list(x))
        fitness = -sum([self.dist[first, second]
                       for first, second in izip(x, x[1:] +  [x[0]])])
        return fitness

    def names(self, x):
        """Get names of cities in order of solution x."""
        return map(lambda y: self.cities[y], x)


class Knapsack():
    """Knapsack problem."""
    def __init__(self, N=50, max_weight=100, alpha=10, P=0.3):
        """Initialize Knapsack with -N- items, -max_weight- maximum weight,
        -alpha- scaling, and -P- initial item probability."""
        self.domain = [(0, 1)]*N
        self.N = N
        self.max_weight = max_weight
        self.alpha = alpha
        self.P = P
        self.weights = np.random.randint(1, max_weight/alpha, size=N)
        self.values = np.random.randint(1, max_weight, size=N)
        self.name = "Knapsack"

    def random_solution(self):
        """Generate random feasible solution."""
        sol = []
        while not self.valid(sol):
            sol = list(bernoulli.rvs(self.P, size=self.N))
        return sol

    def get_items(self, x):
        """Get list of items (their indices) in the knapsack solution x."""
        return [i for i, e in enumerate(x) if e != 0]

    def total_weight(self, x):
        """Calculate total weight of solution x."""
        return sum([self.weights[i] for i in self.get_items(x)])

    def total_value(self, x):
        """Calculate total value of solution x."""
        return sum([self.values[i] for i in self.get_items(x)])

    def valid(self, x):
        """Check validity of solution x."""
        len_N = (len(x) == self.N)
        if len_N and self.total_weight(x) <= self.max_weight:
            return True
        else:
            return False 

    def neighbors(self, x):
        """Generate list of neighbors to solution x."""
        # Create list of neighboring solutions
        neighbors = []
        x = list(x)
        for j in range(len(self.domain)):
            # Flip bits
            if x[j] == 1:
                neighbors.append(x[0:j] + [0] + x[j+1:])
            else:
                neighbors.append(x[0:j] + [1] + x[j+1:])
        neighbors = filter(lambda x: self.valid(x), neighbors)
        return neighbors
    
    def repair(self, x, method="random"):
        """Genetic repairing function to deal with bad crossovers."""
        y = list(x)
        if method == "random":
            # Random repair method
            while self.total_weight(y) > self.max_weight:
                # Pick a random item
                ii = [i for i, e in enumerate(y) if e > 0]
                i = ii[random.randint(0, len(ii)-1)]
                # Remove one instance of that item
                y[i] -= 1
        elif method == "greedy":        
            # Greedy method: in order of profit to weight ratios
            pass  # Not implemented.
        return y
    
    def penalty(self, x, growth="quadratic"):
        """Penalty function to penalize (but allow!) bad crossovers."""
        overflow = self.total_weight(x) - self.max_weight
        if overflow > 0:
            if growth == "quadratic":
                return overflow**2
            elif growth == "linear":
                return overflow
            elif growth == "log":
                return math.log(overflow)

    def crossover(self, x, y):
        """Crossover with genetic repairing."""
        # Crossover with genetic repairing
        i = random.randint(1, len(self.domain)-2)
        xy = x[0:i] + y[i:]
        return self.repair(xy)     

    def fitf(self, x):
        """Fitness function with penalty."""
        return self.total_value(x) - self.penalty(x)
    

class SquareFunc():
    """Square Function problem with single optima. Just for testing."""
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


class ANN():
    """Artifical neural network weights problem. The ANN is trying to segment
    an outdoor image into 7 different labels"""
    def __init__(self, num_hidden=5, momentum=0.1, weightdecay=0.01, 
                 verbose=False):
        """Initialize with 5 hidden nodes and image segmentation data."""
        self.num_hidden = num_hidden
        self.train_file = 'data/segmentation.train'
        self.test_file = 'data/segmentation.test'
        self.train = self.load_data(self.train_file)
        self.test = self.load_data(self.test_file)
        self.network = self.make_network(num_hidden)
        self.domain = [(float('-Inf'), float('Inf'))]*len(self.network.params)
        self.trainer = self.make_trainer(momentum=momentum, 
                                         weightdecay=weightdecay, 
                                         verbose=verbose)


    def load_data(self, fname):
        """Load image segmentation data."""
        with open(fname) as f:
            data = csv.reader(f, delimiter=',')
            data = [row for row in data]
        tx = [map(float, x[1:len(x)]) for x in data[6:]]
        ty = [x[0] for x in data[6:]]
        for i, label in enumerate(ty):
            if label == 'BRICKFACE':
                ty[i] = 0
            elif label == 'SKY':
                ty[i] = 1
            elif label == 'FOLIAGE':
                ty[i] = 2
            elif label == 'CEMENT':
                ty[i] = 3
            elif label == 'WINDOW':
                ty[i] = 4
            elif label == 'PATH':
                ty[i] = 5
            elif label == 'GRASS':
                ty[i] = 6
            else:
                print 'error'
        ds = ClassificationDataSet(19, 1, nb_classes=7)
        for x, y in zip(tx, ty):
            ds.addSample(x, y)
        ds._convertToOneOfMany()
        return ds

    def make_network(self, num_hidden=5):
        """Build neural network with -num_hidden- nodes."""
        return buildNetwork(self.train.indim, num_hidden, self.train.outdim)

    def make_trainer(self, momentum=0.1, weightdecay=0.01, verbose=False):
        """Make backpropagation trainer."""
        return BackpropTrainer(self.network, dataset=self.train,
                               momentum=momentum, weightdecay=weightdecay,
                               verbose=verbose)

    def train_network(self):
        """Train network with one iteration of backpropagation."""
        self.trainer.trainEpochs(1)

    def fitf(self, weights=[], train=True):
        """Calculate fitness score on either training or test set."""
        weights = list(weights)
        if not weights:
            weights = self.network.params
        if train:
            ds = self.train
        else:
            ds = self.test
        self.network._setParameters(weights)
        pred = self.network.activateOnDataset(ds)
        preds = [y.argmax() for y in pred]
        return accuracy_score(preds, ds['class'], normalize=True)

    def neighbors(self, weights):
        """Generate list of neighbors by perturbing current solution with
        standard normal noise on a weight-by-weight basis."""
        weights = list(weights)
        neighbors = []
        for i in range(len(weights)):
            step = random.gauss(0, 1)
            neighbors.append(weights[0:i] + [weights[i]+step] + weights[i+1:])
            neighbors.append(weights[0:i] + [weights[i]-step] + weights[i+1:])
        return neighbors

    def random_solution(self):
        "Generate random feasible solution."""
        self.network.randomize()
        return list(self.network.params)
