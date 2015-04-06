import numpy as np
import matplotlib.pyplot as plt
import pickle
from ann import ANN  # ANN code
from datasets import Segmentation, Forest, Alertness  # Datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def exp1(data_class, K=7):
    """Run the clustering algorithms on the datasets and describe what you see.
    """
    # Load data set
    data = data_class()

    # Find K-means cluster
    print '-'*20 + ' K-means ' + '-'*20
    scaler = StandardScaler(with_mean=False)
    km = KMeans(n_clusters=K)
    X = scaler.fit_transform(data.train.X)
    Y = km.fit_predict(X)
    
    # Save K-means results
    fname = 'data/exp1_km.pickle'
    with open(fname, 'w') as f:
        pickle.dump([km, data, X, Y], f)

    # Do the clusters line up with the labels?
    print 'ARI: {}'.format( metrics.adjusted_rand_score(data.train.Y, Y))
    print 'AMI: {}'.format(metrics.adjusted_mutual_info_score(data.train.Y, Y))
    
    # How good are the clusters?
    print 'Homogeneity: {}'.format(metrics.homogeneity_score(data.train.Y, Y))
    print 'Completeness: {}'.format(metrics.completeness_score(data.train.Y, Y))
    print 'Silhouette: {}'.format(metrics.silhouette_score(X, km.labels_))

    # Find EM clusters
    print '-'*20 + ' EM ' + '-'*20
    em = GMM(n_components=K)
    em.fit(data.train.X)
    Y = em.predict(data.train.X)

    # Save EM results
    fname = 'data/exp1_em.pickle'
    with open(fname, 'w') as f:
        pickle.dump([em, data, Y], f)

    # Do the clusters line up with the labels?
    print 'ARI: {}'.format( metrics.adjusted_rand_score(data.train.Y, Y))
    print 'AMI: {}'.format(metrics.adjusted_mutual_info_score(data.train.Y, Y))
    
    # How good are the clusters?
    print 'Homogeneity: {}'.format(metrics.homogeneity_score(data.train.Y, Y))
    print 'Completeness: {}'.format(metrics.completeness_score(data.train.Y, Y))
    print 'Silhouette: {}'.format(metrics.silhouette_score(X, Y))


def plot1(f1=13, f2=15):
    """Plot results from experiment 1."""
    # Load K-means data for experiment 1
    fname = 'data/exp1_km.pickle'
    with open(fname) as f:
        km, data, X, Y = pickle.load(f)

    # Plot K-means clusters
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(km.cluster_centers_)))
    colors = plt.cm.rainbow(np.linspace(0, 1, data.n_class))
    shapes = ["o", "v", "s", "p", "*", "x", "+"]
    shapes = ["o"]*7
    for i in range(data.n_class):
        for j in range(len(km.cluster_centers_)):
            msk = (data.train.Y == i) & (Y == j)
            plt.scatter(data.train.X[msk, f1], data.train.X[msk, f2], 
                        color=colors[i], marker=shapes[j])
    plt.title("K-means")
    plt.show()
        
    # Load EM data for experiment 1
    fname = 'data/exp1_em.pickle'
    with open(fname) as f:
        em, data, Y = pickle.load(f) 

    # Plot EM clusters
    for i in range(data.n_class):
        for j in range(len(em.means_)):
            msk = (data.train.Y == i) & (Y == j)
            plt.scatter(data.train.X[msk, f1], data.train.X[msk, f2], 
                        color=colors[i], marker=shapes[j])
    plt.title("EM")
    plt.show()


def exp2(data_class, N=6):
    """Apply the dimensionality reduction algorithms to the two datasets and
    describe what you see."""
    # Load datamentation datasets
    data = data_class()

    # Apply PCA
    print '-'*20 + ' PCA ' + '-'*20
    scaler = StandardScaler()
    pca = PCA(n_components=N)
    X = scaler.fit_transform(data.train.X)
    X = pca.fit_transform(X)
    
    # Save PCA results
    fname = 'data/exp2_pca.pickle'
    with open(fname, 'w') as f:
        pickle.dump([pca, data, X], f)

    # Describe PCA results
    eigvals = np.linalg.eigvals(pca.get_covariance())
    expl_var = sum(pca.explained_variance_ratio_) 
    R = scaler.inverse_transform(pca.inverse_transform(X))  # Reconstruction
    R_error = sum(map(np.linalg.norm, R-data.train.X))
    print 'Eigenvalues:'
    print '{}'.format(eigvals)
    print 'Explained variance (%): {}'.format(expl_var)
    print 'Reconstruction error: {}'.format(R_error) 

    # Apply ICA
    print '-'*20 + ' ICA ' + '-'*20
    ica = FastICA(n_components=N, max_iter=200)
    X = ica.fit_transform(data.train.X)
    
    # Save ICA results
    fname = 'data/exp2_ica.pickle'
    with open(fname, 'w') as f:
        pickle.dump([ica, data, X], f)

    # Describe ICA results
    R = ica.inverse_transform(X)
    R_error = sum(map(np.linalg.norm, R-data.train.X))
    print 'Reconstruction error: {}'.format(R_error)

    # Apply "Randomized Components Analysis"
    print '-'*20 + ' RCA ' + '-'*20
    scaler = StandardScaler()
    rca = GaussianRandomProjection(n_components=N)
    X = scaler.fit_transform(data.train.X)
    X = rca.fit_transform(X)

    # Save RCA results
    fname = 'data/exp2_rca.pickle'
    with open(fname, 'w') as f:
        pickle.dump([rca, data, X], f)

    # Describe RCA results
    inv = np.linalg.pinv(rca.components_)
    R = scaler.inverse_transform(np.dot(X, inv.T))  # Reconstruction
    R_error = sum(map(np.linalg.norm, R-data.train.X))
    print 'Reconstruction error: {}'.format(R_error) 

    # Apply Linear Discriminant Analysis
    print '-'*20 + ' LDA ' + '-'*20
    lda = LDA(n_components=N)
    X = lda.fit_transform(data.train.X, data.train.Y)     

    # Save LDA results
    fname = 'data/exp2_lda.pickle'
    with open(fname, 'w') as f:
        pickle.dump([lda, data, X], f)

    # Describe LDA results
    inv = np.linalg.pinv(lda.scalings_[:, 0:N])
    R = np.dot(X, inv) + lda.xbar_
    R_error = sum(map(np.linalg.norm, R-data.train.X))
    print 'Reconstruction error: {}'.format(R_error)


def plot2(f1=0, f2=1):
    """Plot experiment 2 results."""
    # Load PCA results
    fname = 'data/exp2_pca.pickle'
    with open(fname) as f:
        pca, data, X = pickle.load(f)

    # Plot PCA
    colors = plt.cm.rainbow(np.linspace(0, 1, data.n_class))
    for i in range(data.n_class):
        msk = data.train.Y == i
        plt.scatter(X[msk, f1], X[msk, f2], color=colors[i])
    plt.title("PCA")
    plt.show()

    # Load ICA results
    fname = 'data/exp2_ica.pickle'
    with open(fname) as f:
        ica, data, X = pickle.load(f)

    # Plot ICA
    for i in range(data.n_class):
        msk = data.train.Y == i
        plt.scatter(X[msk, f1], X[msk, f2], color=colors[i])
    plt.title("ICA")
    plt.show()

    # Load RCA results
    fname = 'data/exp2_rca.pickle'
    with open(fname) as f:
        rca, data, X = pickle.load(f)

    # Plot RCA
    for i in range(data.n_class):
        msk = data.train.Y == i
        plt.scatter(X[msk, f1], X[msk, f2], color=colors[i])
    plt.title("RCA")
    plt.show()

    # Load LDA results
    fname = 'data/exp2_lda.pickle'
    with open(fname) as f:
        lda, data, X = pickle.load(f)

    # Plot LDA
    colors = plt.cm.rainbow(np.linspace(0, 1, data.n_class))
    for i in range(data.n_class):
        msk = data.train.Y == i
        plt.scatter(X[msk, f1], X[msk, f2], color=colors[i])
    plt.title("LDA")
    plt.show()


def dim_red_pipelines(N=6):
    """Set up dimensionality reduction pipelines for convenience. 
    Reduce to N dimensions."""
    pipe_pca = Pipeline([('scale', StandardScaler()),
                         ('PCA', PCA(n_components=N))])
    pipe_ica = Pipeline([('ICA', FastICA(n_components=N, max_iter=100))])
    pipe_rca = Pipeline([('scale', StandardScaler()),
                         ('RCA', GaussianRandomProjection(n_components=N))])
    pipe_lda = Pipeline([('LDA', LDA(n_components=N))])
    return [pipe_pca, pipe_ica, pipe_rca, pipe_lda]


def cluster_pipelines(K=7):
    """ Set up cluster pipelines for convenience. Search for K clusters."""
    pipe_km = Pipeline([('scale', StandardScaler(with_mean=False)),
                        ('K-means', KMeans(n_clusters=K))])
    pipe_em = Pipeline([('EM', GMM(n_components=K))])
    return [pipe_km, pipe_em]


def exp3(data_class, N=6, K=7):
    """Reproduce your clustering experiments, but on the data after you've run
    dimensionality reduction on it."""

    # Run all pairs of reduction and clustering routines
    dim_red = dim_red_pipelines(N)
    cluster = cluster_pipelines(K)
    for dr in dim_red:
        for ca in cluster:
            # Print name of algorithms used
            print '\n{} & {}'.format(dr.steps[-1][0], ca.steps[-1][0])

            # Load datamentation data set
            data = data_class()
            
            # Reduce dimensionality
            if dr.steps[-1][0] == "LDA":
                X = dr.fit_transform(data.train.X, data.train.Y)
            else:
                X = dr.fit_transform(data.train.X)

            # Cluster
            if ca.steps[-1][0] == "EM": 
                # Because of Scikit bug, need to use GMM directly
                ca.steps[-1][1].fit(X)
                C = ca.steps[-1][1].predict(X)
            else:
                ca.fit(X)
                C = ca.predict(X)

            # Save results
            alg1 = dr.steps[-1][0]
            alg2 = ca.steps[-1][0]
            fname = "data/exp3_" + alg1 + '_' + alg2 + ".pickle"
            with open(fname, 'w') as f:
                pickle.dump([data, X, C], f)

            # Do the clusters line up with the labels?
            print 'ARI: {}'.format( metrics.adjusted_rand_score(data.train.Y, C))
            print 'AMI: {}'.format(metrics.adjusted_mutual_info_score(data.train.Y, C))
        
            # How good are the clusters?
            print 'Homogeneity: {}'.format(metrics.homogeneity_score(data.train.Y, C))
            print 'Completeness: {}'.format(metrics.completeness_score(data.train.Y, C))
            print 'Silhouette: {}'.format(metrics.silhouette_score(X, C))

def plot3(f1=0, f2=1):
    """Plot results for experiment 3."""
    # Plot clusters in the reduced dimensional spaced for all combinations of 
    # dimensionality reduction and clustering algorithms
    for dr in ["PCA", "ICA", "RCA", "LDA"]:
        for ca in ["K-means", "EM"]:
            # Load data
            fname = "data/exp3_" + dr + '_' + ca + '.pickle'    
            with open(fname) as f:
                data, X, C = pickle.load(f)
        
            # Plot results
            colors = plt.cm.rainbow(np.linspace(0, 1, data.n_class))
            for i in range(data.n_class):
                msk = C == i
                plt.scatter(X[msk, f1], X[msk, f2], color=colors[i])
            plt.title(dr + ' & ' + ca)
            plt.show()


def exp4(data_class, N=6, max_iter=50):
    """Apply the dimensionality reduction algorithms to one of your datasets
    from assignment #1, then rerun your neural network learner on the newly
    projected data."""
    
    # Load "clean" dataset
    data = data_class()

    # Set up dimensionality reduction pipelines
    dim_red = dim_red_pipelines(N)

    # Build the neural network without dimensionality reduction
    nn = ANN()
    nn.train = nn.load_data(data.train.X, data.train.Y, data.n_class)
    nn.test = nn.load_data(data.test.X, data.test.Y, data.n_class)
    nn.make_network()
    nn.make_trainer()

    # Train and run the neural network as a baseline
    print 'Baseline neural network (no dimensionality reduction)'
    for iter in range(max_iter):
        nn.train_network()
        print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                     nn.fitf(train=False))

    # Apply dimensionality reduction and run neural network for all algorithms 
    for dr in dim_red:
        # Print name of algorithms used
        print '\n{}'.format(dr.steps[-1][0])

        # Apply dimensionality reduction algorithm to training and test sets.
        if dr.steps[-1][0] != 'LDA':
            train_X = dr.fit_transform(data.train.X)
        else:
            train_X = dr.fit_transform(data.train.X, data.train.Y)
        test_X = dr.transform(data.test.X)

        # Build neural network
        nn = ANN()
        nn.train = nn.load_data(train_X, data.train.Y, data.n_class)
        nn.test = nn.load_data(test_X, data.test.Y, data.n_class)
        nn.make_network()
        nn.make_trainer()

        # Run neural network
        for iter in range(max_iter):
            nn.train_network()
            print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                     nn.fitf(train=False))


def exp5(data_class, N=6, K=7, max_iter=50):
    """Apply the clustering algorithms to the same dataset to which you just
    applied the dimensionality reduction algorithms, treating the clusters as
    if they were new (additional) features. Rerun your neural network leaner
    on the newly projected data."""

    # Load dataset
    data = data_class()

    # Set up dimensionality reduction and clustering pipelines
    dim_red = dim_red_pipelines(N)
    cluster = cluster_pipelines(K)

    # Build the neural network without dimensionality reduction
    nn = ANN()
    nn.train = nn.load_data(data.train.X, data.train.Y, data.n_class)
    nn.test = nn.load_data(data.test.X, data.test.Y, data.n_class)
    nn.make_network()
    nn.make_trainer()

    # Apply dimensionality reduction + clustering, then run neural network
    for dr in dim_red:
        for ca in cluster:
            # Print name of algorithms used
            print '\n{} & {}'.format(dr.steps[-1][0], ca.steps[-1][0])

            # Apply dimensionality reduction algorithm to training and test sets.
            if dr.steps[-1][0] != 'LDA':
                train_X = dr.fit_transform(data.train.X)
            else:
                train_X = dr.fit_transform(data.train.X, data.train.Y)
            test_X = dr.transform(data.test.X)

            # Apply clustering to the reduced dimensionality dataset
            if ca.steps[-1][0] == "EM":
                ca.steps[-1][1].fit(train_X)
                C_train = ca.steps[-1][1].predict(train_X)
                C_test = ca.steps[-1][1].predict(test_X)
            else:
                ca.fit(train_X)
                C_train = ca.predict(train_X)
                C_test = ca.predict(test_X)

            # Add cluster assignment as a feature
            train_X = [np.append(x, c) for x, c 
                       in zip(train_X, C_train)]
            test_X = [np.append(x, c) for x, c 
                      in zip(test_X, C_test)]

            # Build neural network
            nn = ANN()
            nn.train = nn.load_data(train_X, data.train.Y, data.n_class)
            nn.test = nn.load_data(test_X, data.test.Y, data.n_class)
            nn.make_network()
            nn.make_trainer()

            # Run neural network
            for iter in range(max_iter):
                nn.train_network()
                print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                         nn.fitf(train=False))


def main():
    # Run experiment 1 on two datasets
    print '#'*20
    print exp1.__doc__
    print '#'*20

    print '### Segmentation dataset ###' 
    exp1(Segmentation)
    print '\n'

    print '### Alertness dataset ###'
    exp1(Alertness)
    print '\n'*2
    
    # Run experiment 2 on two datasets
    print '#'*20
    print exp2.__doc__
    print '#'*20

    print '### Segmentation dataset ###' 
    exp2(Segmentation)
    print '\n'

    print '### Alertness dataset ###'
    exp2(Alertness)
    print '\n'*2

    # Run experiment 3 on two datasets
    print '#'*20
    print exp3.__doc__
    print '#'*20

    print '### Segmentation dataset ###' 
    exp3(Segmentation)
    print '\n'

    print '### Alertness dataset ###'
    exp3(Alertness)
    print '\n'*2

    # Run experiment 4 on one dataset from HW#1
    print '#'*20
    print exp4.__doc__
    print '#'*20

    print '### Segmentation dataset ###'
    exp4(Segmenation)
    print '\n'*2

    # Run experiment 5 on one dataset from HW#1
    print '#'*20
    print exp5.__doc__
    print '#'*20

    print '### Segmentation dataset ###' 
    exp5(Segmentation)


if __name__ == "__main__":
    main()
