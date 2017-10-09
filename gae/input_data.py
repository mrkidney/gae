import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

#Adds an extra feature to denote dummy nodes
def pad(A, X, size):
    Apad = np.zeros((size, size))
    Xpad = np.zeros((size, X.shape[1]+1))

    Apad[:A.shape[0], :A.shape[1]] = A
    Xpad[:X.shape[0], :X.shape[1]] = X
    Xpad[X.shape[0]:, -1] = 1
    return Apad, Xpad

def random_order(A, X):
    perm = np.random.permutation(A.shape[0])
    A[perm, :] = A
    A[:, perm] = A
    X[perm, :] = X
    return A, X

def load_mutag():
    #hard-coded maximum vertex count specifically for the mutag dataset
    VERTICES = 30

    def clean(L):
        return L.split('\n')[1:-1]

    As = []
    Xs = []
    Cs = np.zeros(188)
    for i in range(1,189):
        
        with open("data/mutag/mutag_" + str(i) + ".graph") as f:
            V, E, C = f.read().split('#')[1:4]
            V = clean(V)
            E = clean(E)
            C = clean(C)

            row, col, data = zip(*[e.split(",") for e in E])

            V = [int(val) for val in V]
            row = [int(val) - 1 for val in row]
            col = [int(val) - 1 for val in col]
            data = [int(val) for val in data]
            Cs[i-1] = 1.0 if C[0] == '1' else 0.0

            A = sp.csr_matrix((data,(row,col)), dtype = 'int16')
            row = range(len(V))
            col = [0] * len(V)
            X = sp.csr_matrix((V,(row,col)), dtype = 'int16')

            A, X = A.todense(), X.todense()
            A, X = pad(A, X, VERTICES)
            A, X = random_order(A, X)
            X = np.hstack((X, np.identity(VERTICES)))

            As.append(A)
            Xs.append(X)

    A = np.dstack(tuple(As))
    A = np.transpose(A, axes=(2, 0, 1))
    X = np.dstack(tuple(Xs))
    X = np.transpose(X, axes=(2, 0, 1))
    C = np.stack(Cs)
    return A, X, C, 2

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
