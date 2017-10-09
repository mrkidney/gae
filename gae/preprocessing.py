import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    return adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)


def construct_feed_dict(adj, adj_orig, features, labels, placeholders, indices):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['mask']: indices})
    return feed_dict

def get_split(size, prob):
    train_mask = np.random.choice(2, size, p=[prob, 1 - prob])
    val_mask = 1 - train_mask
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    return train_indices, val_indices

def apply_indices(arrays, indices):
    return tuple([np.copy(array[indices]) for array in arrays])

