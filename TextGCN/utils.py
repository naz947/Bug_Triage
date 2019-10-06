import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from scipy.sparse.linalg import norm as sparsenorm
from scipy.linalg import qr
import random
# from sklearn.metrics import f1_score


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#
# def calc_f1(y_true, y_pred):
#     y_true = np.argmax(y_true, axis=1)
#     y_pred = np.argmax(y_pred, axis=1)
#     return f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="macro")
#

#
# def load_data(dataset_str):
#     """Load data."""
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#
#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#
#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
#

'''
def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data='R8_train'
    for i in range(len(names)):
        with open(dataset_str+"/ind.{}.{}".format(data, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_original(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    data='R8_train'
    for i in range(len(names)):
        with open(dataset_str+"/ind.{}.{}".format(data, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    features = sp.identity(features.shape[0])
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
'''
def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    data='R8_train'
    for i in range(len(names)):
        with open(dataset_str+"/ind.{}.{}".format(data, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    features = sp.identity(features.shape[0])
    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        dataset_str+"/{}.train.index".format(data))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    idx_vocab = range(train_size, allx.shape[0])

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    vocab_mask = sample_mask(idx_vocab, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_vocab = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    y_vocab[vocab_mask, :] = labels[vocab_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, y_vocab, train_mask, val_mask, test_mask, vocab_mask, train_size, test_size

def load_pmi_graph(dataset):
    # load pmi graph
    names = ['allx', 'adj']
    objects = []
    data='R8_train'
    for i in range(len(names)):
        with open(dataset+"/ind.{}.{}".format(data, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    allx, adj = tuple(objects)

    features = sp.vstack(allx).tolil()

    return adj, features

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def nontuple_preprocess_adj(adj):
    if adj.shape[0] == adj.shape[1]:
        adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    else:
        rowsum = np.array(adj.sum(1))
        rowdegree_inv = np.power(rowsum, -0.5).flatten()
        rowdegree_inv[np.isinf(rowdegree_inv)] = 0.
        rowdegree_mat_inv = sp.diags(rowdegree_inv)

        colsum = np.array(adj.sum(0))
        coldegree_inv = np.power(colsum, -0.5).flatten()
        coldegree_inv[np.isinf(coldegree_inv)] = 0.
        coldegree_mat_inv = sp.diags(coldegree_inv)

        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()

def column_prop(adj):
    column_norm = sparsenorm(adj, axis=0)
    # column_norm = pow(sparsenorm(adj, axis=0),2)
    norm_sum = sum(column_norm)
    return column_norm/norm_sum

def mix_prop(adj, features, sparseinputs=False):
    adj_column_norm = sparsenorm(adj, axis=0)
    if sparseinputs:
        features_row_norm = sparsenorm(features, axis=1)
    else:
        features_row_norm = np.linalg.norm(features, axis=1)
    mix_norm = adj_column_norm*features_row_norm

    norm_sum = sum(mix_norm)
    return mix_norm / norm_sum

def lanczos(A,k,q):
    n = A.shape[0]
    Q = np.zeros((n,k+1))

    Q[:,0] = q/norm(q)

    alpha = 0
    beta = 0

    for i in range(k):
      if i == 0:
        q = np.dot(A,Q[:,i])
      else:
        q = np.dot(A, Q[:,i]) - beta*Q[:,i-1]
      alpha = np.dot(q.T, Q[:,i])
      q = q - Q[:,i]*alpha
      q = q - np.dot(Q[:,:i], np.dot(Q[:,:i].T, q)) # full reorthogonalization
      beta = norm(q)
      Q[:,i+1] = q/beta
      print(i)

    Q = Q[:,:k]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    # A2 = np.dot(Q[:,:k], np.dot(Sigma[:k,:k], Q[:,:k].T))
    # return A2
    return Q, Sigma

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_appr = np.array(sp.csr_matrix.todense(adj))
    # # adj_appr = dense_lanczos(adj_appr, 100)
    # adj_appr = dense_RandomSVD(adj_appr, 100)
    # if adj_appr.sum(1).min()<0:
    #     adj_appr = adj_appr- (adj_appr.sum(1).min()-0.5)*sp.eye(adj_appr.shape[0])
    # else:
    #     adj_appr = adj_appr + sp.eye(adj_appr.shape[0])
    # adj_normalized = normalize_adj(adj_appr)

    # adj_normalized = normalize_adj(adj+sp.eye(adj.shape[0]))
    # adj_appr = np.array(sp.coo_matrix.todense(adj_normalized))
    # # adj_normalized = dense_RandomSVD(adj_appr,100)
    # adj_normalized = dense_lanczos(adj_appr, 100)


    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)

#from lanczos import lanczos
def dense_lanczos(A,K):
    q = np.random.randn(A.shape[0], )
    Q, sigma = lanczos(A, K, q)
    A2 = np.dot(Q[:,:K], np.dot(sigma[:K,:K], Q[:,:K].T))
    return sp.csr_matrix(A2)

def sparse_lanczos(A,k):
    q = sp.random(A.shape[0],1)
    n = A.shape[0]
    Q = sp.lil_matrix(np.zeros((n,k+1)))
    A = sp.lil_matrix(A)

    Q[:,0] = q/sparsenorm(q)

    alpha = 0
    beta = 0

    for i in range(k):
      if i == 0:
        q = A*Q[:,i]
      else:
        q = A*Q[:,i] - beta*Q[:,i-1]
      alpha = q.T*Q[:,i]
      q = q - Q[:,i]*alpha
      q = q - Q[:,:i]*Q[:,:i].T*q # full reorthogonalization
      beta = sparsenorm(q)
      Q[:,i+1] = q/beta
      print(i)

    Q = Q[:,:k]

    Sigma = Q.T*A*Q
    A2 = Q[:,:k]*Sigma[:k,:k]*Q[:,:k].T
    return A2
    # return Q, Sigma

def dense_RandomSVD(A,K):
    G = np.random.randn(A.shape[0],K)
    B = np.dot(A,G)
    Q,R =qr(B,mode='economic')
    M = np.dot(Q, np.dot(Q.T, A))
    return sp.csr_matrix(M)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
