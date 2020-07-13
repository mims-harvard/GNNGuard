from sklearn.metrics import jaccard_score
import numpy as np
from sklearn.preprocessing import normalize
import scipy as sp
import torch
import tqdm
from sklearn.metrics.pairwise import euclidean_distances


def att_coef(fea, edge_index):
    # the weights of self-loop
    edge_index = edge_index.tocoo()
    fea = fea.todense()
    fea_start, fea_end = fea[edge_index.row], fea[edge_index.col]
    isbinray = np.array_equal(fea, fea.astype(bool)) #check is the fea are binary
    np.seterr(divide='ignore', invalid='ignore')
    if isbinray:
        fea_start, fea_end = fea_start.T, fea_end.T
        sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
    else:
        sim_matrix = euclidean_distances(X=fea, Y=fea)
        sim = sim_matrix[edge_index.row, edge_index.col]
        w = 1 / sim
        w[np.isinf(w)] = 0
        sim = w

    """build a attention matrix"""
    att_dense = np.zeros([fea.shape[0], fea.shape[0]], dtype=np.float32)
    row, col = edge_index.row, edge_index.col
    att_dense[row, col] = sim
    if att_dense[0, 0] == 1:
        att_dense = att_dense - np.diag(np.diag(att_dense))
    # normalization, make the sum of each row is 1
    att_dense_norm = normalize(att_dense, axis=1, norm='l1')
    # np.seterr(divide='ignore', invalid='ignore')
    character = np.vstack((att_dense_norm[row, col], att_dense_norm[col, row]))
    character = character.T


    if att_dense_norm[0, 0] == 0:
        # the weights of self-loop
        degree = (att_dense != 0).sum(1)[:, None]
        lam = np.float32(1 / (degree + 1))  # degree +1 is to add itself
        lam = [x[0] for x in lam]
        self_weight = np.diag(lam)
        att = att_dense_norm + self_weight  # add the self loop
    else:
        att = att_dense_norm
    att = np.exp(att) - 1  # make it exp to enhance the difference among edge weights
    att_dense[att_dense <= 0.1] = 0  # set the att <0.1 as 0, this will decrease the accuracy for clean graph

    att_lil = sp.sparse.lil_matrix(att)
    return att_lil


def accuracy_1(output, labels):
    """"""
    try:
        num = len(labels)
    except:
        num = 1

    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor([labels])

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / num, preds, labels


def drop_dissimilar_edges(features, adj, threshold=0.01, binary_fea=True):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    modified_adj = adj.copy().tolil()
    print('=== GCN-Jaccrad ===')
    # isSparse = sp.issparse(features)
    edges = np.array(modified_adj.nonzero()).T
    removed_cnt = 0
    for edge in tqdm(edges, disable=True): # disable=True to turn off the progress bar
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue

        if binary_fea==True:  # if it is binary
            J = _jaccard_similarity(features[n1], features[n2])

            if J < threshold:
                modified_adj[n1, n2] = 0
                modified_adj[n2, n1] = 0
                removed_cnt += 1
    print('removed', removed_cnt, 'edges in the original graph')
    return modified_adj

def _jaccard_similarity(a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J