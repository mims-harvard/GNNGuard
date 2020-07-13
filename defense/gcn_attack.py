import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import scipy
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import numpy as np
from deeprobust.graph.utils import *
from torch_geometric.nn import GINConv, global_add_pool, GATConv, GCNConv, ChebConv, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from deeprobust.graph.defense.basicfunction import att_coef


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, edge_weight=None):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight) # this function seems do message passing
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_attack(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, device=None):

        super(GCN_attack, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        # self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        nclass = int(nclass)



        """define the networks: deeprobust"""
        # self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        # self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = GCNConv(nfeat, nhid, bias=True,)
        self.gc2 = GCNConv(nhid, nclass, bias=True, )

        """GAT from torch-geometric"""
        # self.gc1 = GATConv(nfeat, nhid, heads=8, dropout=0.6)
        # # On the Pubmed dataset, use heads=8 in conv2.
        # self.gc2 = GATConv(nhid*8, nclass, heads=1, concat=True, dropout=0.6)

        """GIN from torch-geometric"""
        # num_features = nfeat
        # dim = 32
        # nn1 = Sequential(Linear(num_features, dim), ReLU(), )
        # self.gc1 = GINConv(nn1)
        # # self.bn1 = torch.nn.BatchNorm1d(dim)
        #
        # nn2 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc2 = GINConv(nn2)
        # self.jump = JumpingKnowledge(mode='cat')
        # # self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.fc2 = Linear(dim, nclass)


    def forward(self, x, adj_lil):
        '''
            adj: normalized adjacency matrix
        '''
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj = self.att_coef(x, adj) # update the attention by jaccard coefficient

        """In oder to use geometric, should convert the features and adj into dense matrix,
        x: [2708, 1433], adj: [2, 10556]"""
        # adj = adj_lil
        # edge_weight=None

        x = x.to_dense()
        adj = adj_lil.coalesce().indices()
        edge_weight = adj_lil.coalesce().values()

        """GCN and GAT"""

        x = F.relu(self.gc1(x, adj, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)

        """GIN"""
        # if not self.attention:
        #     edge_weight = None
        # x = F.relu(self.gc1(x, adj, edge_weight=edge_weight))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj, edge_weight = self.att_coef(x, adj) # update the attention by L2
        # x = F.dropout(x, p=0.2, training=self.training)
        # # x = self.bn1(x)
        # x = F.relu(self.gc2(x, adj, edge_weight=edge_weight))
        # x = [x] ### Add Jumping
        # x = self.jump(x)
        #
        # x = F.dropout(x, p=0.2,training=self.training)
        # x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    #
    # def att_coef(self, fea, edge_index, adj_lil=False):
    #     edge_index = edge_index.data.numpy()
    #     fea = fea.data.numpy()
    #     cor_x, cor_y = edge_index[0], edge_index[1]
    #
    #     isbinray = np.array_equal(fea, fea.astype(bool)) #check is the fea are binary
    #     if isbinray:
    #         sim_matrix = self.sim.todense()
    #         sim = np.squeeze(sim_matrix[cor_x, cor_y])
    #     else:
    #         sim_matrix = euclidean_distances(X=fea, Y=fea)
    #         sim = sim_matrix[cor_x, cor_y]
    #         np.seterr(divide='ignore', invalid='ignore')
    #         w = 1/sim
    #         w[np.isinf(w)] = 0
    #         sim = w
    #         # sim[sim>6]=6  # clip the upper bound as 10
    #
    #     """build a attention matrix"""
    #     att_dense = np.zeros([fea.shape[0], fea.shape[0]], dtype=np.float32)
    #     row, col = cor_x, cor_y
    #     att_dense[row, col] = sim
    #     # if the self-attention is 1, remove the self-attention add the original self-weight
    #     # self_attention = np.diag(edge_index)  # keep the self-weight
    #     if att_dense[0,0]==1:
    #         att_dense = att_dense - np.diag(np.diag(att_dense)) #+ np.diag(self_attention)
    #     # normalization, make the sum of each row is 1
    #     att_dense_norm = normalize(att_dense, axis=1, norm='l1') # norm could be l1 or l2
    #
    #
    #     # if the self-attention is 0, add the 1/degree as the self-weight
    #     if att_dense_norm[0,0]==0:
    #         # the weights of self-loop
    #         degree = (att_dense != 0).sum(1)[:, None]
    #         lam = np.float32(1 / (degree + 1))  # degree +1 is to add itself
    #         # lam = 0.2 * np.ones([data.num_nodes, 1])
    #         lam = [x[0] for x in lam]
    #         self_weight = np.diag(lam)
    #         att = att_dense_norm + self_weight  # add the self loop
    #     else:
    #         att = att_dense_norm
    #     att = np.exp(att) - 1  # make it exp to enhance the difference among edge weights
    #     att_d = att
    #
    #     if adj_lil == False:
    #         # transfer to tensor.sparse_coo_tensor
    #         att = scipy.sparse.lil_matrix(att).tocoo()
    #         indices = np.vstack((att.row, att.col))
    #         values = att.data
    #         size = att.shape
    #         att_adj = torch.tensor(indices, dtype=torch.int64)
    #         att_edge_weight = torch.tensor(values, dtype=torch.float32)
    #     ### the normalize will be conducted inner the GCN model, don't need normalize here
    #     #     att_lil = torch.sparse_coo_tensor(indices=torch.tensor(indices, dtype=torch.int64),
    #     #                                       values=torch.tensor(values, dtype=torch.float32), size=size)
    #     #     """Normalize D^{-1/2}AD^{-1/2}"""
    #     #     att_lil = utils.normalize_adj_tensor(att_lil, sparse=True)
    #     # else:
    #     #     att_lil = scipy.sparse.lil_matrix(att)
    #
    #     return att_adj, att_edge_weight, att_d

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=101, att_0=None,
            attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=500, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None
        self.attention = attention
        if self.attention:
            att_0 = att_coef(features, adj)
            adj = att_0 # update adj
            self.sim = att_0 # update att_0

        self.idx_test = idx_test
        # self.model_name = model_name
        # self.device = self.gc1.weight.device

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        normalize = False # we don't need normalize here, the norm is conducted in the GCN (self.gcn1) model
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        """Make the coefficient D^{-1/2}(A+I)D^{-1/2}"""
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            # print('iterations:', i)
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            # pred = output[self.idx_test].max(1)[1]

            acc_test =accuracy(output[self.idx_test], labels[self.idx_test])
            # acc_test = pred.eq(labels[self.idx_test]).sum().item() / self.idx_test.shape[0]



            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 200 == 0:
                print('Epoch {}, training loss: {}, test acc: {}'.format(i, loss_train.item(), acc_test))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))


            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test, model_name=None):
        # self.model_name = model_name
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output

    def _set_parameters(self):
        # TODO
        pass

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''

        # self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

