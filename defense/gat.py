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
from scipy.sparse import lil_matrix

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


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, n_edge=1,with_relu=True, drop=False,
                 with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = int(nclass)
        self.dropout = dropout
        self.lr = lr
        self.drop = drop
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1)) # creat a generator between [0,1]
        # self.beta = Parameter(torch.Tensor(self.n_edge))
        self.bns = torch.nn.BatchNorm1d(nhid)
        nclass = int(nclass)

        """define the networks: deeprobust"""
        # self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        # self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        """GCN from geometric"""
        """network from torch-geometric, """
        # self.gc1 = GCNConv(nfeat, nhid, bias=True,)
        # self.gc2 = GCNConv(nhid, nclass, bias=True, )

        """GAT from torch-geometric"""
        self.gc1 = GATConv(nfeat, nhid, heads=8, dropout=0.6)
        self.gc2 = GATConv(nhid*8, nclass, heads=1, concat=True, dropout=0.6)

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


    def forward(self, x, adj):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense()
        edge_index = adj._indices()

        """GCN and GAT"""
        if self.attention:
            adj = self.att_coef(x, adj, i=0)
        x = self.gc1(x, edge_index, edge_weight=adj._values())
        x = F.relu(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1)
            adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        else:
            adj_values = adj._values()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)


        # """GIN"""
        # x = F.relu(self.gc1(x, adj, edge_weight=edge_weight))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj, edge_weight_2 = self.att_coef_2(x, adj) # update the attention by L2
        #     try:
        #         edge_weight = self.gate* edge_weight_2 + (1-self.gate)* edge_weight # involve the last layer's attention
        #     except:
        #         edge_weight = edge_weight_2
        #         print('the gate is not ok')
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu(self.gc2(x, adj, edge_weight=edge_weight))
        # # x = [x] ### Add Jumping        # x = self.jump(x)
        # x = F.dropout(x, p=0.2,training=self.training)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        # row, col = edge_index[0], edge_index[1]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        # sim_matrix = torch.from_numpy(sim_matrix)
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                     att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            # degree = degree.squeeze(-1).squeeze(-1)
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        att_adj = edge_index
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0]-1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=81, att_0=None,
            attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=500, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None
        self.attention = attention
        self.idx_test = idx_test

        # self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        # normalize = False # we don't need normalize here, the norm is conducted in the GCN (self.gcn1) model
        # if normalize:
        #     if utils.is_sparse_tensor(adj):
        #         adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         adj_norm = utils.normalize_adj_tensor(adj)
        # else:
        #     adj_norm = adj
        adj = self.add_loop_sparse(adj)

        self.adj_norm = adj
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
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=None)  # this weight is the weight of each training nodes
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

            # acc_test =accuracy(output[self.idx_test], labels[self.idx_test])

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            # if verbose and i % 20 == 0:
            #     print('Epoch {}, training loss: {}, val acc: {}'.format(i, loss_train.item(), acc_val))

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
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
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

