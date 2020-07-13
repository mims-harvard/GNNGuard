import torch
import sys
sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)



adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)


perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

#1. to CSR sparse
adj, features = csr_matrix(adj), csr_matrix(features)

"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1



# Setup GCN as the Surrogate Model
surrogate = GCN_attack(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

attention = args.GNNGuard # if True, our method; if False, run baselines

def test(adj):
    # """defense models"""
    # classifier = globals()[args.defensemodel](nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)

    ''' testing model '''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    classifier.eval()

    # classifier.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    acc_test, output = classifier.test(idx_test)
    return acc_test.item()

def main():
    ''
    """save the mettacked adj"""
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    # pickle.dump(modified_adj, open('saved_pk/mettack_modified_adj_cora2484_0.05fake.pk', 'wb'))
    # print('modified_adj saved')

    print('=== testing GCN on original(clean) graph ===')
    test(adj)

    print('=== testing GCN on Mettacked graph ===')
    test(modified_adj)


if __name__ == '__main__':
    main()

