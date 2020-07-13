import torch
import sys
sys.path.insert(0, '/n/scratch2/xz204/Dr37/lib/python3.7/site-packages')
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
# from deeprobust.graph.defense import GCN, GCNSVD, GCNJaccard, RGCN, GCN_attack, att_coef
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from deeprobust.graph.defense import *
from tqdm import tqdm
import scipy
from sklearn.preprocessing import normalize
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=14, help='seed.')
# cora and citeseer are binary, pubmed has not binary features
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, )

adj, features, labels = data.adj, data.features, data.labels
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)

"""set the number of training/val/testing nodes"""
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
"""add undirected edges, orgn-arxiv is directed graph, we transfer it to undirected closely following 
https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
"""
adj = adj + adj.T
adj[adj>1] = 1


# Setup Surrogate model
surrogate = GCN_attack(nfeat=features.shape[1], nclass=labels.max().item()+1, n_edge=adj.nonzero()[0].shape[0],
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device, )
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)  # change this train_iters to 201: train_iters=201

# Setup Attack Model
target_node = 859

model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
model = model.to(device)

def main():
    degrees = adj.sum(0).A1
    # How many perturbations to perform. Default: Degree of the node
    n_perturbations = int(degrees[target_node])

    # # indirect attack/ influencer attack
    model.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5)
    modified_adj = model.modified_adj
    modified_features = model.modified_features

    print('=== testing GNN on original(clean) graph ===')
    test(adj, features, target_node,  attention=args.GNNGuard)

    print('=== testing GCN on perturbed graph ===')
    test(modified_adj, modified_features, target_node,attention=args.GNNGuard)


def test(adj, features, target_node, attention=False):
    ''
    """test on GCN """
    """model_name could be 'GCN', 'GAT', 'GIN','JK'  """
    # for orgn-arxiv: nhid =256, layers =3, epoch =500

    gcn = globals()[args.modelname](nfeat=features.shape[1], nhid=16,  nclass=labels.max().item() + 1, dropout=0.5,
              device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val=idx_val,
            idx_test=idx_test,
            attention=attention, verbose=True, train_iters=201)
    gcn.eval()
    _, output = gcn.test(idx_test=idx_test)

    probs = torch.exp(output[[target_node]])[0]
    print('probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def multi_test():
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(num_target=10)
    # node_list = [439, 1797]
    print(node_list)

    num = len(node_list)
    print('=== Attacking %s nodes respectively ===' % num)
    num_tar = 0
    for target_node in tqdm(node_list):
        # """for test"""
        # target_node = 419
        n_perturbations = int(degrees[target_node])
        if n_perturbations <1:  # at least one perturbation
            continue

        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, direct=False, n_influencers=5, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
        num_tar += 1
        print('classification rate : %s' % (1-cnt/num_tar), '# of targets:', num_tar)

"""Set attention"""
attention = args.GNNGuard

def single_test(adj, features, target_node):
    'ALL the baselines'
    # """defense models"""
    # classifier = globals()[args.defensemodel](nnodes=adj.shape[0], nfeat=features.shape[1], nhid=16,
    #                                           nclass=labels.max().item() + 1, dropout=0.5, device=device)

    ''' test on GCN (poisoning attack), model could be GCN, GAT, GIN'''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    print('modelname', args.modelname)
    classifier = classifier.to(device)
    classifier.fit(features, adj, labels, idx_train,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=attention) #model_name=model_name
    classifier.eval()
    acc_overall, output =  classifier.test(idx_test, ) #model_name=model_name

    probs = torch.exp(output[[target_node]])
    acc_test, pred_y, true_y = accuracy_1(output[[target_node]], labels[target_node])
    print('target:{}, pred:{}, label: {}'.format(target_node, pred_y.item(), true_y.item()))
    print('Pred probs', probs.data)
    return acc_test.item()

"""=======Basic Functions============="""
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_score
import numpy as np
# def att_coef(fea, edge_index):
#     # the weights of self-loop
#     edge_index = edge_index.tocoo()
#     fea = fea.todense()
#     fea_start, fea_end = fea[edge_index.row], fea[edge_index.col]
#     isbinray = np.array_equal(fea, fea.astype(bool)) #check is the fea are binary
#     np.seterr(divide='ignore', invalid='ignore')
#     if isbinray:
#         fea_start, fea_end = fea_start.T, fea_end.T
#         sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
#     # else:
#     #     sim_matrix = euclidean_distances(X=fea, Y=fea)
#     #     sim = sim_matrix[edge_index.row, edge_index.col]
#     #     w = 1 / sim
#     #     w[np.isinf(w)] = 0
#     #     sim = w
#
#     """build a attention matrix"""
#     att_dense = np.zeros([fea.shape[0], fea.shape[0]], dtype=np.float32)
#     row, col = edge_index.row, edge_index.col
#     att_dense[row, col] = sim
#     if att_dense[0, 0] == 1:
#         att_dense = att_dense - np.diag(np.diag(att_dense))
#     # normalization, make the sum of each row is 1
#     att_dense_norm = normalize(att_dense, axis=1, norm='l1')
#     # np.seterr(divide='ignore', invalid='ignore')
#     # att_dense_norm = np.nan_to_num(att_dense / np.sum(att_dense, axis=0)[:, None])
#
#     if att_dense_norm[0, 0] == 0:
#         # the weights of self-loop
#         degree = (att_dense != 0).sum(1)[:, None]
#         lam = np.float32(1 / (degree + 1))  # degree +1 is to add itself
#         lam = [x[0] for x in lam]
#         self_weight = np.diag(lam)
#         att = att_dense_norm + self_weight  # add the self loop
#     else:
#         att = att_dense_norm
#     att = np.exp(att) - 1  # make it exp to enhance the difference among edge weights
#     att_dense[att_dense <= 0.1] = 0  # set the att <0.1 as 0, this will decrease the accuracy for clean graph
#
#     att_lil = scipy.sparse.lil_matrix(att)
#     return att_lil

def select_nodes(num_target = 10):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    gcn = globals()[args.modelname](nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_test, verbose=True)
    gcn.eval()
    output = gcn.predict()
    degrees = adj.sum(0).A1

    margin_dict = {}
    for idx in tqdm(idx_test):
        margin = classification_margin(output[idx], labels[idx])
        acc, _, _ = accuracy_1(output[[idx]], labels[idx])
        if acc==0 or int(degrees[idx])<1: # only keep the correctly classified nodes
            continue
        """check the outliers:"""
        neighbours = list(adj.todense()[idx].nonzero()[1])
        y = [labels[i] for i in neighbours]
        node_y = labels[idx]
        aa = node_y==y
        outlier_score = 1- aa.sum()/len(aa)
        if outlier_score >=0.5:
            # print('outlier_score', outlier_score) # the lower, the better
            continue

        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: num_target]]
    low = [x for x, y in sorted_margins[-num_target: ]]
    other = [x for x, y in sorted_margins[num_target: -num_target]]
    other = np.random.choice(other, 2*num_target, replace=False).tolist()

    return other + low+ high


if __name__ == '__main__':
    main()
    # multi_test()
