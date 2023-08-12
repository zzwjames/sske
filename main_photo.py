import argparse
from model import Grace, Model, Grace_MLP
from aug import aug
from dataset import load
from sklearn.manifold import TSNE
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from eval import label_classification
import warnings
import numpy as np
from dgl.data import CoraGraphDataset
# from pyvis.network import Network
import random
from scipy.spatial.distance import pdist
import statistics
warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_mlp', type=bool, default=False, help='Specify whether to use MLP')

parser.add_argument('--epochs', type=int, default=500, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--act_fn', type=str, default='relu')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
# parser.add_argument('--act_fn', type=str, default='relu')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--der1', type=float, default=0.2, help='Drop edge ratio of the 1st augmentation.')
parser.add_argument('--der2', type=float, default=0.2, help='Drop edge ratio of the 2nd augmentation.')
parser.add_argument('--dfr1', type=float, default=0.2, help='Drop feature ratio of the 1st augmentation.')
parser.add_argument('--dfr2', type=float, default=0.2, help='Drop feature ratio of the 2nd augmentation.')
parser.add_argument("--threshold", type=float, default=0, help='threshold')
parser.add_argument("--split_size", type=int, default=256, help='split_size')
parser.add_argument('--coeff1', type=float, default=1.0, help='')
parser.add_argument('--coeff2', type=float, default=1.0, help='')

args = parser.parse_args()

citegraph = ['cora', 'citeseer', 'pubmed']
cograph = ['photo', 'comp', 'cs', 'physics']

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
seed = 43


grid = {
    'dre1':[0.4,0.5,0.6,0.7,0.8,0.9],
    'dre2':[0.1,0.5,0.6,0.7,0.8,0.9],
    'dfr1':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'dfr2':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'temp':[0.1,0.15,0.2,0.3,0.4]
}



if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)



    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    in_dim = feat.shape[1]
    graph_cuda = graph.to(args.device)
    graph_cuda = graph_cuda.remove_self_loop().add_self_loop()
    # graph = graph.to(args.device)
    feat = feat.to(args.device)
    label = labels.to(args.device)
    n_node = graph.number_of_nodes()

    lbl1 = th.ones(n_node)
    lbl2 = th.zeros(n_node)
    lbl = th.cat((lbl1, lbl2))
    lbl = lbl.to(args.device)

    # Step 3: Create model =================================================================== #
    model = Model(in_dim, hid_dim, out_dim, num_layers, act_fn, temp, args)
        
    model = model.to(args.device)
    # print(f'# params: {count_parameters(model)}')

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)
        
        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        
        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2, epoch, args.threshold, args.split_size, args)
        loss.backward()
        optimizer.step()


    # Step 5: Linear evaluation ============================================================== #

        if epoch > args.epochs-2:
            embeds = model.get_embedding_1(graph_cuda, feat)
            train_embs = embeds[train_idx]
            val_embs = embeds[val_idx]
            test_embs = embeds[test_idx]

            
            # visualize_embeddings(embeds[test_idx], label[test_idx], 'embeds_1.png')

            train_labels = label[train_idx]
            val_labels = label[val_idx]
            test_labels = label[test_idx]

            train_feat = feat[train_idx]
            val_feat = feat[val_idx]
            test_feat = feat[test_idx]

            logreg = LogReg(train_embs.shape[1], num_class)
            opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

            logreg = logreg.to(args.device)
            loss_fn = nn.CrossEntropyLoss()

            best_val_acc = 0
            eval_acc = 0
            for epoch in range(1, 2000):
                logreg.train()
                opt.zero_grad()
                logits = logreg(train_embs)
                preds = th.argmax(logits, dim=1)
                train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
                loss = loss_fn(logits, train_labels)
                loss.backward()
                opt.step()

                logreg.eval()
                with th.no_grad():
                    val_logits = logreg(val_embs)
                    test_logits = logreg(test_embs)

                    val_preds = th.argmax(val_logits, dim=1)
                    test_preds = th.argmax(test_logits, dim=1)

                    val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                    test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        if test_acc > eval_acc:
                            eval_acc = test_acc

            print(eval_acc.item())
                
            
