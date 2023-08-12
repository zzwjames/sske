import argparse
from model import Grace, Model
from aug import aug
from dataset import load

import numpy as np
import torch as th
import torch.nn as nn

from eval import label_classification
import warnings

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=str, default='random')

parser.add_argument('--epochs', type=int, default=500, help='Number of training periods.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')

parser.add_argument('--act_fn', type=str, default='relu')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--der1', type=float, default=0.2, help='Drop edge ratio of the 1st augmentation.')
parser.add_argument('--der2', type=float, default=0.2, help='Drop edge ratio of the 2nd augmentation.')
parser.add_argument('--dfr1', type=float, default=0.2, help='Drop feature ratio of the 1st augmentation.')
parser.add_argument('--dfr2', type=float, default=0.2, help='Drop feature ratio of the 2nd augmentation.')
parser.add_argument("--threshold", type=float, default=0, help='threshold')
parser.add_argument("--split_size", type=int, default=128, help='split_size')
args = parser.parse_args()

citegraph = ['cora', 'citeseer', 'pubmed']
cograph = ['photo', 'comp', 'cs', 'physics']

# if args.dataname in citegraph:
#     args.split='public'

# if args.dataname in cograph:
#     args.split='random'

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':

    # Step 1: Load hyperparameters =================================================================== #
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
    graph, feat, labels, train_mask, test_mask = load(args.dataname)
    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Model(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f'# params: {count_parameters(model)}')

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2, epoch, args.threshold, args.split_size)
        loss.backward()
        optimizer.step()

        print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")

    # graph = graph.add_self_loop()
    # graph = graph.to(args.device)
    # feat = feat.to(args.device)
    # embeds_1 = model.get_embedding_1(graph, feat)
    # embeds_2 = model.get_embedding_2(graph, feat)

    # '''Evaluation Embeddings  '''
    # label_classification(embeds_1, labels, train_mask, test_mask, split=args.split)
    # label_classification(embeds_2, labels, train_mask, test_mask, split=args.split)

    num_samples = 3300  # Number of nodes to sample
    np.random.seed(42)  # Set a seed for reproducibility
    sampled_nodes = np.random.choice(graph.number_of_nodes(), num_samples, replace=False)
    same_label_neighbors = []

    for node in sampled_nodes:
        # Get the neighbors
        _, neighbors = graph.in_edges(node)
        neighbors = neighbors.tolist()
        
        # Get the labels of the neighbors
        neighbor_labels = labels[neighbors]
        
        # Check which neighbors have the same label as the node
        # Check which neighbors have the same label as the node
        same_label = (neighbor_labels == labels[node]).nonzero().squeeze()

        # If same_label is a number, convert it to a list with one element
        if same_label.dim() == 0:
            same_label = [same_label.item()]
        else:
            same_label = same_label.tolist()

        # Store the nodes with the same label
        same_label_neighbors.extend(np.array(neighbors)[same_label])


    same_label_neighbors = th.tensor(same_label_neighbors).unique()  # Remove duplicates

    # ... (your training code)

    # Step 5: Linear evaluation (now only on the nodes with the same label neighbors)
    print("=== Final ===")

    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    feat = feat.to(args.device)
    embeds_1 = model.get_embedding_1(graph, feat)
    embeds_2 = model.get_embedding_2(graph, feat)

    # Create a mask for the nodes of interest
    mask = th.zeros(len(labels), dtype=bool)
    mask[same_label_neighbors] = True
    # print(mask)

    # Evaluate embeddings only on the nodes of interest
    label_classification(embeds_1, labels, mask, mask, split=args.split)
    label_classification(embeds_2, labels, mask, mask, split=args.split)
