import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import numpy as np
import random

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling
from torch.nn.functional import cosine_similarity
bce_loss = nn.BCEWithLogitsLoss()

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    def forward(self, teacher, student, split_size):
        # N x C
        # N x N x C
        perm=th.randperm(len(teacher))
        teacher=teacher[perm]
        student=student[perm]
        quarter_len = len(teacher) // 4
        indices = random.sample(range(len(teacher)), quarter_len)
        teacher = teacher[indices]
        student = student[indices]
        split_size = split_size  # Choose the split size
        num_splits = (student.size(0) + split_size - 1) // split_size
        padded_size = num_splits * split_size
        student = F.pad(student, (0, 0, 0, padded_size - student.size(0)))
        teacher = F.pad(teacher, (0, 0, 0, padded_size - teacher.size(0)))

        total_loss = 0

        for i in range(num_splits):
            start = i * split_size
            end = start + split_size
            student_split = student[start:end]
            teacher_split = teacher[start:end]

            # with th.no_grad():
            t_d = pdist(teacher_split, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

            d = pdist(student_split, squared=False)
            mean_d = d[d>0].mean()
            d = d / mean_d

            loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
            total_loss += loss

        total_loss /= num_splits
        return total_loss


class RKdAngle(nn.Module):
    def forward(self, teacher, student, split_size):
        # N x C
        # N x N x C
        perm=th.randperm(len(teacher))
        teacher=teacher[perm]
        student=student[perm]
        quarter_len = len(teacher) // 4
        indices = random.sample(range(len(teacher)), quarter_len)
        teacher = teacher[indices]
        student = student[indices]
        split_size = split_size  # Choose the split size
        num_splits = (student.size(0) + split_size - 1) // split_size
        padded_size = num_splits * split_size
        student = F.pad(student, (0, 0, 0, padded_size - student.size(0)))
        teacher = F.pad(teacher, (0, 0, 0, padded_size - teacher.size(0)))

        total_loss = 0

        for i in range(num_splits):
            start = i * split_size
            end = start + split_size
            student_split = student[start:end]
            teacher_split = teacher[start:end]

            # with th.no_grad():
            td = (teacher_split.unsqueeze(0) - teacher_split.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = th.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            sd = (student_split.unsqueeze(0) - student_split.unsqueeze(1))
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = th.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

            loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
            total_loss += loss

        total_loss /= num_splits
        return total_loss


# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers = 2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

        self.convs.append(GraphConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat))

        return feat

# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self,_, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)
class MLP_(nn.Module):
    def __init__(self, nfeat, nhid, use_bn=True):
        super(MLP_, self).__init__()
        # print('nfeat:{}'.format(nfeat))
        # print('nhid:{}'.format(nhid))
        self.layer1 = nn.Linear(nfeat,nhid*2,bias=True)
        self.layer2 = nn.Linear(nhid*2, nhid,bias=True)

        

        self.bn = nn.BatchNorm1d(nhid*2)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        # x = self.layer1(x)
        # if self.use_bn:
        #     x = self.bn(x)

        # x = self.act_fn(x)
        # x = self.layer2(x)
        z = F.elu(self.layer1(x))
        return self.layer2(z)

        return z
# model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
# loss = model(graph1, graph2, feat1, feat2)
class Grace_MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp, lbl, args, use_mlp=False):
        super(Grace_MLP, self).__init__()
        self.encoder = MLP_(in_dim, hid_dim)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lbl=lbl
    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
      

        similarity = cosine_similarity(z1, z2, dim=1)
        # print(similarity)
        return similarity

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    # feat2 is shuffle feat
    def forward(self, graph, feat, graph1, graph2, feat1, feat2):
        # encoding
        h0 = self.encoder(graph, feat)
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        # projection
        z0 = self.proj(h0,h0)
        z1 = self.proj(h1,h1)
        z2 = self.proj(h2,h2)

        # # get loss
        sim1 = self.get_sim(z0, z1)
        sim2 = self.get_sim(z0, z2)
        sim = th.cat((sim1,sim2))
        loss = self.loss_fn(self.lbl,sim)*100
        

        # ret = (l1 + l2) * 0.5

        return loss


class Grace(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp, use_mlp=False):
        super(Grace, self).__init__()
        if use_mlp==True:
            self.encoder = MLP_(in_dim, hid_dim)
        else:
            self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))        # intra-view pairs
        between_sim = f(self.sim(z1, z2))     # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    def forward(self, graph1, graph2, feat1, feat2):
        # encoding
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)

        # projection
        z1 = self.proj(h1,h1)
        z2 = self.proj(h2,h2)

        # # get loss
        # l1 = self.get_loss(z1, z2)
        # l2 = self.get_loss(z2, z1)

        # ret = (l1 + l2) * 0.5

        return z1,z2

class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp, args):
        super(Model, self).__init__()
        self.encoder1=Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
        if args.use_mlp==True:
            self.encoder2=Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp, use_mlp=True)
        else:
            self.encoder2=Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
        self.reg_criterion=RKdAngle()
        self.reg_eucl=RkdDistance()
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)
    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s
    
    def sim_dis(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.abs(th.sum(z1*z2,dim=1))
        s = s.sum()/len(s)
        return s

    def euc_dis(self, z1, z2):

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        diff = z1 - z2

# Square the element-wise differences
        squared_diff = th.pow(diff, 2)

# Sum the squared differences across the columns
        sum_squared_diff = th.sum(squared_diff, dim=1)

# Take the square root of the summed squared differences
        euclidean_dist = th.sqrt(sum_squared_diff)

        euclidean_dist = euclidean_dist.sum()/len(z1)

# Print the Euclidean distance for each row
        return euclidean_dist
    
    def get_loss(self,a,c):
        # between_sim.diag(): positive pairs
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(a)        # intra-view pairs
        between_sim = f(c)     # inter-view pairs
        
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h1 = self.encoder1.encoder(graph, feat)
        h2 = self.encoder2.encoder(graph, feat)
        h = 0.5*h1+0.5*h2
        return h.detach()
    
    def get_embedding_1(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder1.encoder(graph, feat)
        return h.detach()
    def get_embedding_2(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder2.encoder(graph, feat)
        return h.detach()
    
    def forward(self, graph1, graph2, feat1, feat2, epoch, threshold, split_size, args):
        z1,z2=self.encoder1(graph1, graph2, feat1, feat2)
        z1_,z2_=self.encoder2(graph1, graph2, feat1, feat2)


        a=self.sim(z1,z1)
        b=self.sim(z1,z2)
        c=self.sim(z1_,z1_)
        d=self.sim(z1_,z2_)
        prod = th.mul(a, c)

        # print(prod)




        
        
        a = th.where(prod>args.threshold, a, -100)
        c = th.where(prod>args.threshold, c, -100)

        prod1 = th.mul(b, d)
        eye = th.eye(prod1.shape[0]).bool().to(prod1.device)

# Set the diagonal elements of prod to 1
        prod1[eye] = 1
        b = th.where(prod1>args.threshold, b, -100)
        d = th.where(prod1>args.threshold, d, -100)

#         # Compute the total number of elements in `prod`
#         total_elements = prod.numel()
        

# # Create a Boolean mask tensor for elements greater than 0
#         mask = prod > args.threshold

#         # if epoch>args.epochs-2:
# # Count the number of elements greater than 0
#         count = th.sum(mask)

#         mask = prod1 > args.threshold

#         count += th.sum(mask)-len(prod)

# # Calculate the percentage
#         percentage = (count / (total_elements*2-len(prod))) * 100

# # Print the percentage
#         print("Percentage of elements greater than 0:", percentage.item())


        
        loss=0.5*(self.get_loss(a,b)+self.get_loss(b,a)).mean()+0.5*(self.get_loss(c,d)+self.get_loss(d,c)).mean()
        # print('contrastive loss:{}'.format(loss))

        l_ke1 = self.reg_criterion(th.cat((z1,z2),dim=0),th.cat((z1_,z2_),dim=0),split_size)
        
        l_ke1 = l_ke1*args.coeff1
        # print('l_ke1:{}'.format(l_ke1))
        
        # l_ke2 = self.euc_dis(z1,z1_)+self.euc_dis(z2,z2_)
        # l_ke2 = th.exp(-l_ke2)
        # l_ke2 = self.reg_eucl(th.cat((z1_,z2_),dim=0),th.cat((z1,z2),dim=0),split_size)
        
        # l_ke2 = l_ke2*args.coeff2
        # print('l_ke2:{}'.format(l_ke2))
        a = th.cat((z1, z2), dim=0)

        # Concatenate z1_ and z2_ along dimension 0
        b = th.cat((z1_, z2_), dim=0)

        # Compute Euclidean distance for each row of a and b using cdist
        euclidean_distances = th.norm(a - b, dim=1)

        # Calculate the average Euclidean distance per row
        l_ke2=average_euclidean_distance = th.mean(euclidean_distances)
        
        loss += l_ke1
        
        loss += th.exp(-l_ke2)*0.00001
        
        
        
        return loss




    
