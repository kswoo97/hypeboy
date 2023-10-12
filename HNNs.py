import math
import torch
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional, Callable

from sklearn.metrics import average_precision_score as apscore # True / Pred
from sklearn.metrics import roc_auc_score as auroc

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as Linear2
from torch_geometric.nn.inits import zeros

from tqdm import tqdm
    
class UniGCNIIConv(MessagePassing):
    
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        self.W = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, X, hyperedge_index, alpha, beta, X0, degV, degE, aug_weight = None, give_edge = False) :
        
        if give_edge : 
            
            num_nodes = X.shape[0]
            num_edges = int(hyperedge_index[1][-1]) + 1
            
            De = scatter_add(X.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)
            
            De_inv = 1.0 / De
            De_inv[De_inv == float('inf')] = 0

            norm_n2e = De_inv[hyperedge_index[1]]
            
            N = X.shape[0]            
            Xe = self.propagate(hyperedge_index, x=X, norm=norm_n2e,
                               size=(num_nodes, num_edges))  # Node to edge: First pooling
            Xe = Xe * degE 
            Xv = self.propagate(hyperedge_index.flip([0]), x=Xe, norm=None,
                               size=(num_edges, num_nodes))  # Edge to node
            
            Xv = Xv * degV

            X = Xv 

            Xi = (1-alpha) * X + alpha * X0
            X = (1-beta) * Xi + beta * self.W(Xi)
            
            return X, Xe
            
        elif aug_weight is not None : 
            
            num_nodes = X.shape[0]
            num_edges = int(hyperedge_index[1][-1]) + 1
            
            De = scatter_add(X.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)
            
            De_inv = 1.0 / De
            De_inv[De_inv == float('inf')] = 0

            norm_n2e = De_inv[hyperedge_index[1]]
            
            N = X.shape[0]            
            Xe = self.propagate(hyperedge_index, x=X, norm=norm_n2e,
                               size=(num_nodes, num_edges), aug_weight = aug_weight)  # Node to edge: First pooling
            Xe = Xe * degE 
            Xv = self.propagate(hyperedge_index.flip([0]), x=Xe, norm=None,
                               size=(num_edges, num_nodes), aug_weight = aug_weight)  # Edge to node
            
            Xv = Xv * degV

            X = Xv 

            Xi = (1-alpha) * X + alpha * X0
            X = (1-beta) * Xi + beta * self.W(Xi)
            
            return X, Xe
            
        else : 
        
            N = X.shape[0]

            Xve = X[hyperedge_index[0]] # [nnz, C]
            Xe = scatter(Xve, hyperedge_index[1], dim=0, reduce='mean') # [E, C]
            Xe = Xe * degE 

            Xev = Xe[hyperedge_index[1]] # [nnz, C]
            Xv = scatter(Xev, hyperedge_index[0], dim=0, reduce='sum', dim_size=N) # [N, C]
            Xv = Xv * degV

            X = Xv 

            Xi = (1-alpha) * X + alpha * X0
            X = (1-beta) * Xi + beta * self.W(Xi)

            return X
    
    def reset_parameters(self):
        self.W.reset_parameters()
    
def get_degree_of_hypergraph(hyperedge_index, device) : ## For UNIGCNII

    ones = torch.ones(hyperedge_index[0].shape[0], dtype = torch.int64).to(device)
    dV = scatter(src = ones, 
        index = hyperedge_index[0], reduce = 'sum')

    dE = scatter(src = dV[hyperedge_index[0]], 
                index= hyperedge_index[1], reduce = 'mean')

    dV = dV.pow(-0.5)
    dE = dE.pow(-0.5)
    dV[dV.isinf()] = 1
    dE[dE.isinf()] = 1

    del ones

    return dV.reshape(-1, 1), dE.reshape(-1, 1)

class HyperEncoder(nn.Module):
    
    def __init__(self, in_dim, edge_dim, node_dim, drop_p = 0.5, num_layers=2, cached = False):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = torch.nn.ReLU()
        self.DropLayer = torch.nn.Dropout(p=drop_p)
        self.convs = nn.ModuleList()

        self.lamda, self.alpha = 0.5, 0.1
        self.convs.append(torch.nn.Linear(self.in_dim, self.node_dim))
        for _ in range(self.num_layers) : 
            self.convs.append(UniGCNIIConv(self.node_dim, self.node_dim))
                
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int) :
        
        degV, degE = get_degree_of_hypergraph(hyperedge_index, hyperedge_index.device) # Getting degree
        degV = degV.reshape(-1,1)
        degE = degE.reshape(-1,1)

        x = self.DropLayer(x)
        x = torch.relu(self.convs[0](x))
        x0 = x

        lamda, alpha = 0.5, 0.1

        for i,conv in enumerate(self.convs[1:]) : 
            x = self.DropLayer(x)
            beta = math.log(lamda/(i+1)+1)
            if i == len(self.convs[1:]) - 1 : 
                x = conv(x, hyperedge_index, alpha, beta, x0, degV, degE)
            else :
                x = conv(x, hyperedge_index, alpha, beta, x0, degV, degE)
                x = torch.relu(x)
        return x # Only Returns Node Embeddings
    
class HyperDecoder(nn.Module):
    
    def __init__(self, in_dim, edge_dim, node_dim, drop_p = 0.5, num_layers=2, cached = False, 
                device = 'cuda:0'):
        super(HyperDecoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = torch.nn.ReLU()
        self.DropLayer = torch.nn.Dropout(p=drop_p)

        self.convs = nn.ModuleList()
                
        self.lamda, self.alpha = 0.5, 0.1
        
        self.convs.append(torch.nn.Linear(self.in_dim, self.node_dim))
        for _ in range(self.num_layers) : 
            self.convs.append(UniGCNIIConv(self.node_dim, self.node_dim))
                
        self.reset_parameters()
        w1 = torch.empty(node_dim)
        w2 = torch.empty(in_dim)
        
        self.input_mask = torch.nn.Parameter(torch.zeros(node_dim), requires_grad = True)
        self.embedding_mask = torch.nn.Parameter(torch.zeros(in_dim), requires_grad = True)
                
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int) :
        
        degV, degE = get_degree_of_hypergraph(hyperedge_index, hyperedge_index.device) # Getting degree
        degV = degV.reshape(-1,1)
        degE = degE.reshape(-1,1)

        x = self.DropLayer(x)
        x = torch.relu(self.convs[0](x))
        x0 = x

        lamda, alpha = 0.5, 0.1

        for i,conv in enumerate(self.convs[1:]) : 
            x = self.DropLayer(x)
            beta = math.log(lamda/(i+1)+1)
            if i == len(self.convs[1:]) - 1 : 
                x = conv(x, hyperedge_index, alpha, beta, x0, degV, degE)
            else :
                x = torch.relu(conv(x, hyperedge_index, alpha, beta, x0, degV, degE))
                    
        return x # Only Returns Node Embeddings
        
class end2endNN(nn.Module) : # HNN Fine-Tuning
    
    def __init__(self, encoder, hidden_dim, n_class) : 
        super(end2endNN, self).__init__() 
        
        self.encoder = encoder
        self.classifier1 = torch.nn.Linear(hidden_dim, n_class)
        
    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int) : 
        Z = self.encoder(x = x, hyperedge_index = hyperedge_index, num_nodes = num_nodes, num_edges = num_edges)
        Z = self.classifier1(Z) # No need of Logits
        
        return Z
    
class MLP(nn.Module) : 
    
    def __init__(self, in_dim, hidden_dim, n_class) :
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, n_class)
        
    def forward(self, x) : 
        x = self.linear1(x)
        return x
    
class MLP_HENN(nn.Module) :
    
    def __init__(self, in_dim, hidden_dim, p = 0.5) : 
        super(MLP_HENN, self).__init__() 
        
        self.classifier1 = torch.nn.Linear(in_dim, hidden_dim)
        self.classifier2 = torch.nn.Linear(hidden_dim, 1)
        self.dropouts = torch.nn.Dropout(p = p)
        
    def forward(self, x, target_nodes, target_ids: list) : 
        
        Z = scatter(src = x[target_nodes, :], index = target_ids, dim = 0, reduce = 'sum')
        Z = (self.classifier1(Z)) # No need of Logits
        Z = torch.relu(Z)
        Z = self.dropouts(Z)
        Z = (self.classifier2(Z)) # No need of Logits
        
        return torch.sigmoid(Z).squeeze(-1) # Edge Prediction Probability
    
class key_query_mapper(nn.Module) : 
    
    def __init__(self, hidden_dim) :
        super(key_query_mapper, self).__init__()
        
        self.W1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, Z) : 
        
        return self.W2(torch.relu(self.W1(Z))) # Encode given embedding
    
def GNN_evaluator(model, X, hyperedge_index, Y, test_idx) : 
    with torch.no_grad() :
        model.eval()
        n_node, n_edge = torch.max(hyperedge_index[0]) + 1, torch.max(hyperedge_index[1]) + 1
        pred_label = torch.argmax(model(X, hyperedge_index, n_node, n_edge)[test_idx], 1)
        acc = (torch.sum(pred_label == Y[test_idx]).to('cpu'))/(pred_label.shape[0])
        return acc
    
def evaluate(model, X, Y, test_idx) :
    
    with torch.no_grad() : 
        model.eval()
        pred = torch.argmax(model(X), dim = 1)
        acc = torch.sum((pred == Y)[test_idx])/len(test_idx)
        return acc.to('cpu').detach().item()
    
def HE_evaluator(model, X, E, n_node, n_edge, test_Vs, test_IDXs, labels, device) : 
    
    with torch.no_grad() : 
        model.eval()
        pred = model(X, test_Vs, test_IDXs).to('cpu').detach().squeeze(-1).numpy()
        score = auroc(labels, pred)
        return score
    
def train_MLP(classifier, encoder, X, Y, E, train_idx, valid_idx, test_idx, 
              lr = 1e-3, epochs = 200, w_decay = 1e-6, device = 'cuda:0') : 
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr = lr, weight_decay = w_decay)
    criterion = torch.nn.CrossEntropyLoss()
    valid_acc = 0
    if encoder != None : 
        encoder = encoder.to(device)
        Y = Y.to(device)
        E = E.to(device)
        with torch.no_grad() : 
            encoder.eval()
            Z = encoder(X, E, torch.max(E[0]) + 1, torch.max(E[1]) + 1).detach()
        Z = Z.to('cpu').detach().to(device) # Using embedding as a Feature Matrix
        
        for ep in range(epochs) : 
            classifier.train()
            optimizer.zero_grad()
            pred = classifier(Z)[train_idx, :]
            loss = criterion(pred, Y[train_idx])
            loss.backward()
            optimizer.step()

            if ep % 10 == 0 :     
                cur_valid = evaluate(classifier, Z, Y, valid_idx)
                if cur_valid > valid_acc : 

                    valid_acc = cur_valid
                    param = copy.deepcopy(classifier.state_dict())

        classifier.load_state_dict(param)
        test_acc = evaluate(classifier, Z, Y, test_idx)
    else : 
        Y = Y.to(device)
        X = X.to(device)
        for ep in range(epochs) : 
            classifier.train()
            optimizer.zero_grad()
            pred = classifier(X)[train_idx, :]
            loss = criterion(pred, Y[train_idx])
            loss.backward()
            optimizer.step()

            if ep % 10 == 0 :     
                cur_valid = evaluate(classifier, X, Y, valid_idx)
                if cur_valid > valid_acc : 

                    valid_acc = cur_valid
                    param = copy.deepcopy(classifier.state_dict())

        classifier.load_state_dict(param)
        test_acc = evaluate(classifier, X, Y, test_idx)
    
    return float(valid_acc), float(test_acc)

def train_HE_predictor(model, X, edge_buckets, 
                       lr = 1e-3, epochs = 200, device = "cuda:0", seed = 0) :
        
    train_vidx = torch.tensor(edge_buckets[0][0]).to(device)
    train_eidx = torch.tensor(edge_buckets[0][1]).to(device)
    train_label = torch.tensor(edge_buckets[0][2]).to(device)
    
    valid_vidx = torch.tensor(edge_buckets[1][0]).to(device)
    valid_eidx = torch.tensor(edge_buckets[1][1]).to(device)
    valid_label = torch.tensor(edge_buckets[1][2]).numpy()
    
    test_vidx = torch.tensor(edge_buckets[2][0]).to(device)
    test_eidx = torch.tensor(edge_buckets[2][1]).to(device)
    test_label = torch.tensor(edge_buckets[2][2]).numpy()
    
    edges = edge_buckets[3].to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-6)
    criterion = torch.nn.BCELoss()
        
    X = X.to(device)
    valid_score = 0
    loss_lists = []

    for ep in (range(epochs)) : 

        model.train()
        optimizer.zero_grad()
        pred = model(X, train_vidx, train_eidx)
        loss = criterion(pred, train_label)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0: 
            cur_score = HE_evaluator(model = model, X = X, E = None, 
                                 n_node = None,  n_edge = None, 
                                 test_Vs = valid_vidx, test_IDXs = valid_eidx, 
                                 labels = valid_label, device = device)

            if cur_score > valid_score : 
                valid_score = cur_score
                param = copy.deepcopy(model.state_dict())

    model.load_state_dict(param)
    test_score = HE_evaluator(model = model, X = X, E = None, 
                                 n_node = None,  n_edge = None, 
                                 test_Vs = test_vidx, test_IDXs = test_eidx, 
                                 labels = test_label, device = device)
    
    return float(valid_score), float(test_score)

def train_FineTuning(model, X, H, Y, train_idx, valid_idx, test_idx, lr = 1e-3, w_decay = 1e-6, epochs = 200) : 
    
    n_class = torch.unique(Y).shape[0]
    n_node = X.shape[0]
    n_edge = int(torch.max(H[1]) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = w_decay)
    criterion = torch.nn.CrossEntropyLoss()
    valid_score = 0
    
    for ep in (range(200)) : 
        model.train()
        optimizer.zero_grad()
        pred = model(x = X, hyperedge_index = H, num_nodes = n_node, num_edges = n_edge)[train_idx, :]
        loss = criterion(pred, Y[train_idx])
        loss.backward()
        optimizer.step()
        if (ep + 1) % 10 == 0 : 
            val_acc = GNN_evaluator(model = model, X = X, hyperedge_index = H, 
                                    Y = Y, test_idx = valid_idx)
            if val_acc > valid_score : 
                params = copy.deepcopy(model.state_dict())
                valid_score = val_acc
                best_ep = ep

    model.load_state_dict(params)
    test_acc = GNN_evaluator(model = model, X = X, hyperedge_index = H, Y = Y, test_idx = test_idx)
    
    return float(valid_score), float(test_acc)