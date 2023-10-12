import random
import copy
import itertools
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt
from torch_scatter import scatter
from tqdm import tqdm

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
class related_tool() : 
    
    def __init__(self, X, edge, d_name, device) : 

        
        ## Create pre-requisite materials for hyperedge filling
        
        edge_index = edge.to('cpu').numpy().copy()
        self.edges = edge.clone()
        self.edges = self.edges.to(device)
        self.device = device
        self.pairwise_srcs = []
        self.pairwise_targets = []
        
        e_idx = 0
        prev_indptr = 0

        self.edge_dict = dict()
        self.totalE = []
        
        for i in range(edge_index.shape[1]) : 
            cur_edge_index = edge_index[1][i]
            if e_idx != cur_edge_index : 
                if i - prev_indptr > 1 : # Size is greater than 2
                    self.edge_dict[e_idx] = (list(edge_index[0][prev_indptr : i]))
                    self.totalE.append(list(edge_index[0][prev_indptr : i]))
                        
                e_idx = cur_edge_index
                prev_indptr = i
            
        if i - prev_indptr > 1 : # Size is greater than 2
            self.edge_dict[e_idx] = (list(edge_index[0][prev_indptr : i + 1]))
            self.totalE.append(list(edge_index[0][prev_indptr : i + 1]))
            
        eidx = 0
        n_total_negs = 0
            
        if d_name in ['dblp_coauth', 'news'] : 
            
            pushVs = []
            pushIDX = []
            pullsrc = []

            for i, e in enumerate(self.totalE) :

                Vs = e

                for k in range(len(Vs)) : 
                    pushVs.extend(Vs[:k] + Vs[(k+1):])
                    pushIDX.extend([eidx] * (len(Vs) - 1))
                    pullsrc.append(Vs[k])
                    eidx += 1

            self.target = torch.tensor(pushIDX).to(self.device)
            self.pushVs = pushVs
            self.pullsrc = pullsrc
            
            self.batch = True
            self.total_target = []
            self.total_pushVs = []
            self.total_pullsrc = []
            
            if d_name in ['dblp_coauth'] : 
                batch_size = 5000
            else : # news
                batch_size = 10
                
            n_batch = (len(self.totalE) // batch_size) + 1
            
            for idx in range(n_batch) : 

                t_start , t_end = int(idx * batch_size) , int((idx + 1) * batch_size)
                pushVs = []
                pushIDX = []
                pullsrc = []
                eidx = 0

                for i, e in enumerate(self.totalE[t_start:t_end]) : 

                    Vs = e

                    for k in range(len(Vs)) : 
                        pushVs.extend(Vs[:k] + Vs[(k+1):])
                        pushIDX.extend([eidx] * (len(Vs) - 1))
                        pullsrc.append(Vs[k])
                        eidx += 1

                target = torch.tensor(pushIDX).to(self.device) 
                self.total_target.append(target)
                self.total_pushVs.append(pushVs)
                self.total_pullsrc.append(pullsrc)
            
        else : 
            self.batch = False
            pushVs = []
            pushIDX = []
            pullsrc = []

            for i, e in enumerate(self.totalE) :

                Vs = e

                for k in range(len(Vs)) : 
                    pushVs.extend(Vs[:k] + Vs[(k+1):])
                    pushIDX.extend([eidx] * (len(Vs) - 1))
                    pullsrc.append(Vs[k])
                    eidx += 1

            self.target = torch.tensor(pushIDX).to(self.device)
            self.pushVs = pushVs
            self.pullsrc = pullsrc
            
            
    def return_loss(self, Z, head1, head2, encoding_type = 'head') : 
        
        loss1 = 0
        loss2 = 0
        
        if encoding_type == 'head' : # With head
            
            if self.batch : # batchwise loss computation

                normZ = torch.nn.functional.normalize(head1(Z), p=2.0, dim=1, eps=1e-12, out=None)

                for i in range(len(self.total_target)) : 

                    pushVs = self.total_pushVs[i]
                    target = self.total_target[i]
                    pullsrc = self.total_pullsrc[i]
                    aggZ = scatter(src = Z[pushVs, :], index = target, reduce = 'sum', dim = 0)
                    aggZ = torch.nn.functional.normalize(head2(aggZ), p=2.0, dim=1, eps=1e-12, out=None)    
                    denom = torch.mm(aggZ, normZ.transpose(1,0))
                    loss1 += -torch.sum(denom[range(len(pullsrc)), pullsrc])
                    loss2 += torch.sum(torch.logsumexp(denom, dim = 1))
            
            else : 
                
                aggZ = scatter(src = Z[self.pushVs, :], index = self.target, reduce = 'sum', dim = 0)
                Z = torch.nn.functional.normalize(head1(Z), p=2.0, dim=1, eps=1e-12, out=None)
                aggZ = torch.nn.functional.normalize(head2(aggZ), p=2.0, dim=1, eps=1e-12, out=None)
                denom = torch.mm(aggZ, Z.transpose(1,0))
                loss1 += -torch.sum(denom[range(len(self.pullsrc)), self.pullsrc])
                loss2 += torch.sum(torch.logsumexp(denom, dim = 1))
            
        else :  # W/O Head
            
            if self.batch : # batchwise loss computation

                normZ = torch.nn.functional.normalize(Z, p=2.0, dim=1, eps=1e-12, out=None)

                for i in range(len(self.total_target)) : 

                    pushVs = self.total_pushVs[i]
                    target = self.total_target[i]
                    pullsrc = self.total_pullsrc[i]
                    aggZ = scatter(src = Z[pushVs, :], index = target, reduce = 'sum', dim = 0)
                    aggZ = torch.nn.functional.normalize(aggZ, p=2.0, dim=1, eps=1e-12, out=None)    
                    denom = torch.mm(aggZ, normZ.transpose(1,0))
                    loss1 += -torch.sum(denom[range(len(pullsrc)), pullsrc])
                    loss2 += torch.sum(torch.logsumexp(denom, dim = 1))
            else : 
                aggZ = scatter(src = Z[self.pushVs, :], index = self.target, reduce = 'sum', dim = 0)
                Z = torch.nn.functional.normalize(Z, p=2.0, dim=1, eps=1e-12, out=None)
                aggZ = torch.nn.functional.normalize(aggZ, p=2.0, dim=1, eps=1e-12, out=None)
                denom = torch.mm(aggZ, Z.transpose(1,0))
                loss1 += -torch.sum(denom[range(len(self.pullsrc)), self.pullsrc])
                loss2 += torch.sum(torch.logsumexp(denom, dim = 1))
                
        return loss1 + loss2
    
def augment_edge(edge_dict, p, n_node, device) :
    
    isolated_nodes = {i : 0 for i in range(n_node)}
    n_given = len(edge_dict) - int(p * len(edge_dict))
    given_edges = []
    e_given = np.random.choice(a = list(edge_dict.keys()), size = n_given, replace = False) # Drop
    e_given = {i : 0 for i in e_given}
    
    EDGEs = []
    VIDXs = []
    EIDXs = []
    eidx = 0
    non_given = []
    for e in edge_dict : 
        try : 
            e_given[e]
            for v in edge_dict[e] : 
                try : 
                    del isolated_nodes[v]
                except : 
                    None
            VIDXs.extend(edge_dict[e])
            EIDXs.extend([eidx] * len(edge_dict[e]))
            given_edges.append(edge_dict[e])
            eidx += 1
        except : 
            EDGEs.append(edge_dict[e])
            non_given.append(e)
    for v in isolated_nodes : 
        VIDXs.append(v)
        EIDXs.append(eidx)
        eidx += 1
    
    E = torch.tensor([VIDXs, EIDXs]).to(device)
    
    return E
    
def augment_feature(X, p, device) :
    
    return X * (((torch.rand(X.shape) > p).float()).to(device))
    
def hyperedge_filling(X, encoder, head1, head2, device, required_tools, 
                              lr = 1e-3, epoch = 100, 
                              w_decay = 1e-6, head_type = 'head',
                              prob_x = 0.5, prob_e = 0.5, seed = 0) : 
    
    if head_type == 'no_head' : 
        optimizer = torch.optim.Adam(encoder.parameters(),
                                     lr = lr, weight_decay = w_decay) # Optimizer 
    else : 
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head1.parameters()) + list(head2.parameters()), 
                                     lr = lr, weight_decay = w_decay) # Optimizer 

        head1.train()
        head2.train()
    
    encoder.train()
    np.random.seed(seed)
    fixed_feature = X.to(device)
    edge_dict = required_tools.edge_dict
    n_node = X.shape[0]
    
    for ep in (range(epoch)) : 
        
        optimizer.zero_grad()
        curX = augment_feature(fixed_feature, prob_x, device)
        curE = augment_edge(edge_dict, prob_e, n_node, device)
        n_edge = int(curE[1][-1]) + 1
        Z = encoder(curX, curE, n_node, n_edge)
        loss = required_tools.return_loss(Z, head1, head2, head_type)
        loss.backward()
        optimizer.step()
        
    encoder_param = copy.deepcopy(encoder.state_dict())
        
    return encoder_param

def feature_reconstruction(X, encoder, decoder, device, required_tools, 
                       lr = 1e-3, epoch = 100, w_decay = 1e-6, 
                       prob_x = 0.5, prob_e = 0.5) : 
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = w_decay)
    encoder.train()
    decoder.train()
    totalX = X.to(device)
    n_node = X.shape[0]
    n_mask = int(n_node * prob_x)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    edge_dict = required_tools.edge_dict
    
    for ep in range(epoch) : 
        
        optimizer.zero_grad()
        masked_idx = list(np.random.choice(a = n_node, size = n_mask, replace = False))
        masked_idx.sort()
        curX = torch.clone(totalX)
        curX[masked_idx, :] = decoder.input_mask
        curE = augment_edge(edge_dict, prob_e, n_node, device)
        n_edge = int(curE[1][-1]) + 1
        Z1 = encoder(curX, curE, n_node, n_edge)
        Z1[masked_idx, :] = decoder.embedding_mask
        Z2 = decoder(Z1, curE, n_node, n_edge)
        totalL = torch.mean((1 - cos(totalX[masked_idx, :], Z2[masked_idx, :])))
        totalL.backward()
        optimizer.step()
        
    encoder_param = copy.deepcopy(encoder.state_dict())
            
    return encoder_param

def HypeBoy(X, encoder, decoder, head1, head2, device, required_tools, 
                        lr1 = 1e-3, lr2 = 1e-3, epoch1 = 300, epoch2 = 200,
                         prob_x1 = 0.5, prob_x2 = 0.5, prob_e1 = 0.2, prob_e2 = 0.9) : 
    
    # Step 1: Feature Reconstruction
    
    parameters = feature_reconstruction(X, encoder, decoder, device, required_tools, 
                       lr = lr1, epoch = epoch1, w_decay = 1e-6, 
                       prob_x = prob_x1, prob_e = prob_e1)
    encoder.load_state_dict(parameters)

    # Step 2: Hyperedge Filling!

    encoder_parameters = hyperedge_filling(X, encoder, head1, head2, 
                                                        device, required_tools, 
                                                        lr = lr2, 
                                                        epoch = epoch2, w_decay = 1e-6, 
                                                        head_type = 'head',
                                                        prob_x = prob_x2, prob_e = prob_e2)
    
    return encoder_parameters