"""Readable reference implementations. Write your version in exercises first."""
import torch
from torch import nn

def fm_interaction(emb):
    # emb: [batch, fields, embedding_dim]; output: [batch]
    return .5 * ((emb.sum(dim=1)**2 - (emb**2).sum(dim=1)).sum(dim=1))

def mlp(in_dim, hidden):
    layers=[]
    for width in hidden:
        layers += [nn.Linear(in_dim,width),nn.ReLU()]
        in_dim=width
    return nn.Sequential(*layers)

class Features(nn.Module):
    def __init__(self, sizes, n_num, dim):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(n,dim,padding_idx=0) for n in sizes])
        for layer in self.embeddings:
            nn.init.normal_(layer.weight,mean=0,std=.01)
            with torch.no_grad():
                layer.weight[0].zero_()
        self.n_num=n_num
        self.out_dim=len(sizes)*dim+n_num

    def forward(self, cat, num):
        emb=torch.stack([layer(cat[:,i]) for i,layer in enumerate(self.embeddings)],dim=1)
        return emb, torch.cat([emb.flatten(start_dim=1),num],dim=1)

class DeepFM(nn.Module):
    """Single target. FM interactions here cover categorical fields only.

    Numerical features enter linear and MLP branches. This documented minimal
    variant is not claimed to be an exact reproduction of every DeepFM library.
    """
    def __init__(self,sizes,n_num,dim=8,hidden=(32,16)):
        super().__init__()
        self.features=Features(sizes,n_num,dim)
        self.first=nn.ModuleList([nn.Embedding(n,1,padding_idx=0) for n in sizes])
        for layer in self.first:
            nn.init.zeros_(layer.weight)
        self.linear_num=nn.Linear(n_num,1)
        self.deep=nn.Sequential(mlp(self.features.out_dim,hidden),nn.Linear(hidden[-1],1))

    def forward(self,cat,num):
        emb,x=self.features(cat,num)
        linear=sum(layer(cat[:,i]).squeeze(-1) for i,layer in enumerate(self.first))+self.linear_num(num).squeeze(-1)
        return (linear+fm_interaction(emb)+self.deep(x).squeeze(-1)).unsqueeze(1)

class SingleMLP(nn.Module):
    """Matched single-task baseline for SharedBottom."""
    def __init__(self,sizes,n_num,dim=8,hidden=(32,16)):
        super().__init__()
        self.features=Features(sizes,n_num,dim)
        self.bottom=mlp(self.features.out_dim,hidden)
        self.head=nn.Linear(hidden[-1],1)

    def forward(self,cat,num):
        _,x=self.features(cat,num)
        return self.head(self.bottom(x))

class SharedBottom(nn.Module):
    def __init__(self,sizes,n_num,dim=8,hidden=(32,16)):
        super().__init__()
        self.features=Features(sizes,n_num,dim)
        self.bottom=mlp(self.features.out_dim,hidden)
        self.heads=nn.ModuleList([nn.Linear(hidden[-1],1) for _ in range(2)])

    def forward(self,cat,num):
        _,x=self.features(cat,num)
        z=self.bottom(x)
        return torch.cat([h(z) for h in self.heads],dim=1)

class MMoE(nn.Module):
    def __init__(self,sizes,n_num,dim=8,hidden=(32,16),experts=3):
        super().__init__()
        self.features=Features(sizes,n_num,dim)
        self.experts=nn.ModuleList([mlp(self.features.out_dim,hidden) for _ in range(experts)])
        self.gates=nn.ModuleList([nn.Linear(self.features.out_dim,experts) for _ in range(2)])
        self.heads=nn.ModuleList([nn.Linear(hidden[-1],1) for _ in range(2)])

    def forward(self,cat,num):
        _,x=self.features(cat,num)
        stack=torch.stack([e(x) for e in self.experts],dim=1)
        logits=[]
        for gate,head in zip(self.gates,self.heads):
            weights=gate(x).softmax(dim=1)
            mixed=(stack*weights.unsqueeze(-1)).sum(dim=1)
            logits.append(head(mixed))
        return torch.cat(logits,dim=1)

def build_model(name,sizes,n_num,dim=8,hidden=(32,16),experts=3):
    if dim<1 or not hidden or min(hidden)<1 or experts<1:
        raise ValueError('Embedding dimension, hidden widths and expert count must be positive.')
    cls={'deepfm':DeepFM,'single':SingleMLP,'shared':SharedBottom,'mmoe':MMoE}[name]
    return cls(sizes,n_num,dim,hidden,experts=experts) if name=='mmoe' else cls(sizes,n_num,dim,hidden)
