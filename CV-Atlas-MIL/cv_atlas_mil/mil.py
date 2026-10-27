"""Weakly supervised attention MIL. No pixel-level supervision is assumed."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class AtlasMIL(nn.Module):
    def __init__(self,d=20):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(d,36),nn.ReLU(),nn.Linear(36,20),nn.ReLU())
        self.logit=nn.Linear(20,1)
        self.attn=nn.Sequential(nn.Linear(20,16),nn.Tanh(),nn.Linear(16,1))
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x,explain=False):
        h=self.embed(x)
        evidence=self.logit(h).squeeze(-1)
        weights=torch.softmax(self.attn(h).squeeze(-1),dim=1)
        z=.55*torch.topk(evidence,4,dim=1).values.mean(1)+.45*(weights*evidence).sum(1)+self.bias
        if explain:return z,evidence,weights
        return z


def vectors(G,regions,normal):
    center,scale=normal
    geo=np.clip((G-center)/scale,-8,8).astype(np.float32)
    raw=np.concatenate((geo,np.log1p(np.maximum(regions,0))),axis=2)
    row=np.repeat(np.arange(8),8).astype(np.float32)/3.5-1
    col=np.tile(np.arange(8),8).astype(np.float32)/3.5-1
    coord=np.broadcast_to(np.stack([row,col],axis=1),(len(G),64,2))
    return np.ascontiguousarray(np.concatenate([raw,coord],axis=2),dtype=np.float32)


def train_model(x,y,epochs=35,seed=20261025,threads=2):
    torch.set_num_threads(int(threads))
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
    model=AtlasMIL(x.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.008)
    tx=torch.from_numpy(x);ty=torch.from_numpy((np.asarray(y)==0).astype(np.float32))
    rng=np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/96)))):
            logits,inst,_=model(tx[ids],True)
            loss=F.binary_cross_entropy_with_logits(logits,ty[ids])
            healthy=(ty[ids]<.5)
            if healthy.any():loss=loss+.02*F.softplus(inst[healthy]).mean()
            opt.zero_grad();loss.backward();opt.step()
    model.eval()
    return model


def infer_model(model, x):
    model.eval()
    probs=[];attns=[]
    with torch.no_grad():
        for st in range(0,len(x),128):
            z,_,w=model(torch.from_numpy(x[st:st+128]),True)
            probs.append(torch.sigmoid(z).numpy())
            attns.append(w.numpy())
    return np.concatenate(probs),np.concatenate(attns)


def state_to_numpy(model):
    return {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}


def from_numpy(weights,d=20):
    model=AtlasMIL(d)
    model.load_state_dict({k:torch.from_numpy(np.asarray(v)) for k,v in weights.items()})
    model.eval()
    return model
