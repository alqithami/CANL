"""One-view supervised contrastive feature penalty; not original two-view SupCon training.
Khosla et al., NeurIPS 2020, Eq. (2), with temperature=base_temperature=0.1.
No projection head; average the penalties of the same layers used for DCR.
"""
import torch
import torch.nn.functional as F

def supcon(features,labels,temperature=.1):
    if temperature<=0 or features.ndim!=2 or len(features)!=len(labels):raise ValueError('Invalid SupCon input')
    z=F.normalize(features.float(),dim=1,eps=1e-6);n=len(z)
    diagonal=torch.eye(n,dtype=torch.bool,device=z.device)
    positives=labels[:,None].eq(labels[None,:]) & ~diagonal
    counts=positives.sum(1)
    if n<2 or (counts==0).any():raise ValueError('Every anchor needs another same-class example')
    scores=z@z.T/temperature
    logden=torch.logsumexp(scores.masked_fill(diagonal,-torch.inf),dim=1)
    logprob=scores-logden[:,None]
    return -((logprob*positives).sum(1)/counts).mean()

def contrastive_penalty(features,labels,coefficient,temperature=.1):
    if coefficient<0:raise ValueError('Negative contrastive strength')
    return coefficient*torch.stack([supcon(z,labels,temperature) for z in features.values()]).mean()
