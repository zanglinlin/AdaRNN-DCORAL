import torch.nn as nn

def cosine(source, target):
    source, target = source.mean(dim=1), target.mean(dim=1)
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()