import torch 
import torch.nn as nn
import numpy as np
import itertools
import torch.nn.functional as F
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(1, 50),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100,100),
        )
        # define the decoder network
        self.decoder = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100,50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

def combination(iterable, r):
    pool = list(iterable)
    n = len(pool)

    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)

def get_triplets(labels):
    triplets = []

    for label in set(labels):
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combination(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))

def triplet_hashing_loss(embedding, labels, margin=1):
    
    triplets = get_triplets(labels)
    ap_distances = (embedding[triplets[:, 0]] - embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances = (embedding[triplets[:, 0]] - embedding[triplets[:, 2]]).pow(2).sum(1)
    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()

def onehot_to_scalar(onehot):
    y=[]
    for x in onehot:
        y.append(x.tolist().index(1))
    return y

