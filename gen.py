import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
from ae import AutoEncoder
from train import generator
import os
import numpy as np
num_class=10
# transform the output of generator to image
def to_img(x):
    out=0.5*(x+1)
    out=out.clamp(0,1)
    out=out.view(-1,1,28,28)
    return out


####### define generator #####
# the input is got from the autoencoder and the output is distributed between -1 and 1
 

AE=AutoEncoder()
G=generator()

AE.load_state_dict(torch.load('autoencoder.pth',map_location=torch.device('cpu')))
G.load_state_dict(torch.load('generator.pth',map_location=torch.device('cpu')))

flabel=torch.ones(200,1)
for i in range(10):
	for j in range(20):
		flabel[20*i+j]=i

flabel=flabel.view(-1,1)
fz, _ = AE(flabel)

fimg=G(fz)
fimages = to_img(fimg.cpu().data)
save_image(fimages, 'fimg.png',nrow=20)