#coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
from ae import *
import os
import numpy as np
import argparse
#create the folder 
if not os.path.exists('./img'):
    os.mkdir('./img')

parser = argparse.ArgumentParser("gan")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--load', action='store_true', default=False)
args = parser.parse_args()
# transform the output of generator to image
def to_img(x):
    out=0.5*(x+1)
    out=out.clamp(0,1)
    out=out.view(-1,1,28,28)
    return out
 
batch_size=args.batch_size
num_epoch=args.num_epoch
z_dimension=100
num_class=10
gepoch=1
 
#preprocessing images
img_transform = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,),std=(0.5,))
    ])

 
# mnist dataset 
mnist=datasets.MNIST(
    root='./data/',train=True,transform=img_transform,download=True
)
 
# data loader
dataloader=torch.utils.data.DataLoader(
    dataset=mnist,batch_size=batch_size,shuffle=True
)
 
 
####### define discriminator #####
 

# Expand the picture 28x28 into 784, and then pass it to the middle layer, the output is used for image classification

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,10),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.dis(x)
        return x
 
 
####### define generator #####
# the input is got from the autoencoder and the output is distributed between -1 and 1
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.Dropout(p=0.7),
            nn.ReLU(True),
            nn.Linear(256,784),
            nn.Tanh()# make the output distributed between -1 and 1
        )
 
    def forward(self, x):
        x=self.gen(x)
        return x
 

if __name__ == '__main__':
    #create the object
    D=discriminator()
    G=generator()
    AE = AutoEncoder()
    if args.load:
        D.load_state_dict(torch.load('discriminator.pth',map_location='cpu'))
        G.load_state_dict(torch.load('generator.pth',map_location='cpu'))
        AE.load_state_dict(torch.load('autoencoder.pth',map_location='cpu'))
    if torch.cuda.is_available():
        D=D.cuda()
        G=G.cuda()
        AE=AE.cuda()
    
     

    criterion = nn.BCELoss() #the loss function for discriminator and generator is binary cross entropy loss

    d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
    g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
    ae_optimizer=torch.optim.Adam(AE.parameters(),lr=0.0003)


    for epoch in range(num_epoch): #train for num_epoch times
        for i,(img, label) in enumerate(dataloader):
            num_img=img.size(0)

            # =============================train the discriminator==================
            img = img.view(num_img, -1)  # expand the image to size of 784
            real_img = Variable(img)  # put the tensor into the computation graph
            labels_onehot = np.zeros((num_img,num_class))
            labels_onehot[np.arange(num_img),label.numpy()]=1 # the label for true image is 1 at the corresponding position
            real_label=Variable(torch.from_numpy(labels_onehot).float()) 
            fake_label=Variable(torch.zeros(num_img,num_class))# the label for fake image is 0 at all positions

            label_float =torch.from_numpy(np.array(label)).float()
            label_float=label_float.view(-1,1)


            #cuda-relative
            if torch.cuda.is_available():
                real_img=real_img.cuda()
                real_label=real_label.cuda()
                fake_label=fake_label.cuda()
                label_float=label_float.cuda()

     
            # calculate the loss for true images
            real_out = D(real_img)  
            d_loss_real = criterion(real_out, real_label)  
            real_scores = real_out  # it would be better if real_scores is closer to 1/num_class

            # calculate the loss for fake images
            z, _ = AE(label_float) # generate conditional noise from autodecoder
            fake_img = G(z) 
            fake_out = D(fake_img)  
            d_loss_fake = criterion(fake_out, fake_label) 
            fake_scores = fake_out  # it would be better if fake_scores is closer to 0
     
            # total loss function and optimizer
            d_loss = d_loss_real + d_loss_fake 
            d_optimizer.zero_grad()  # set the gradient to zero before backpropagate
            d_loss.backward()  # backpropagate the loss 
            d_optimizer.step()  # update the parameters

            # =============================train the generator==================
            z, _= AE(label_float)  # generate conditional noise from autodecoder
            fake_img = G(z) 
            output = D(fake_img)  

            # calculate the loss for generator
            g_loss = criterion(output, real_label)  
            g_optimizer.zero_grad()  # set the gradient to zero before backpropagate
            g_loss.backward(retain_graph=False)   # backpropagate the loss 
            g_optimizer.step()  # update the parameters

            # =============================train the autoencoder==================
            if i==0:
                torch.save(AE.state_dict(),'./autoencoder.pth') # save before train

            x=np.arange(num_class) 
            x=np.tile(x,10) # repeat the input to guarantee the same element occur at least twice
            x_float=torch.from_numpy(x).float() # set the type to float
            x_float=x_float.view(-1,1) 

            #cuda-relative
            if torch.cuda.is_available():
                x_float=x_float.cuda()

            z, decoder_out = AE(x_float) # generate output

            mse=nn.MSELoss()

            # calculate the loss for autoencoder
            ae_loss=mse(decoder_out,x_float)+triplet_hashing_loss(z,x) # the total loss for autodecoder
            
            ae_optimizer.zero_grad() # set the gradient to zero before backpropagate
            ae_loss.backward() # backpropagate the loss 
            ae_optimizer.step() # update the parameters
            
            # print the intermediate loss
            if (i+1)%100==0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},'
                      'D real: {:.6f},D fake: {:.6f},AE_loss: {:.6f}'.format(
                    epoch,num_epoch,d_loss.item(),g_loss.item(),
                    real_scores.data.mean(),fake_scores.data.mean() 
                    ,ae_loss.data.mean()
                ))
            
            if epoch==0 and i==0:
                real_images=to_img(real_img.cpu().data) 
                save_image(real_images, './img/real_images.png') # save real images
                
            if i==0:
                flabel=np.arange(num_class)
                flabel=torch.from_numpy(flabel).float()
                flabel=flabel.view(-1,1)

                #cuda-relative
                if torch.cuda.is_available():
                    flabel=flabel.cuda()
                fz, _ = AE(flabel)
                fimg=G(fz) # generate fake images 
                fimages = to_img(fimg.cpu().data)
                string = ''.join(str(i.item()) for i in flabel)
                save_image(fimages, './img/fake_images-{}.png'.format(epoch+1),nrow=5) # save fake images

                
        #save model
        torch.save(G.state_dict(),'./generator.pth')
        torch.save(D.state_dict(),'./discriminator.pth')
