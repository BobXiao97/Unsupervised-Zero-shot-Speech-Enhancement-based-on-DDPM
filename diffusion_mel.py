from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os
import pyworld as pw
import librosa
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import UNet2DModel
device='cuda'

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t, return_dict=False)

        # MSE loss
        return F.mse_loss(noise, eps_theta[0])

x_train=torch.load('clean_data_all.pt')
train_dataset=Data.TensorDataset(x_train)
train_dataloader=Data.DataLoader(train_dataset,batch_size=16,shuffle=True)

x_test=torch.load('clean_test.pt')
test_dataset=Data.TensorDataset(x_test)
test_dataloader=Data.DataLoader(test_dataset,batch_size=16,shuffle=True)
epoch=500

model=UNet2DModel(sample_size=[128,256],
                    in_channels=1,
                    out_channels=1,
                    block_out_channels=[64,64,128,128,256,256],
                    down_block_types=["DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"],
                    up_block_types=["UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"])
#model=nn.DataParallel(model) # use multiple gpus to train
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.00002)
train_loss_list=[]
test_loss_list=[]

for i in range(epoch):
  #print('epoch: '+str(i+1))
  train_total_loss=0
  model.train()
  with tqdm(total=len(train_dataloader)) as pbar:
    pbar.set_description('epoch '+str(i+1))
    for j,x in enumerate(train_dataloader):
        pbar.update(1)
        x=x[0]
        x=x.to(device)
        diffusion=DenoiseDiffusion(eps_model=model,n_steps=200,device=device)
        loss=diffusion.loss(x0=x)
        train_total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  model.eval()
  test_total_loss=0
  with torch.no_grad():
    for j,x in enumerate(test_dataloader):
        x=x[0]
        x=x.to(device)
        diffusion=DenoiseDiffusion(eps_model=model,n_steps=200,device=device)
        loss=diffusion.loss(x0=x)
        test_total_loss+=loss.item()
  if (i+1)%5==0:
    name='model_at_epoch_'+str(i+1)
    torch.save(model,'/data/tianqi/denoising_model/diffusion/checkpoints/'+name+'.pt')
    print('training loss: '+str(train_total_loss/len(train_dataloader)))
    print('testing loss: '+str(test_total_loss/len(test_dataloader)))

  train_loss_list.append(train_total_loss/len(train_dataloader))
  test_loss_list.append(test_total_loss/len(test_dataloader))

  x=list(range(1,i+2))
  plt.plot(x,train_loss_list)
  plt.plot(x,test_loss_list)
  plt.savefig('plot.png')

torch.save(model,'diffusion_model_mel.pt')
