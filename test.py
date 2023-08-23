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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets
from PIL import Image
import soundfile as sf
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
        eps_theta = self.eps_model(xt, t, return_dict=False)[0]
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

def normalize(data):
    max_val=data.max()
    min_val=data.min()
    return (data-min_val)/(max_val-min_val)

def reshape_data_1d(x):
    h=x.shape[0]
    if h>256*256-1:
        return x[:256*256-1]
    else:
        h_need=256*256-1-h
        if h_need%2==0:
            result=np.pad(x,(h_need//2,h_need//2),'constant',constant_values=0)
        else:
            result=np.pad(x,(h_need//2+1,h_need//2),'constant',constant_values=0)
        return result


model=UNet2DModel(sample_size=[128,256],
                    in_channels=1,
                    out_channels=1,
                    block_out_channels=[64,64,128,128,256,256],
                    down_block_types=["DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"],
                    up_block_types=["UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"])
model=torch.load('/data/tianqi/denoising_model/diffusion/diffusion_model_mel.pt')
model=model.to(device)
diffusion=DenoiseDiffusion(eps_model=model,n_steps=200,device=device)

audio_path='/data/tianqi/denoising_model/noisy_testset_wav'
audio_list=os.listdir(audio_path)
for a in audio_list:
    audio,_=librosa.load(audio_path+'/'+str(a),mono=True,sr=16000)
    audio=reshape_data_1d(audio)
    S=librosa.feature.melspectrogram(y=audio,sr=16000,n_fft=1024,hop_length=256,n_mels=128)
    log_S=librosa.power_to_db(S, ref=np.max,top_db=80)
    x=torch.Tensor(normalize(log_S)).to(device)

    with torch.no_grad():
        x=x.view(1,1,128,256)
        for i in range(20):
            t=20-i-1
            x=diffusion.p_sample(x,x.new_full((1,),t,dtype=torch.long))
            x=torch.clamp(x,0,1)
        
    x=x.cpu().numpy().reshape(1,128,256)
    for i in x:
        log_S=i*80-80
        S=librosa.db_to_power(log_S)
        wav=librosa.feature.inverse.mel_to_audio(M=S,sr=16000,n_fft=1024,hop_length=256,n_iter=32)
        sf.write('/data/tianqi/denoising_model/diffusion_results/'+str(a),wav,16000)