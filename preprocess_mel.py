import numpy as np
import os
import pyworld as pw
import librosa
import torch.nn as nn
import random
import torch
import torchaudio as T
import torchaudio.transforms as TT
from tqdm import tqdm
device='cuda'

def audio_to_image(audio):
    S=librosa.feature.melspectrogram(y=audio,sr=16000,n_fft=1024,hop_length=256,n_mels=128)
    log_S=librosa.power_to_db(S, ref=np.max,top_db=80)
    return log_S

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

def create_train_data_1_folder(file_path):
    print('Processing data')
    data_train=[]
    wav_list=os.listdir(file_path)
    with tqdm(total=len(wav_list)) as pbar:
        for wav in wav_list:
            pbar.update(1)
            path=file_path+'/'+wav
            audio,_=librosa.load(path,mono=True,sr=16000)
            audio=reshape_data_1d(audio)
            mel=audio_to_image(audio)
            mel=normalize(mel)
            data_train.append(torch.Tensor(mel))
    return data_train

def create_train_data_2_folders(file_path):
    print('Processing data')
    data_train=[]
    speaker_list=os.listdir(file_path)
    for speaker in speaker_list:
        if speaker[0]!='R' and speaker[0]!='.':
          corpus_list=os.listdir(file_path+'/'+speaker)
          for corpus in corpus_list:
            path=file_path+'/'+speaker+'/'+corpus
            audio,_=librosa.load(path,mono=True,sr=16000)
            audio=reshape_data_1d(audio)
            mel=audio_to_image(audio)
            mel=normalize(mel)
            data_train.append(torch.Tensor(mel))
    return data_train

def normalize(data):
    max_val=data.max()
    min_val=data.min()
    return (data-min_val)/(max_val-min_val)

filepath='/data/tianqi/model_part_3/dataset'
#filepath='/data/tianqi/model_part_3/vcc2016/wav/Testing Set'

data=create_train_data_1_folder(filepath)
#data=create_train_data_2_folders(filepath)
random.shuffle(data)
x=torch.stack(data)
x=x.unsqueeze(1)
print(x.shape)
torch.save(x,'clean_data_part_2.pt')