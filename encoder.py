from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn.modules.rnn import LSTM

class Encoder(nn.Module):
    def __init__(self, model_length, input_chennels, SALayers,head_num,conv_kernel = 3):
        super(Encoder, self).__init__()
        self.SALayers = SALayers
        self.Subsample = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 8,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 8,out_channels = 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 64,out_channels = input_chennels,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(input_chennels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(p = 0.1)
        )

        self.MHSA = MHSA(input_chennels,head_num,0.1)
        self.pos_encoder = PositionalEncoding(input_chennels,model_length)
        self.Feedforward = nn.Sequential(
                nn.Linear(input_chennels,4*input_chennels),
                nn.SiLU(inplace=True),
                nn.Dropout(p = 0.1),
                nn.Linear(4*input_chennels,input_chennels), 
                nn.Dropout(p = 0.1)
        )
        self.ConvModule = nn.Sequential(
            #expand length for glu
            nn.LayerNorm(model_length),
            nn.Conv1d(input_chennels, input_chennels*2, kernel_size = 1),
            nn.GLU(dim=1),
            nn.Conv1d(input_chennels, input_chennels, kernel_size = conv_kernel, groups = input_chennels, padding = int((conv_kernel-1)/2)),
            nn.BatchNorm1d(input_chennels),
            nn.SiLU(inplace=True),
            nn.Conv1d(input_chennels, input_chennels, kernel_size = 1),
            nn.Dropout(p = 0.1)
            
        )
        self.LN = nn.LayerNorm(model_length)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x :torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(dim = 1)
        x = self.Subsample(x)
        #shape of (batch ,channel = 64 , model_length*2)
        for i in range(self.SALayers):
            x = x + 0.5*self.Feedforward(self.LN(x).transpose(1,2)).transpose(1,2)
            '''
            y = self.LN(x)
            y = y.transpose(1,2)
            y = self.Feedforward(y)
            y = y.transpose(1,2)
            x = x + 0.5*(y)'''
   
            y = self.LN(x)
            y = self.pos_encoder(y)
            y,_ = self.MHSA(y)
            #y = self.relu(y)
            x = x + y
            x = x + self.ConvModule(x)

            x = x + 0.5*self.Feedforward(self.LN(x).transpose(1,2)).transpose(1,2)
            x = self.LN(x)
        return x

class RGRGR(nn.Module):
    def __init__(self):
        super(RGRGR,self).__init__()
        self.Subsample = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 8,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 8,out_channels = 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(p = 0.1),

        )
        self.lstm = nn.LSTM(input_size=128,hidden_size=128,num_layers=2,batch_first=True,bidirectional=True)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(dim = 1)
        x = self.Subsample(x)
        x = x.transpose(1,2)
        x,(_,_) = self.lstm(x)
        x = x.transpose(1,2)
        return x
            

class SACall(nn.Module):
    def __init__(self):
        super(SACall,self).__init__()
        self.Subsample = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 8,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 8,out_channels = 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 64,out_channels = 256,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(p = 0.1)
        )
        self.MHSA = MHSA(256,8,0.1)
        self.pos_encoder = PositionalEncoding(256,450)
        self.relu = nn.LeakyReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256,256)
        self.dropout = nn.Dropout(0.3)
        self.layernorm = nn.LayerNorm(450)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(dim=1)
        x = self.Subsample(x)
        x = self.pos_encoder(x)
        for i in range(5):
            y,w = self.MHSA(x)
            y = self.relu(y)
            x = x + y
            x = self.batchnorm(x)
            x = x.transpose(1,2)
            x = self.dropout(x)
            y = self.fc(x)
            y = self.fc(y)
            x = x + y
            x = x.transpose(1,2)
            x = self.layernorm(x)
            x = self.relu(x)

        return x
class Encoder_crossFFN(nn.Module):
    def __init__(self, model_length, input_chennels, SALayers,head_num,conv_kernel=3) -> None:
        super(Encoder_crossFFN,self).__init__()
        self.Subsample = Encoder(model_length, input_chennels, SALayers,head_num,conv_kernel=3).Subsample
        self.crossFFN = crossFFN(input_chennels,model_length)
        self.BN = nn.BatchNorm1d(input_chennels)
        #todo:activation,norm,residual
    def forward(self,x:torch.Tensor):
        x = x.unsqueeze(dim = 1)
        x = self.Subsample(x)
        for i in range(15):
            x = x + self.crossFFN(x)
            x = self.BN(x)
        return x




class Encoder_kernel6(nn.Module):
    def __init__(self, model_length, input_chennels, SALayers,head_num,conv_kernel=3):
        super(Encoder_kernel6, self).__init__()
        self.SALayers = SALayers
        self.Subsample = nn.Sequential(
            nn.Conv1d(in_channels = 1,out_channels = 8,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 8,out_channels = 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Conv1d(in_channels = 64,out_channels = input_chennels,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.BatchNorm1d(input_chennels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(p = 0.1)
        )
        self.MHSA = MHSA(input_chennels,head_num,0.1)
        self.pos_encoder = PositionalEncoding(input_chennels,model_length)
        self.Feedforward = nn.Sequential(
                #nn.LayerNorm(model_length),
                nn.Linear(input_chennels,input_chennels),
                nn.SiLU(inplace=True),
                nn.Dropout(p = 0.1),
                nn.Linear(input_chennels,input_chennels), 
                nn.Dropout(p = 0.1)
        )
        self.ConvModule = nn.Sequential(
            #expand length for gluf.Subsample(x)
            nn.LayerNorm(model_length),
            nn.Conv1d(input_chennels, input_chennels*2, kernel_size = 1),
            nn.GLU(dim=1),
            nn.Conv1d(input_chennels, input_chennels, kernel_size = conv_kernel, groups = input_chennels, padding = int((conv_kernel-1)/2)),
            nn.BatchNorm1d(input_chennels),
            nn.SiLU(inplace=True),
            nn.Conv1d(input_chennels, input_chennels, kernel_size = 1),
            nn.Dropout(p = 0.1)
        )
        self.LN = nn.LayerNorm(model_length)
        self.fc1 = nn.Linear(64,input_chennels)
        self.fc2 = nn.Linear(model_length*2, model_length)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x :torch.Tensor)->torch.Tensor:
        x = x.unsqueeze(dim = 1)

        x = self.Subsample(x)
        x = self.pos_encoder(x)
        for i in range(self.SALayers):
            x = self.LN(x)
            x = x.transpose(1,2)
            x = x + 0.5*(self.Feedforward(x))
            x = x.transpose(1,2)
            y,_ = self.MHSA(x)
            y = self.relu(y)
            x = x + y
            x = x + self.ConvModule(x)
            x = self.LN(x)
            x = x.transpose(1,2)
            x = x + 0.5*(self.Feedforward(x))
            x = x.transpose(1,2)
            x = self.LN(x)
        return x


class crossFFN(nn.Module):
    def __init__(self,dim,length) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(dim,length)
        self.lengthFFN = nn.Linear(length,length)
        self.dimFFN = nn.Linear(dim,dim)
        self.SiLU = nn.SiLU()
        self.Dropout = nn.Dropout(0.3)
    def forward(self,x :torch.Tensor):
        x = self.pos_encoder(x)
        x = self.lengthFFN(x)
        x = self.SiLU(x)
        x = x.transpose(1,2)
        x = self.dimFFN(x)
        x = self.Dropout(x)
        x = x.transpose(1,2)
        return x
        

class MHSA(nn.Module):
    def __init__(self,embed_dim , num_heads, dropout):
        super().__init__()
        self.multihead_attn  = nn.MultiheadAttention(embed_dim,num_heads,dropout,batch_first=True)
    def forward(self,input:torch.Tensor):
        input = input.permute(0,2,1)
        attn_output, attn_output_weights = self.multihead_attn(input, input, input)
        attn_output = attn_output.permute(0,2,1)
        return attn_output,attn_output_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int,  max_len: int ,dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(2,0,1)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,2,0)
        return self.dropout(x)
        