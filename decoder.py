import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
#out put of encoder(batch_size, channel_num, model_length)


class LSTMDecoder(nn.Module):
    def __init__(self, input_chennels, model_length,output_length,class_num):
        super().__init__()
        self.fc = nn.Linear(model_length,output_length)
        self.lstm = nn.LSTM(input_chennels,class_num)
        self.relu = nn.LeakyReLU()
    def forward(self,x: torch.Tensor):
        x = self.fc(x)
        x = x.permute(0,2,1)
        #(batch_size, output_length, channel_num)
        x,_ = self.lstm(x)
        x = x.transpose(1,2)

        return x

class CTCDecoder(nn.Module):
    def __init__(self, model_length,output_length,input_chennels,class_num) -> None:
        super().__init__()
        self.fc1 = nn.Linear(model_length,output_length)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(input_chennels,class_num)
    def forward(self,x: torch.Tensor):
        x = self.fc1(x)
        #(batch_size, channel_num, output_length)
        x = x.transpose(1,2)
        #(batch_size, output_length, channel_num)
        x = self.fc2(x)
        x = x.transpose(1,2)
        # a log softmax is added in traning step
        return x