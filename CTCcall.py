import tarfile
import torch
import torch.nn as nn
import numpy as np
from model import *
from build_model import *
import decoder
import encoder
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



#load training data and put into tensor
#source = torch.from_numpy(np.load("data/Ecoli/CP045741/test/Chunk.npy")).to(device,dtype=torch.float)
source = torch.from_numpy(np.load("data/Ecoli/CP045741/test/Chunk.npy")).to(device)

output_length = 900
input_length = len(source[0])
model_length = input_length//8
input_chennels = 256
SALayers = 5
head_num = 8
class_num = 5
conv_kernel = 11
#encoder = encoder.Encoder(model_length,input_chennels,SALayers,head_num,conv_kernel)
encoder = encoder.RGRGR()
decoder = decoder.CTCDecoder(model_length,output_length,input_chennels,class_num)

model = build_model(encoder,decoder,device)
#load model
path = input("model path :")
model.load_state_dict(torch.load(path))
model.eval()

classes = [
    "A",
    "C",
    "G",
    "T"
]
Time_start = time.time()
#convert log probability into argmax and remove repeated symbols
arry = []
with torch.no_grad():
    for i in range(len(source)):
        output = model(source[i:i+1]).log_softmax(1)
        probability = torch.exp(output)
        pred_raw = torch.argmax(output,dim=1).squeeze(dim=0).detach().cpu().numpy()
        del_repeat = pred_raw
        for j in range(len(del_repeat)-1):
            if del_repeat[j] == del_repeat[j+1]:
                del_repeat[j] = 0
        arry.append(pred_raw)

#covert 0,1,2,3,4 into A,T,C,G as 0 = blank, 1 = A, 2 = T, 3 = C, 4 = G
seq=[]
for i in range(len(arry)):
    seq.append(">"+str(i)+"\n")
    for j in range(len(arry[i])):
        if arry[i][j]>0:
            seq.append(classes[arry[i][j]-1])
    seq.append("\n")
#save as fasta file
def seq2fasta(seq):
    with open('call.fasta', 'w') as f:
        f.write("".join(seq))
seq2fasta(seq)
Time_end = time.time()
Time_spend = Time_end-Time_start
print(Time_spend)
'''
minimap2 -c --eqx ref.fasta call.fasta > alignment.paf
minimap2 -c --eqx train.fasta call.fasta > alignment.paf
minimap2 -c --eqx data/Ecoli/CP026085.1.fasta call.fasta > alignment.paf
minimap2 -c --eqx data/Ecoli/Tools_polished/consensus_homopolished.fasta call.fasta > alignment.paf
minimap2 -c --eqx data/Ecoli/CP045741/CP045741.1.fasta call.fasta > alignment.paf
'''
