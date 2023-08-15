import tarfile
import torch
import torch.nn as nn
import numpy as np
from model import *
from build_model import *
import decoder
import encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#load training data and put into tensor
#source = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/validation/chunks.npy")).to(device)
source = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/validation/chunks.npy")).to(device)
target = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/validation/references.npy")).to(device)
'''
torch_dataset = data.TensorDataset(source,target)


loader =data.DataLoader(
    dataset=torch_dataset,
    batch_size=60,
    shuffle=True,
    num_workers=2
)

'''
output_length = 486
input_length = len(source[0])
model_length = input_length//8
input_chennels = 256
head_num = 8
SALayers = 5
class_num = 5

encoder = encoder.Encoder(model_length,input_chennels,SALayers,head_num)
decoder = decoder.CTCDecoder(model_length,output_length,input_chennels,class_num)

model = build_model(encoder,decoder,device)
#load model
path = input("model path :")
model.load_state_dict(torch.load(path))
model.eval()

classes = [
    "A",
    "T",
    "C",
    "G"
]
arry = []
with torch.no_grad():
    for i in range(len(source)):
        pred = model(source[i:i+1])
        pred = pred.squeeze(0)
        arry.append(torch.argmax(pred,dim=0).detach().cpu().numpy())
    print(pred.shape)
    print(np.shape(arry))

seq=[]
for i in range(len(arry)):
    seq.append("> "+str(i)+"\n")
    for j in range(len(arry[i])):
        if arry[i][j]>0:
            seq.append(classes[arry[i][j]-1])
    seq.append("\n")
def seq2fasta(seq):
    with open('call.fasta', 'w') as f:
        f.write("".join(seq))
seq2fasta(seq)


'''
minimap2 -c call.fasta ref.fasta > alignment.paf

'''
