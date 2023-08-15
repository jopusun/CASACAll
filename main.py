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
input = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/chunks.npy"))
target = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/references.npy"))
target_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/reference_lengths.npy")))
valid_lengths = torch.from_numpy(np.int16(np.load("data/dna_r9.4.1/validation/reference_lengths.npy")))

torch_dataset = data.TensorDataset(input,target,target_lengths)

loader =data.DataLoader(
    dataset=torch_dataset,
    batch_size=50,
    shuffle=True,
    num_workers=4
)

val_input = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/validation/chunks.npy"))
val_target = torch.from_numpy(np.load("/home/jopu/Desktop/TCC/data/dna_r9.4.1/validation/references.npy"))
val_dataset = data.TensorDataset(val_input,val_target,valid_lengths)


val_loader =data.DataLoader(
    dataset=val_dataset,
    batch_size=50,
    shuffle=True,
    num_workers=4
)



#train_features, train_labels = next(iter(loader))

output_length = 486
input_length = len(input[0])
model_length = input_length//8
input_chennels = 256
SALayers = 5
head_num = 8
class_num = 5

encoder = encoder.Encoder(model_length,input_chennels,SALayers,head_num)
decoder = decoder.CTCDecoder(model_length,output_length,input_chennels,class_num)

model = build_model(encoder,decoder,device)
# 建構 optimizer

######
#model.load_state_dict(torch.load("saved_model.pth"))
####
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
#0 = target padding should
loss_fn = nn.CrossEntropyLoss()
optimizer.state
losses,val_losses = [],[]
'''
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
'''

epochs = 5
min_valid_loss = np.inf
for epoch in range(epochs):
    current_loss = 0
    for j, (x, y, target_lengths) in enumerate(loader):
        x, y  = x.to(device), y.to(device)
        # Compute prediction error
        pred = model(x)

        #(B,C,L)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if j % 100 == 0:
        current_loss += loss.item()
    

    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for j, (x, y, valid_lengths) in enumerate(val_loader):
            x, y  = x.to(device), y.to(device)
            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y.long())

            valid_loss += loss.item()
        model.train()

    current_loss = current_loss / len(loader)
    valid_loss = valid_loss / len(val_loader)
    
    print(f"loss    : {current_loss:>7f} ","progress",epoch)
    print(f"val_loss: {valid_loss:>7f} ","progress",epoch)
    losses.append(current_loss)
    val_losses.append(valid_loss)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})  Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')
import matplotlib.pyplot as plt

plt.plot(range(1,epochs+1),losses,color = 'r', label="train")
plt.plot(range(1,epochs+1),val_losses,color = 'b', label="val")
plt.show()
'''
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
'''
