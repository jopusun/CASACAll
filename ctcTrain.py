from logging import root

from model import *
from build_model import *
import decoder
import encoder
import data_path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



torch_dataset,val_dataset = data_path.dataset("dna_r9.4.1")
loader =data.DataLoader(
    dataset=torch_dataset,
    batch_size=45,
    shuffle=True,
    num_workers=4,
    drop_last=True
)
val_loader =data.DataLoader(
    dataset=val_dataset,
    batch_size=45,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

#set hyperparameters
output_length = 1200
input_length = 3600
model_length = input_length//8
input_chennels = 256
SALayers = 5
head_num = 8
class_num = 5
conv_kernel = 59

encoder = encoder.Encoder(model_length,input_chennels,SALayers,head_num,conv_kernel)
#encoder = encoder.RGRGR()
decoder = decoder.CTCDecoder(model_length,output_length,input_chennels,class_num)
model = build_model(encoder,decoder,device)

#load model
#model.load_state_dict(torch.load('exp_data/20220910_InStackPos/kernel_33/ecoli/saved_model.pth'))

 # 建構 optimizer
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-5,max_lr=5e-4,cycle_momentum = False)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-5,steps_per_epoch=24429,epochs=30,pct_start=0.3,div_factor=1e3,final_div_factor=1e4)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-1,steps_per_epoch=4568,epochs=30,pct_start=0.3,div_factor=1e3,final_div_factor=1e4)
loss_fn = nn.CTCLoss()

losses,val_losses = [],[]
'''
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
'''

input_lengths = torch.full(size=(loader.batch_size,), fill_value = output_length, dtype=torch.long).to(device)

epochs = 50
min_valid_loss = np.inf
for epoch in range(epochs):
    current_loss = 0
    for j, (x, y, target_lengths) in enumerate(loader):
        x, y, target_lengths  = x.to(device), y.to(device), target_lengths
        # Compute prediction error    
        pred = model(x).log_softmax(1).requires_grad_()
        pred = pred.permute(2,0,1)
        loss = loss_fn(pred, y.long(),input_lengths,target_lengths)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss += loss.item()
    valid_loss = 0.0

    model.eval()
    with torch.no_grad():
        for j, (x, y, valid_lengths) in enumerate(val_loader):
            x, y, valid_lengths = x.to(device), y.to(device), valid_lengths.to(device)
            # Compute prediction error
            pred = model(x).log_softmax(1)
            pred = pred.permute(2,0,1)
            loss = loss_fn(pred, y.long(),input_lengths,valid_lengths)

            valid_loss += loss.item()
    model.train()

    current_loss = current_loss / len(loader)
    valid_loss = valid_loss / len(val_loader)

    print(f"loss:     {current_loss:>7f} ","progress",epoch)
    print(f"val_loss: {valid_loss:>7f} ","progress",epoch)
    losses.append(current_loss)
    val_losses.append(valid_loss)

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})  Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')
#np.save("losses.npz",losses = losses,val_losses = val_losses)

import matplotlib.pyplot as plt
plt.title('Training and Validation loss')
plt.plot(range(1,epochs+1),losses,color = 'r', label="train")
plt.plot(range(1,epochs+1),val_losses,color = 'b', label="val")

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


