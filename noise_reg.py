import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import time
from PIL import Image
import tqdm
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import os
device= "mps" if torch.mps.is_available() else "cpu"
class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features ):
        super().__init__()
        self.convstack= nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU()
            # nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_features), 
            # nn.ReLU()
        )
    def forward(self,x):
        return self.convstack(x)

class NOISE(nn.Module):
    def __init__(self, in_channels=3,  out_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.up= nn.ModuleList()
        self.down= nn.ModuleList()
        self.pool= nn.AvgPool2d(kernel_size=2, stride=2)
        #down part
        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels=feature
        
        #up part
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2))#this will make the model move up
            self.up.append(DoubleConv(feature*2, feature))#this will do the conv part up 2 conv like this combined
        
        #bottom part
        self.bottom= DoubleConv(in_features=features[-1], out_features=features[-1]*2)#in feature then out feature 512*2
        self.final_conv= nn.Conv2d(features[0], out_channels,  kernel_size=1)
    def forward(self, x):
        skip=[]
        #downward pass
        for index in self.down:
            x=index(x)
            skip.append(x)#this is the part where it will skip the connection from the heightest to the lowest connection
            x=self.pool(x)#160x160-> 80x80
        x=self.bottom(x)
        skip= skip[::-1]#reversing the list 
        for index in range (0,len(self.up), 2):
            x=self.up[index](x)#up sampling
            skip_tensor= skip[index//2]#divides and then rounds to nearest whole number
            if skip_tensor.shape == x.shape:
                concat_skip= torch.cat((skip_tensor, x),dim=1)    
            else:
                l= x.shape[2]
                w=x.shape[3]
                concat_skip= torch.cat((skip_tensor[:,:, :l, :w], x),dim=1)
            x= self.up[index+1](concat_skip)
        return torch.sigmoid(self.final_conv(x))


#loading image dataset
class ImageDataset(Dataset):
    def __init__ (self, image_dir, noise_range=(10,50), resize_shape=(160,160), max_image=100):
        self.image_dir= image_dir
        self.noise_ranges= noise_range
        self.reshape= resize_shape
        all_images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])#this sorts the list so computer doest do random input and messes the data
        self.images= all_images[:max_image]
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image_name=self.images[index]
        image_path= os.path.join(self.image_dir, image_name)#creates a direct path to the image
        image= Image.open(image_path).convert("RGB")
        #resizing image
        image= TF.resize(image,self.reshape)
        clear_tensor= TF.to_tensor(image)
        sigma = np.random.uniform(*self.noise_ranges) / 255.0 #to convert all the pixel data values we use divide by 255.0, * is the unpacking opertor random.uniform take input as low, high and if i will not use star it will keep low=(10,50) insted of low=10
        noise = torch.randn_like(clear_tensor) * sigma
        noisy_tensor = torch.clamp(clear_tensor + noise, 0.0, 1.0)
        return noisy_tensor, clear_tensor
#loading data
full_data= ImageDataset(image_dir="data/clear_images/", resize_shape=(160,160), max_image=120)
train_size= int(0.8*(len(full_data)))
val_size= len(full_data)-train_size
train_dataset, val_dataset= random_split(full_data, [train_size, val_size])
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=4,        
    shuffle=True)
val_loader= DataLoader(
    dataset=val_dataset,
    batch_size=4,
    shuffle=False)

#setting up device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"
#defining model
model= NOISE(in_channels=3, out_channels=3).to(device)
#defining loss and optimizer
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn= nn.L1Loss()
#making checkpoints
checkpoint_dir= "checkpoint"
os.makedirs(checkpoint_dir,exist_ok=True)
latest_chkp=os.path.join(checkpoint_dir, "latest.pth")
if os.path.exists(latest_chkp):
    checkpoint= torch.load(latest_chkp, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch= checkpoint["epoch"]+1
    print("resumed from checkpoint")

#training/eval
run_loss=0
start_epoch=0
epochs=5
start= time.time()
net_loss_train= []
net_loss_test= []
epoch_bar= tqdm.tqdm(range(start_epoch, epochs), desc= "Training Model")
for epoch in epoch_bar:
    model.train()
    for batch, (x,y) in enumerate(train_loader):
        x.to(device), y.to(device)
        y_train_logits= model(x.to(device))
        loss_train=loss_fn(y_train_logits,y.to(device))
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        net_loss_train.append(loss_train.item())
        run_loss+= loss_train.item()
    avg_loss=run_loss/len(train_loader)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer.state": optimizer.state_dict(),
        "loss": avg_loss,
    }, latest_chkp)

    model.eval()
    with torch.inference_mode():
        for batch, (x,y) in enumerate(val_loader):
            y_logits_test= model(x.to(device))
            loss_test= loss_fn(y_logits_test, y.to(device)) 
            net_loss_test.append(loss_test.item())#why item is imp since if we dont do item it would be like this loss value, mps, gradfunc all this if we do item its just the loss value
end=time.time()
print("\ntime taken", end-start,"\nloss train", loss_train,"\nloss test", loss_test, "\nloss every batch train", net_loss_train, "\nloss every batch test",net_loss_test) 
#visualisation
model.eval()

with torch.inference_mode():
    x, y = next(iter(val_loader))          # x =noisy,y = clean
    preds = model(x.to(device)).cpu()      # model's denoised output

# pick first image in the batch
idx = 0
noisy_img = x[idx].permute(1, 2, 0).numpy()      #(C,H,W) -> (H,W,C) for plotting
clean_img = y[idx].permute(1, 2, 0).numpy()
pred_img  = preds[idx].permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(noisy_img); axes[0].set_title("Noisy (input)"); axes[0].axis("off")
axes[1].imshow(pred_img);  axes[1].set_title("Model output");  axes[1].axis("off")
axes[2].imshow(clean_img); axes[2].set_title("Clean (target)"); axes[2].axis("off")
plt.show()


