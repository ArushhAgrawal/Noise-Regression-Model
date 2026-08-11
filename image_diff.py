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

class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_vector= nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
    def forward(self, time_stamp):
        time_vector= time_stamp.unsqueeze(-1).float()
        return self.time_vector(time_vector)
    

class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_features)
        self.relu = nn.ReLU()
        self.time_proj = nn.Linear(512, out_features)  # Projects 512 global vector -> block channel count
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_features)
    def forward(self,x, time_stamp):
        x = self.conv1(x)
        x = self.norm1(x)
        t = self.time_proj(time_stamp)
        t = t.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch, channels, 1, 1)
        x = x + t  # Add time embedding to feature map
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x

class NOISE(nn.Module):
    def __init__(self, time_stamp, in_channels=3,  out_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.up= nn.ModuleList()
        self.down= nn.ModuleList() 
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
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
    def forward(self, x, time_stamp):
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
        return self.final_conv(x)


#loading image dataset
class ImageDataset(Dataset):
    def __init__ (self, image_dir, resize_shape=(160,160), max_image=100, sigma=20, timestamp=1):
        self.image_dir= image_dir
        self.reshape= resize_shape
        all_images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])#this sorts the list so computer doest do random input and messes the data
        self.images= all_images[:max_image]
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        start=1
        diffrence= 1
        image_name=self.images[index]
        image_path= os.path.join(self.image_dir, image_name)#creates a direct path to the image
        image= Image.open(image_path).convert("RGB")
        #resizing image
        image= TF.resize(image,self.reshape)
        clear_tensor= TF.to_tensor(image)
        timestamp=torch.randint(1,1000,())
        start, sigma= torch.round((start+(timestamp-1)*diffrence),2)#to get sigma and start value in 2 decmial points
        beta = (sigma/255.0)
        noise= torch.randn_like(clear_tensor) * beta
        noisy_tensor = torch.clamp(clear_tensor + noise, 0.0, 1.0)
        time_stamp=timestamp
        return noisy_tensor,time_stamp, noise
    
#loading data
if __name__=="__main__":#runs from here after each time we run the code not from the import
    full_data= ImageDataset(image_dir="data/clear_images/", resize_shape=(160,160), max_image=220)
    train_size= int(0.8*(len(full_data)))
    val_size= len(full_data)-train_size
    train_dataset, val_dataset= random_split(full_data, [train_size, val_size])
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=4, 
        persistent_workers=True,       
        shuffle=True)
    val_loader= DataLoader(
        dataset=val_dataset,
        num_workers=2,
        persistent_workers=True,
        batch_size=4,
        shuffle=False)
# x, y = next(iter(val_loader))
# print(y[0].shape)
    
# # setting up device diagnostic code
    device= "mps" if torch.mps.is_available() else "cpu"
#defining model
    model= NOISE(in_channels=3, out_channels=3).to(device)
#defining loss and optimizer
    optimizer= torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn= nn.L1Loss()
#making checkpoints
    start_epoch=0
    epochs=120
    run_loss=0
    checkpoint_dir= "checkpoint3"
    os.makedirs(checkpoint_dir,exist_ok=True)
    latest_chkp3=os.path.join(checkpoint_dir, "latest_3.pth")
    if os.path.exists(latest_chkp3):
        checkpoint= torch.load(latest_chkp3, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch= checkpoint["epoch"]+1
        average_loss= checkpoint["loss"]
        print("resumed from checkpoint")

#training/eval
    start= time.time()
    net_loss_train= []
    net_loss_test= []
    epoch_bar= tqdm.tqdm(range(start_epoch, epochs), desc= "Training Model")
    for epoch in epoch_bar:
        model.train()
        run_loss=0
        for batch, (x,y) in enumerate(train_loader):
            y_train= y.to(device, non_blocking=True)
            for index in range(3):
                x_train= x[index].to(device, non_blocking=True)
                y_train_logits= model(x_train)
                loss_train=loss_fn(y_train_logits,y_train)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                net_loss_train.append(loss_train.item())
                run_loss+= loss_train.item()
        avg_loss=run_loss/(len(train_loader)*3)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }, latest_chkp3)
    end=time.time()
# print("\ntime taken", end-start,"\nloss train",avg_loss ,"\nloss test", loss_test, "\nloss every batch train", net_loss_train, "\nloss every batch test",net_loss_test) 

#visualisation
    model.eval()


    x, y = next(iter(val_loader))
    noisy_batch = x[0].to(device) 
    clean_batch = y[0].to(device)

    with torch.inference_mode():
        preds = model(noisy_batch).cpu()

# Pick the first image from the batch
    idx = 0
    noisy_img = x[0][idx].permute(1, 2, 0).numpy()
    clean_img = y[0].permute(1, 2, 0).numpy()
    pred_img  = preds[idx].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy_img); axes[0].set_title("Noisy (input)"); axes[0].axis("off")
    axes[1].imshow(pred_img);  axes[1].set_title("Model output");  axes[1].axis("off")
    axes[2].imshow(clean_img); axes[2].set_title("Clean (target)"); axes[2].axis("off")
    plt.show()