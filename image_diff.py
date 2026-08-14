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
import math

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
        x = x + t  # Add time embedding to feature map, this acts as a time based bias system, it affects the activation, working on image generation with timestep
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x

class NOISE(nn.Module):
    def __init__(self,  in_channels=3,  out_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.up= nn.ModuleList()
        self.down= nn.ModuleList() 
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)

        #time embedding 
        self.time_embedding= TimeEmbedding()

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
        time_emb= self.time_embedding(time_stamp)
        #downward pass
        for index in self.down:
            x=index(x, time_emb)
            skip.append(x)#this is the part where it will skip the connection from the heightest to the lowest connection
            x=self.pool(x)#160x160-> 80x80
        x=self.bottom(x, time_emb)
        skip= skip[::-1]#reversing the list 
        for index in range (0,len(self.up), 2):
            x=self.up[index](x)#up sampling, even conv transpose it takes only one argument 
            skip_tensor= skip[index//2]#divides and then rounds to nearest whole number
            if skip_tensor.shape == x.shape:
                concat_skip= torch.cat((skip_tensor, x),dim=1)    
            else:
                l= x.shape[2]
                w=x.shape[3]
                concat_skip= torch.cat((skip_tensor[:,:, :l, :w], x),dim=1)
            x= self.up[index+1](concat_skip, time_emb)#odd one double conv
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
        start=1/2
        diffrence= 1/2
        image_name=self.images[index]
        image_path= os.path.join(self.image_dir, image_name)#creates a direct path to the image
        image= Image.open(image_path).convert("RGB")
        #resizing image
        image= TF.resize(image,self.reshape)
        clear_tensor= TF.to_tensor(image)
        timestamp=torch.randint(1,1000,())
        angle= (timestamp.float()*math.pi)/2000
        image_weight= torch.cos(angle)
        noise_weight= torch.sin(angle)
        noise= torch.randn_like(clear_tensor)
        noisy_tensor = (clear_tensor*image_weight) + (noise*noise_weight)
        time_stamp=timestamp
        return noisy_tensor,time_stamp, noise
    
#loading data
if __name__=="__main__":#runs from here after each time we run the code not from the import
    full_data= ImageDataset(image_dir="data/clear_images/", resize_shape=(160,160), max_image=520)
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
    loss_fn= nn.MSELoss()
#making checkpoints
    start_epoch=0
    epochs=305
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
        for batch, (noisy_img, time_stamp, noise) in enumerate(train_loader):
            image_train= noisy_img.to(device, non_blocking=True)
            time_train= time_stamp.to(device, non_blocking=True)
            noise_train= noise.to(device, non_blocking=True)
            y_train_logits= model(image_train, time_train)
            loss_train=loss_fn(y_train_logits, noise_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            run_loss+= loss_train.item()
        avg_loss=run_loss/(len(train_loader))
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }, latest_chkp3)
    end=time.time()
    #print("\ntime taken", end-start,"\nloss train",avg_loss)

#generating image
    model.eval()
    generative_image = torch.randn((1, 3, 160, 160)).to(device)  # keep batch dim for the model

    with torch.inference_mode():
        for t in tqdm.tqdm(range(850, -1, -1), desc="generating image"):
            time_tensor = torch.tensor([t if t > 0 else 1], device=device)  # model was never trained on t=0
            noise_pred = model(generative_image, time_tensor)

            angle_t = (t * math.pi) / 2000   
            sin_t = math.sin(angle_t)
            cos_t = max(math.cos(angle_t), 1e-3)
            if abs(cos_t)<0.05:
                x0_pred= generative_image
        # recover the best from this step wieght
            else:
                x0_pred = (generative_image - noise_pred * sin_t) / cos_t
                x0_pred = torch.clamp(x0_pred, 0.0, 1.0)  # keep the estimate sane before reusing it

        #regeneration of noise for next step
            t_next = max(t-1,0 )
            angle_next = (t_next * math.pi) / 2000
            cos_next = math.cos(angle_next)
            sin_next = math.sin(angle_next)

            generative_image = x0_pred * cos_next + noise_pred * sin_next

        generative_image = generative_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    #visualizing the generated image
    model.eval()
    with torch.inference_mode():
        noisy_img, time_stamp, noise = next(iter(val_loader))
        img_batch = noisy_img.to(device) 
        time_batch = time_stamp.to(device)
        noise_batch = noise.to(device)

    with torch.inference_mode():
        preds = model(img_batch, time_batch).cpu()

# Pick the first image from the batch
    noisy_img = img_batch[0].cpu().permute(1, 2, 0).numpy()
    noise = noise_batch[0].cpu().permute(1, 2, 0).numpy()
    org_noise= noise*math.sin((time_batch[0].item()*math.pi)/2000)
    pred_noise  = preds[0].cpu().permute(1, 2, 0).numpy()
    rec_noise= pred_noise*math.sin((time_batch[0].item()*math.pi)/2000)
    cos= max(math.cos((time_batch[0].item()*math.pi)/2000),1e-3)
    reconstructed_img = np.clip((noisy_img - rec_noise)/cos, 0.0, 1.0)
    original_img = np.clip((noisy_img - org_noise)/cos, 0.0, 1.0)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    # fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(noisy_img); axes[0].set_title("Noisy Image(input)"); axes[0].axis("off")
    axes[1].imshow(pred_noise);  axes[1].set_title("Model output(Noise)");  axes[1].axis("off")
    axes[2].imshow(reconstructed_img); axes[2].set_title("Reconstructed (by removing noise)"); axes[2].axis("off")
    axes[3].imshow(noise); axes[3].set_title("Noise (target)"); axes[3].axis("off")
    axes[4].imshow(original_img); axes[4].set_title("Original Image (by removing noise)"); axes[4].axis("off")
    axes[5].imshow(generative_image); axes[5].set_title("Generated Image"); axes[5].axis("off")

    plt.show()