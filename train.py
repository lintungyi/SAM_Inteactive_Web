import pathlib
import numpy as np 

import os
os.environ["NEURITE_BACKEND"] = "pytorch"
import neurite as ne
import warnings
import torch
#import monai
from tqdm import tqdm
import cv2 as cv
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
#import src.utils as utils
#from src.dataloader import DatasetSegmentation, collate_fn
#from src.processor import Samprocessor
from segment_anything import build_sam_vit_b, SamPredictor
#from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import losses
from scribbleprompt import ScribblePromptSAM
from scribbleprompt import LineScribble
import gc
"""
scribble focused training
unfinished:batch size can't greater than 1。memory allocation need more optimize

"""
class MedScribble(Dataset):
    def __init__(self, path):
        
        if isinstance(path, str):
            path = pathlib.Path(path)
            
        self.path = path
        self.folders = sorted([x.parent for x in self.path.glob("*/*/*/*/*/scribble_1.npy")])
        assert len(self.folders)==5
    
    def __len__(self):
        return 3 * len(self.folders)
        
    def __getitem__(self, idx):
        
        folder_idx = idx // 3
        annotator_idx = idx % 3
        
        folder = self.folders[folder_idx]
        
        if (folder / "img.npy").exists():
            img = np.load(folder / "img.npy")
        else:
            warnings.warn(f"img.npy missing for {folder}. Please download data following instructions in README")
            img = np.zeros((256,256))
        
        if (folder / "seg.npy").exists():
            seg = np.load(folder / "seg.npy")
        else:
            warnings.warn(f"seg.npy missing for {folder}. Please download data following instructions in README")
            seg = np.zeros((256,256))
        
        manual_scribble = np.load(folder / f"scribble_{annotator_idx}.npy")
        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)
        manual_scribble = torch.from_numpy(manual_scribble)
        img = img.unsqueeze(0)
        seg = seg.unsqueeze(0)
        return img, seg, manual_scribble
            
    """
class trainDataset(Dataset):
    def __init__(self,path,max_pixel):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path
        self.item=[]
        folders = sorted([x.parent for x in self.path.glob("*/*/*/*/*/mask_filname.txt")])
        for folder in folders:
            f=open(folder/"mask_filename.txt",'r')
            f=f.read()
            name_list=eval(f)
            mask2scribble=LineScribble(max_pixels=max_pixel)
            for n in name_list:
                if (folder / n).exists():
                    img = cv.imread(folder / n,cv.IMREAD_GRAYSCALE)
                else:
                    warnings.warn(f"img missing for {n}.")
                    img = np.zeros((256,256))
                if (folder / n).exists():
                    seg = cv.imread(folder / n,cv.IMREAD_GRAYSCALE)
                else:
                    warnings.warn(f"mask missing for {n}.")
                    seg = np.zeros((256,256))
                img = img.unsqueeze(0)#1*H*W
                seg = seg.unsqueeze(0)#1*H*W
                seg_t= seg.unsqueeze(0)#1*1*H*W
                img=torch.from_numpy(img)
                seg=torch.from_numpy(img)
                img=torch.where(img==64,torch.tensor(1.),torch.tensor(0.))
                seg=torch.where(seg==64,torch.tensor(1.),torch.tensor(0.))
                #no neg scribble now
                pos_scr=mask2scribble.batch_scribble(mask=seg_t)#1*1*H*W
                scribble=torch.stack([pos_scr[0][0],torch.zeros(pos_scr[0][0].shape())],dim=0)#2*H*W
                self.item.append((img,seg,scribble))
        #assert len(self.folders)==5

    def __getitem__(self, index):
        img,seg,scribble=self.item[index]              
        return img, seg, scribble

    def __len__(self):
        return len(self.item)
 """        
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

SAM = ScribblePromptSAM()
model=SAM.model
max_pixel=config_file["TRAIN"]["MAX_PIXEL"]
train_dataset=MedScribble(config_file["DATASET"]["TRAIN_PATH"])
dataloader=DataLoader(dataset=train_dataset,batch_size=config_file["TRAIN"]["BATCH_SIZE"],shuffle=True)#, collate_fn=collate_fn
print("data loaded")
# Initialize optimize and Loss
optimizer = Adam([{'params': model.image_encoder.parameters()},{'params': model.mask_decoder.parameters()}], lr=1e-4, weight_decay=0)
#optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = losses.SAM_loss #monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []
print("start training")

num_step=config_file["TRAIN"]["NUM_STEP"]
for epoch in range(num_epochs):
    epoch_losses = []
    gc.collect()
    #nparray to tensor
    for img,seg,initial_scribble in tqdm(dataloader):
            """
            mask, img_features, low_res_logits = sp_sam.predict(
            image,        # (B, 1, H, W) 
            scribbles,    # (B, 2, H, W)
            mask_input,   # (B, 1, 256, 256)
            ) # -> (B, 1, H, W), (B, 16, 256, 256), (B, 1, 256, 256)
            """
            gc.collect()
            epoch_losses_t = []
            global low_res_logits_d
            threshold_mask=torch.nn.Threshold(0, 0)
            mask2scribble=LineScribble(max_pixels=max_pixel)
            
            for o in range(num_step):                
                if(o==0):
                    optimizer.zero_grad()
                    mask, img_features, low_res_logits = SAM.predict(img=img.to(device),scribbles=initial_scribble.to(device))
                else:
                    error_region=seg.to(device)-mask
                    false_pos=threshold_mask(error_region*-1)
                    false_neg=threshold_mask(error_region)
                    scribbles_t=torch.cat([mask2scribble.batch_scribble(mask=false_neg),mask2scribble.batch_scribble(mask=false_pos)],axis=1)
                    del error_region,false_pos,false_neg
                    optimizer.zero_grad()
                    mask, img_features, low_res_logits = SAM.predict(img=img.to(device),scribbles=scribbles_t.to(device),mask_input=low_res_logits_d.to(device))
                loss = seg_loss(low_res_logits, seg.float().to(device))
                gc.collect()
                loss.backward()
                # optimize
                low_res_logits_d=low_res_logits.detach()
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses_t.append(loss.item())
                del loss, low_res_logits
                gc.collect()
            #print(mean(epoch_losses_t))
            epoch_losses.append(mean(epoch_losses_t))
            del epoch_losses_t
            gc.collect()
    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
SAM.sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
torch.save(model.state_dict(),"./scribble_SAM_Lora.pth")
