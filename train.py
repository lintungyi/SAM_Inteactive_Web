import pathlib
import numpy as np 

import os
os.environ["NEURITE_BACKEND"] = "pytorch"
import neurite as ne
import warnings
import torch
#import monai
from tqdm import tqdm
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

"""
unfinished

"""

class MedScribble:
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
        
        return img, seg, manual_scribble
        
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
#train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
# Load SAM model
#sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
#sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
#model = sam_lora.sam

SAM = ScribblePromptSAM()
model=SAM.model

# Process the dataset
#processor = Samprocessor(model)
#train_ds = DatasetSegmentation(config_file, processor, mode="train")

#batch_size=config_file["TRAIN"]["BATCH_SIZE"]


# Create a dataloader
#train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = losses.SAM_loss #monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []
x = MedScribble("./MedScribble")
print("start training")
for epoch in range(num_epochs):
    epoch_losses = []


    #nparray to tensor
    for i in tqdm(range(len(x.folders))):
            img, seg, manual_scribble = x[i]
            img = torch.from_numpy(img)
            seg = torch.from_numpy(seg)
            manual_scribble = torch.from_numpy(manual_scribble)
            #img.to(device)
            #seg.to(device)
            #manual_scribble.to(device)
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            seg = seg.unsqueeze(0)
            seg = seg.unsqueeze(0)
            manual_scribble = manual_scribble.unsqueeze(0)


            """
            mask, img_features, low_res_logits = sp_sam.predict(
            image,        # (B, 1, H, W) 
            scribbles,    # (B, 2, H, W)
            mask_input,   # (B, 1, 256, 256)
            ) # -> (B, 1, H, W), (B, 16, 256, 256), (B, 1, 256, 256)
            
            """
           
            mask, img_features, low_res_logits = SAM.predict(img=img.to(device),scribbles=manual_scribble.to(device))
            loss = seg_loss(low_res_logits, seg.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

            """
            global low_res_logits_t
            for o in range(2):
                
                if(o==0):
                    mask, img_features, low_res_logits = SAM.predict(img=img.to(device),scribbles=manual_scribble.to(device))
                else:
                    mask, img_features, low_res_logits = SAM.predict(img=img.to(device),scribbles=manual_scribble.to(device),mask_input=low_res_logits_t.to(device))
                loss = seg_loss(low_res_logits, seg.float().to(device))
                optimizer.zero_grad()
                low_res_logits_t=low_res_logits.clone()
                loss.backward()
                # optimize
                optimizer.step()
                epoch_losses.append(loss.item())
            """
            
           
            
    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
SAM.sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")





