from segment_anything import SamPredictor, sam_model_registry, Sam
from segment_anything import ResizeLongestSide
import torchvision.transforms
import os
import cv2
import path_process
import generate_prompt
import dice_loss
import focal_loss
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image
from copy import deepcopy
from typing import Tuple
#training setup
my_device = "cuda"
sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters()) 
#loss_fn = torch.nn.MSELoss()
path=os.getcwd()
img_path=os.path.join(path,"Brain Tumor Segmentation Dataset")
msk_path=os.path.join(img_path,"mask","1")
img_path=os.path.join(img_path,"image","1")
img_name = os.listdir(img_path)
print(len(img_name))
#training loop
for f in img_name:
  img_filename=os.path.join(img_path,f)
  img=cv2.imread(img_filename)
  #前處理img
  input_image = ResizeLongestSide.apply_image(img)
  input_image_torch = torch.as_tensor(input_image, device=my_device)
  input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  original_size = img.shape[:2]
  input_size = tuple(input_image_torch.shape[-2:])
  input_image = Sam.preprocess(input_image_torch)

  msk_name=path_process.get_msk_path(f,msk_path)
  msk=cv2.imread(msk_name)
  #讀灰階圖片
  #msk=cv2.imread(msk_name, cv2.IMREAD_GRAYSCALE)
  #將mask轉成只有兩種值
  #_,msk= cv2.threshold(msk, 127, 1, cv2.THRESH_BINARY)
  #用msk產生prompt
  point_coords,point_labels=generate_prompt.generate_prompt(msk)
  point_coords = ResizeLongestSide.apply_coords(point_coords, original_size)
  coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=my_device)
  labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=my_device)
  coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
  points_torch = (coords_torch, labels_torch)
  
  #msk轉tensor
  transf = torchvision.transforms.ToTensor()
  msk_tensor = transf(msk)

  with torch.no_grad():
    image_embedding = sam_model.image_encoder(input_image)
  with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points_torch,
            boxes=None,
            masks=None,
        )
  low_res_masks, iou_predictions = sam_model.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=sam_model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
  )
  upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_size).to(my_device)

  from torch.nn.functional import threshold, normalize

  binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(my_device)

  loss = dice_loss.dice_coeff(binary_mask, msk_tensor)+focal_loss.sigmoid_focal_loss(binary_mask, msk_tensor)*20
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  #測試用break
  #break

#save model
torch.save(sam_model.state_dict(), "model1.pth")
