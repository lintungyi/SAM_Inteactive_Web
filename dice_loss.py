import torch
from torch.nn import functional as F
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1) 
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
 
    return 1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
