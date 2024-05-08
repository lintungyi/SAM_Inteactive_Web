import os
import cv2
def get_msk_path(img_filename,msk_path):
    img_name=os.path.basename(img_filename)
    name,extension = os.path.splitext(img_name)
    name=name +'_m'+extension
    msk_name=os.path.join(msk_path,name)
    return msk_name