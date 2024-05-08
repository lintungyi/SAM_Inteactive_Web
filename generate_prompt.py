import numpy as np
import cv2

def generate_prompt(msk):
    img_size=msk.shape
    w=img_size[1]
    h=img_size[0]
    #找出第一個非0
    check=False
    point_X=0
    point_Y=0
    for i in range(w):
        for o in range(h):
            if(msk[i][o][0]!=0):
                point_X=i
                point_Y=o
                check=True
                break
        if(check):
            break
    points = np.array([[point_X,point_Y]])
    labels = np.array([1])
    return points,labels





