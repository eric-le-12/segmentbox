import numpy as np
import pandas as pd
from PIL import Image
# ftext = open("test_sg.txt")
pred = np.load("./output/pred_mask_seg_2600.npy")
im = np.squeeze(pred,axis=1)
# print(img.convert(mode='RGB').size)

with open("/root/xray_object_detection/Malaria-Project/yolov3/input_seg_2600.txt") as ftext:    
    for i,linetxt in enumerate(ftext):  
        imarr = Image.open(linetxt.replace('\n',''))     
        print(imarr.size)
        img = Image.fromarray(im[i]) 
        img = img.resize(imarr.size, Image.ANTIALIAS)
        img.convert(mode='RGB').save('{}/{}'.format('test_out',linetxt.replace('\n','').split('/')[-1]),'JPEG')               
