import numpy as np

def iou_numpy(outputs: np.array, labels: np.array,smooth=0.000001,thresholding=False):
    SMOOTH = smooth
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    if (thresholding):
        thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        return thresholded.mean()
    else:
        return iou.mean()  # Or thresholded.mean()
