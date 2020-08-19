import torch.nn as nn
import torch
import os
from sklearn.metrics import classification_report
from data_loader import segmentation_dataloader as dataloader
import pandas as pd
import json
from data_loader import transform
from model import segmentation as cls
import numpy as np
from utils import custom_metric as cm

# use this file if you want to quickly test your model

with open("./cfgs/segment.cfg") as f:
    cfg = json.load(f)


def test_result(model, test_loader, device,size=128):
    # testing the model by turning model "Eval" mode
    model.eval()
    all_labels = np.empty((0,1,size,size))
    all_preds = np.empty((0,1,size,size))
    all_inputs = np.empty((0,3,size,size))
    paths = []
    with torch.no_grad():
        for data,path in test_loader:            
            # move-tensors-to-GPU
            data = data.to(device)
            # target=torch.Tensor(target)
            # target = target.to(device)
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # applying Softmax to results
            output = torch.sigmoid(output)
            inputs = data.cpu().detach().numpy()
            preds = (output > 0.5).cpu().detach().numpy()
            # labels = target.cpu().numpy().astype(bool)
            # all_labels = np.vstack((all_labels,labels))
            all_preds = np.vstack((all_preds,preds))
            all_inputs = np.vstack((all_inputs,inputs))
            paths.extend([path])
        np.save("output/pred_mask_seg_2400.npy",all_preds.astype(bool))
        pathseries = pd.Series(paths)
        pathseries.to_csv('output/paths_seg_2400.csv')
        np.save("output/inputs_seg_2400.npy",all_inputs)
        print(all_inputs.shape)
        print(all_preds.shape)
        result = 0
        # result = {"iou" : cm.iou_numpy(all_labels.astype(bool), all_preds.astype(bool)),
        # "dice": cm.dice_numpy(all_labels.astype(bool), all_preds.astype(bool))}
        return result


def main():
    print("Testing process beginning here....")


if __name__ == "__main__":
    main()
    # test_data = cfg["data"]["test_csv_name"]
    test_data = "/root/xray_object_detection/Malaria-Project/yolov3/input_seg_2400.txt"
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data,usecols=["file_name"])
    # prepare the dataset
    testing_set = dataloader.SegmentationTestset(
        test_df, "", transform.val_transform
    )
    # make dataloader
    # 20200806-1551 20200806-1000-model1.pth
    
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False,)
    # load model
    extractor_name = cfg["train"]["extractor"]
    model = cls.SegmentationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("saved/models", "20200811-1717-model1.pth")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # print classification report
    print(test_result(model, test_loader, device))
