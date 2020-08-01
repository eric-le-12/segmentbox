import torch.nn as nn
import torch
import os
from sklearn.metrics import classification_report
from data_loader import dataloader
import pandas as pd
import json
from data_loader import transform
from model import classification as cls
import numpy as np
from utils import custom_metric as cm

# use this file if you want to quickly test your model

with open("./cfgs/tenes.cfg") as f:
    cfg = json.load(f)


def test_result(model, test_loader, device,size=224):
    # testing the model by turning model "Eval" mode
    model.eval()
    all_labels = np.empty((0,1,size,size))
    all_preds = np.empty((0,1,size,size))
    with torch.no_grad():
        for data, target in test_loader:
            # move-tensors-to-GPU
            data = data.to(device)
            # target=torch.Tensor(target)
            target = target.to(device)
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # applying Softmax to results
            output = torch.sigmoid(output)
            preds = (output > 0.5).cpu().detach().numpy()
            labels = target.cpu().numpy().astype(bool)
            all_labels = np.vstack((all_labels,labels))
            all_preds = np.vstack((all_preds,preds))

        return cm.iou_numpy(
            all_labels.astype(bool), all_preds.astype(bool)
        )


def main():
    print("Testing process beginning here....")


if __name__ == "__main__":
    main()
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data, usecols=["file_name", "label"])
    # prepare the dataset
    testing_set = dataloader.ClassificationDataset(
        test_df, data_path, transform.val_transform
    )
    # make dataloader
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False,)
    # load model
    extractor_name = cfg["train"]["extractor"]
    model = cls.ClassificationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("saved/models", cfg["train"]["save_as_name"])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # print classification report
    test_result(model, test_loader, device)
