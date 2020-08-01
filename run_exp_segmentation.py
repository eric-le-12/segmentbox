import json
from data_loader import segmentation_dataloader as dataloader
from model import segmentation as cls
from utils import metrics as metrics
from utils import logger
from pywick import losses as custom_loss
from data_loader import transform
from torchsummary import summary
import pandas as pd
import torch
import torch.nn as nn
import trainer
import test as tester
import logging
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ssl
import os
from datetime import datetime
import argparse


def main():
    # define argument for running experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)

    parser.add_argument(
        "-k",
        "--kfold",
        help="whether applied k fold or not, provide path to k fold folder",
        default=None,
    )

    parser.add_argument(
        "-tta",
        "--autotta",
        help="whether test time augmentation is applied or not",
        default=False,
    )

    args = parser.parse_args()

    path_to_config = args.config
    tta = args.autotta
    kfold = args.kfold

    # read configure file
    with open(path_to_config, "r") as f:
        cfg = json.load(f)

    # using parsed configurations to create a dataset
    data = cfg["data"]["data_csv_name"]
    data_path = cfg["data"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])
    validation_split = float(cfg["data"]["validation_ratio"])
    # create dataset
    training_set = pd.read_csv(data, usecols=["file_name", "mask"])
    train, test, _, _ = dataloader.data_split(training_set, validation_split)

    training_set = dataloader.SegmentationDataset(
        train, data_path, transform.train_transform
    )

    testing_set = dataloader.SegmentationDataset(
        test, data_path, transform.val_transform
    )
    # create dataloaders
    # global train_loader
    # global val_loader
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=batch_size, shuffle=False,
    )

    path_to_transform = "data_loader/transform.py"
    logging.info("Dataset and Dataloaders created")
    # create a model
    extractor_name = cfg["train"]["extractor"]
    model = cls.SegmentationModel(model_name=extractor_name).create_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {} ".format(device))
    # convert to suitable device
    # global model
    model = model.to(device)
    logging.info(summary(model, input_size=(3, 224, 224)))

    logging.info("Model created...")
    # create a metric for evaluating
    # global train_metrics
    # global val_metrics
    train_metrics = metrics.Metrics(cfg["train"]["metrics"])
    val_metrics = metrics.Metrics(cfg["train"]["metrics"])
    print("Metrics implemented successfully")

    # method to optimize the model
    # read settings from json file
    loss_function = cfg["optimizer"]["loss"]
    optimizers = cfg["optimizer"]["name"]
    learning_rate = cfg["optimizer"]["lr"]
    print(loss_function)
    # initlize optimizing methods : lr, scheduler of lr, optimizer
    try:
        # if the loss function comes from nn package
        criterion = getattr(
            custom_loss, loss_function, "The loss {} is not available".format(loss_function)
        )
    except:
        # use custom loss
        criterion = getattr(
            nn,
            loss_function,
            "The loss {} is not available".format(loss_function),
        )
    criterion = criterion()
    optimizer = getattr(
        torch.optim, optimizers, "The optimizer {} is not available".format(optimizers)
    )
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    save_method = cfg["train"]["lr_scheduler_factor"]
    patiences = cfg["train"]["patience"]
    lr_factor = cfg["train"]["reduce_lr_factor"]
    scheduler = ReduceLROnPlateau(
        optimizer, save_method, patience=patiences, factor=lr_factor
    )

    # before training, let's create a file for logging model result
    time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
    log_file = logger.make_file(cfg["session"]["sess_name"], time_str)
    logger.log_initilize(log_file)
    print("Beginning training...")
    # export the result to log file
    logging.info("-----")
    logging.info(
        "session name: {} \n".format(
            cfg["session"]["sess_name"] + "_" + time_str + ".txt"
        )
    )

    logging.info("\n")
    logging.info("CONFIGS \n")
    # logging the configs:
    logging.info(open(path_to_config, "r").read())
    logging.info("-----")
    logging.info("\n Transformation")
    logging.info(open(path_to_transform, "r").read())

    # training models
    num_epoch = int(cfg["train"]["num_epoch"])
    size = int(cfg["data"]["size"])
    best_val_acc = 0
    for i in range(0, num_epoch):
        loss, val_loss, train_result, val_result = trainer.train_one_epoch(
            model,
            train_loader,
            val_loader,
            device,
            optimizer,
            criterion,
            train_metrics,
            val_metrics,
            size
        )

        # lr scheduling
        scheduler.step(val_loss)
        logging.info("\n LR = " + str(optimizer.param_groups[0]["lr"]))
        logging.info(
            "\n Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(
                i + 1, num_epoch, loss
            )
        )
        logging.info(train_result)
        logging.info(
            " \n Validation loss : {} - Other validation metrics:".format(val_loss)
        )
        logging.info(val_result)
        logging.info("\n")
        # saving epoch with best validation accuracy
        if best_val_acc < float(val_result["iou_numpy"]):
            logging.info(
                "Validation iou= "
                + str(val_result["iou_numpy"])
                + "===> Save best epoch"
            )
            best_val_acc = val_result["iou_numpy"]
            torch.save(
                model.state_dict(),
                "saved/models/" + time_str + "-" + cfg["train"]["save_as_name"],
            )
        else:
            logging.info(
                "Validation iou= "
                + str(val_result["iou_numpy"])
                + "===> No saving"
            )
            continue

    # testing on test set
    test_data = cfg["data"]["test_csv_name"]
    data_path = cfg["data"]["data_path"]
    test_df = pd.read_csv(test_data, usecols=["file_name", "mask"])

    # prepare the dataset
    testing_set = dataloader.SegmentationDataset(
        test_df, data_path, transform.val_transform
    )

    # make dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False)
    print("\n Inference on the testing set")

    # load the test model and making inference
    test_model = cls.SegmentationModel(model_name=extractor_name).create_model()
    model_path = os.path.join(
        "saved/models", time_str + "-" + cfg["train"]["save_as_name"]
    )
    test_model.load_state_dict(torch.load(model_path))
    test_model = test_model.to(device)

    if tta:
        logging.info(tester.test_result_with_tta(test_model, test_loader, device))
    else:
        logging.info(tester.test_result(test_model, test_loader, device,size))
    
    print("\n Exiting....")
    # saving torch models


if __name__ == "__main__":
    main()
