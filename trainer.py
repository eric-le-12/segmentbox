import model
import torch
import torch.nn as nn
from barbar import Bar
import numpy as np
from matplotlib import pyplot as plt 

def train_one_epoch(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
    size=120
):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    model.train()
    for idx, (data, target) in enumerate(Bar(train_loader)):
        # move-tensors-to-GPU
        data = data.to(device)
        # target=torch.Tensor(target)
        target = target.to(device)
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        labels = target.cpu().numpy().astype(bool)
        # calculate-the-batch-loss

        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        preds = (torch.sigmoid(output) > 0.5).cpu().detach().numpy()
        # calculate training metrics
        train_metrics.step(labels, preds)

    # validate-the-model
    model.eval()
    all_labels = np.empty((0,1,size,size))
    all_preds = np.empty((0,1,size,size))
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            preds = (output > 0.5).cpu().detach().numpy()
            labels = target.cpu().numpy().astype(bool)
            all_labels = np.vstack((all_labels,labels))
            all_preds = np.vstack((all_preds,preds))
            loss = criterion(output, target)

            # update-average-validation-loss
            valid_loss += loss.item() * data.size(0)

    val_metrics.step(all_labels.astype(bool), all_preds.astype(bool))
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    return (
        train_loss,
        valid_loss,
        train_metrics.epoch(),
        val_metrics.last_step_metrics(),
    )
