import time

import numpy as np
import sklearn.metrics as sm
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, vgg16
from tqdm import tqdm

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def initialize_model(backbone: str):
    if backbone == 'vgg16':
        model = vgg16(weights="DEFAULT")
        # Prepare the model for transfer learning by resetting the last layer to binary classification
        last_layer = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_layer.in_features, out_features=2)
    elif backbone.startswith('resnet'):
        if backbone == 'resnet18':
            model = resnet18(weights="DEFAULT")
        elif backbone == 'resnet34':
            model = resnet34(weights="DEFAULT")
        else:
            raise ValueError(backbone)
        # Prepare the model for transfer learning by resetting the last layer to binary classification
        last_layer = model.fc
        model.fc = nn.Linear(last_layer.in_features, out_features=2)
    else:
        raise ValueError(backbone)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    model = model.to(device)
    return model, device, criterion, optimizer


def train(model, train_loader, valid_loader, criterion, optimizer, epochs=1, metric='acc', disable_progress=False):
    """
    Train for many epochs
    """
    best_val_metric = -np.Inf
    best_epoch = 0
    best_model_filepath = './tmp/best-model.pt'
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        with tqdm(total=len(train_loader), unit='batch', desc=f"Epoch {epoch+1}", disable=disable_progress) as tepoch:
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                # Prediction loss
                output = model(X)
                loss = criterion(output, y)
                running_loss += loss.item()
                # Backprop
                loss.backward()
                optimizer.step()
                # Show train loss each iteration
                tepoch.update(1)
                tepoch.set_postfix(loss=running_loss / (i+1))
                time.sleep(0.1)

            # Evaluation metrics at end of epoch
            eval_metrics = evaluate(model, valid_loader, criterion)
            tepoch.set_postfix({'train_loss': running_loss/len(train_loader), **eval_metrics})
            # tq_batch.write(', '.join([f"{k}={v:.3f}" for k,v in eval_metrics.items()]))
            time.sleep(0.1)
            if eval_metrics[metric] > best_val_metric:
                best_val_metric = eval_metrics[metric]
                best_epoch = epoch+1
                torch.save(model.state_dict(), best_model_filepath)
    model.load_state_dict(torch.load(best_model_filepath))


def train_epoch(model, data_loader, criterion, optimizer, epoch_num=1):
    N = len(data_loader.dataset)
    running_loss = 0
    model.train()
    with tqdm(data_loader, unit='batch', desc=f"Epoch {epoch_num}") as tq_batch:
        for i, (X, y) in enumerate(tq_batch):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            # Prediction loss
            output = model(X)
            loss = criterion(output, y)
            running_loss += loss.item()

            # Backprop
            loss.backward()
            optimizer.step()

            tq_batch.set_postfix(loss=running_loss / (i+1))
            time.sleep(0.1)

            # if batch % print_every == print_every-1:
            #     print(f"[{batch*len(X):>{len(str(N))}d} / {N}] loss: {running_loss / print_every:.3f}")

    return running_loss / len(data_loader)


def evaluate(model, data_loader, criterion, log=False):
    y_true = []
    y_pred = []

    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            y_true.extend(y.cpu().numpy())
            X, y = X.to(device), y.to(device)
            output = model(X)
            _, pred = output.max(1)
            y_pred.extend(pred.cpu().numpy())
            running_loss += criterion(output, y).item()

    metrics = {
        'loss': running_loss / len(data_loader),
        'acc': sm.accuracy_score(y_true, y_pred),
        'auc': sm.roc_auc_score(y_true, y_pred),
        'f1': sm.f1_score(y_true, y_pred),
        'prec': sm.precision_score(y_true, y_pred),
        'rec': sm.recall_score(y_true, y_pred),
    }

    if log:
        print('\t'.join([f"{k}: {v:.3f}" for k,v in metrics.items()]))

    return metrics
