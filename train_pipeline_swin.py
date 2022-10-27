#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
# from eval_metrics import print_metrics_regression
from sklearn import metrics as sklearn_metrics
from torch import nn
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              SubsetRandomSampler, TensorDataset, random_split)
from torchvision import transforms
from tqdm import tqdm

from models.swin_transformer import SwinTransformer

# from openslide import OpenSlide



# import einops



# In[3]:


train = pd.read_pickle("./datasets/train.pkl")
train_x = train["x"]
train_y = train["y"]
train_id = train["id"]
train_x = torch.tensor(torch.stack(train_x).detach().cpu().numpy())
train_y = torch.tensor(train_y)

test = pd.read_pickle("./datasets/test.pkl")
test_x = test["x"]
test_y = test["y"]
test_id = test["id"]
test_x = torch.tensor(torch.stack(test_x).detach().cpu().numpy())
test_y = torch.tensor(test_y)


# In[4]:


min_label = train_y.min().item()
max_label = train_y.max().item()
train_y = (train_y-min_label)/(max_label-min_label)
test_y = (test_y-min_label)/(max_label-min_label)

min_label, max_label


# In[5]:


def min_max_norm(x, min_label=min_label, max_label=max_label):
    return (x-min_label)/(max_label-min_label)

def reverse_min_max_norm(x, min_label=min_label, max_label=max_label):
    return x*(max_label-min_label)+min_label


# In[6]:


train_x.shape, train_y.shape, len(train_id), test_x.shape, test_y.shape, len(test_id)


# In[7]:


class ImageDataset(Dataset):
    def __init__(self, x, y, biopsy_id):
        self.x = x # img_tensor_list
        self.y = y # label
        self.biopsy_id = biopsy_id

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.biopsy_id[index]

    def __len__(self):
        return len(self.x)


# In[8]:


batch_size = 32

epochs = 30
learning_rate = 1e-5
momentum = 0.9
weight_decay = 0.0 # 1e-8

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# In[9]:


train_dataset = ImageDataset(train_x, train_y, train_id)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = ImageDataset(test_x, test_y, test_id)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# In[10]:


# for data in train_dataset:
#     x, y, biopsy_id = data
#     print(x.shape, y, biopsy_id)


# In[11]:


def mse_loss(y_pred, y_true):
    loss_fn = nn.MSELoss()
    return loss_fn(y_pred, y_true)

def focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    loss = torch.mean(loss)
    return loss

def huber_loss(inputs, targets, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    loss = torch.mean(loss)
    return loss

criterion = mse_loss


# In[12]:


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler):
    train_loss = []
    model.train()
    for step, data in enumerate(dataloader):
        batch_x, batch_y, batch_biopsy_id = data
        batch_x, batch_y = (
            batch_x.float().to(device),
            batch_y.float().to(device),
        )
        optimizer.zero_grad()
        # print(batch_x.device, batch_x.shape)
        # print(next(model.parameters()).is_cuda)
        output = model(batch_x)
        output = torch.squeeze(output, dim=1)
        # print(output.shape, batch_y.shape)
        
        loss = loss_fn(output, batch_y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    metric_train_loss = np.array(train_loss).mean()
    scheduler.step(metric_train_loss)
    return metric_train_loss

def val_epoch(model, dataloader):
    y_pred = {} # key: biopsy_id, value: List[slice_stage_pred]
    y_true = {} # key: biopsy_id, value: List[slice_stage_pred]
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            # print(step)
            batch_x, batch_y, batch_biopsy_id = data
            batch_x, batch_y = (
                batch_x.float().to(device),
                batch_y.float().to(device),
            )
            output = model(batch_x)
            output = torch.squeeze(output, dim=1)
            output = output.detach().cpu().numpy().tolist()
            batch_y = batch_y.detach().cpu().numpy().tolist()

            for i in range(len(batch_biopsy_id)):
                biopsy_id = batch_biopsy_id[i]
                if biopsy_id not in y_pred:
                    y_pred[biopsy_id] = []
                    y_true[biopsy_id] = []
                y_pred[biopsy_id].append(output[i])
                y_true[biopsy_id].append(batch_y[i])
    
    prediction_list = []
    ground_truth_list = []
    for biopsy_id in y_pred:
        preds = np.array(y_pred[biopsy_id])
        truths = np.array(y_true[biopsy_id])
        prediction_list.append(preds.mean())
        ground_truth_list.append(truths.mean())
    prediction_list = np.array(prediction_list)
    ground_truth_list = np.array(ground_truth_list)
    prediction_list = reverse_min_max_norm(prediction_list)
    ground_truth_list = reverse_min_max_norm(ground_truth_list)

    mse = sklearn_metrics.mean_squared_error(ground_truth_list, prediction_list)
    return mse


# In[13]:


# model = torchvision.models.resnet18(num_classes=1)
model = SwinTransformer()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
hidden_dim = model.head.in_features
out_dim = 1

model.head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim//16),
    nn.GELU(),
    nn.Linear(hidden_dim//16, out_dim),
    # nn.Linear(hidden_dim, out_dim),
    nn.Sigmoid()
)

model.load_state_dict(torch.load('checkpoints/swin_large_patch4_window7_224_22k.pth')['model'], strict=False)

model.to(device)


# In[14]:


best_score = 1e8
for epoch in range(epochs):
    # print(f'Running epoch {epoch} ...')
    train_loss = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler
    )
    print(f"Epoch {epoch}: Loss = {train_loss}")
    if epoch % 1 == 0:
        metric_valid = val_epoch(model, test_loader)
        print("Val Score:", metric_valid)
        if metric_valid < best_score:
            best_score = metric_valid
            print("Saving best model ...")
            torch.save(
                model.state_dict(),
                f"./checkpoints/model_swin.ckpt",
            )
    


# In[ ]:


print('Best score:', best_score)

