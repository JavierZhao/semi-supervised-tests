import math
import logging
import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import index_to_mask

import sys
sys.path.insert(0, '..')  #go up one directory
from src.data.jetnet_graph import JetNetGraph
from src.models.unimp_model import UniMP
from src.models.mask_feature import MaskFeature
from custom_libraries.my_functions import *
from focal_loss import FocalLoss

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

logging.basicConfig(level=logging.INFO)


def train(model, loader, optimizer, label_rate=0.85, loss_fcn=F.cross_entropy):
    model.train()

    sum_loss = 0
    sum_true = 0
    sum_all = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        train_mask = torch.ones_like(data.x[:, 0], dtype=torch.bool)
        propagation_mask = MaskFeature.ratio_mask(train_mask, ratio=label_rate)
        supervision_mask = train_mask ^ propagation_mask

        data = data.cuda()
        out = model(data.x, data.y, data.edge_index, propagation_mask)
        loss = loss_fcn(out[supervision_mask], data.y[supervision_mask])
        loss.backward()
        sum_loss += float(loss)
        optimizer.step()

        pred = out[supervision_mask].argmax(dim=-1)
        sum_true += int((pred == data.y[supervision_mask]).sum())
        sum_all += pred.size(0)
        logging.info(f"Batch: {i + 1:03d}, Train Loss: {sum_loss:.4f}")

    return float(sum_loss) / (i + 1), float(sum_true) / sum_all


@torch.no_grad()
def test(model, loader, label_rate=0.85, output_pred=False):
    model.eval()

    sum_true = 0
    sum_all = 0
    out_lst, pred_lst, supervision_mask_lst, test_mask_lst = [], [], [], []
    for data in loader:
        data = data.cuda()
        test_mask = torch.ones_like(data.x[:, 0], dtype=torch.bool)
        propagation_mask = MaskFeature.ratio_mask(test_mask, ratio=label_rate, fix_seed=True)
        supervision_mask = test_mask ^ propagation_mask

        out = model(data.x, data.y, data.edge_index, propagation_mask)
        pred = out[supervision_mask].argmax(dim=-1)
        sum_true += int((pred == data.y[supervision_mask]).sum())
        sum_all += pred.size(0)
        
        out_lst.append(out)
        pred_lst.append(pred)
        test_mask_lst.append(test_mask)
        supervision_mask_lst.append(supervision_mask)
    if output_pred:
        return out_lst, pred_lst, float(sum_true) / sum_all, test_mask_lst, supervision_mask_lst
    else:
        return float(sum_true) / sum_all


def collate_fn(items):
    sum_list = sum(items, [])
    return Batch.from_data_list(sum_list)

def reset_params(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(layer)
            layer.reset_parameters()
        else:
            for sublayer in layer:
                if hasattr(sublayer, 'reset_parameters'):
                    print(sublayer)
                    sublayer.reset_parameters()

def plot_acc(train_loss_lst, train_acc_lst, val_acc_lst, odir):
    fig, ax = plt.subplots(figsize=(12, 6))
    marker_size = 3
    plt.plot(train_loss_lst, "r.", markersize=marker_size, label="training loss")
    plt.plot(train_acc_lst, "b.", markersize=marker_size, label="training acc")
    plt.plot(val_acc_lst, "g.",markersize=marker_size, label="val acc")
    plt.xlabel("Epochs", fontsize=18)
    # plt.ylabel("value of params", rotation=90, fontsize=18)
    plt.legend(loc="best", fontsize=10)
    # plt.xlim([200, 500])
    plt.show()
    fig.savefig("%s/loss_acc_curve.pdf" % (odir))
    fig.savefig("%s/loss_acc_curve.png" % (odir))

def main():
    train_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", "train")
    val_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", "val")
    train_dataset = JetNetGraph(train_root, max_jets=3_000, file_start=0, file_stop=1)
    val_dataset = JetNetGraph(val_root, max_jets=3_000, file_start=1, file_stop=2)
    batch_size = 1  # 1k jets per batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate_fn
    val_loader = DataListLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    val_loader.collate_fn = collate_fn

    model = UniMP(
        in_channels=train_dataset.num_features,
        num_classes=train_dataset.num_classes,
        hidden_channels=64,
        num_layers=3,
        heads=2,
    ).to(device)

    logging.info("Model summary")
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    reset_params(model)
    num_epochs = 100
    train_loss_lst, val_acc_lst, train_acc_lst = [], [], []
    best_val_acc = 0
    # Set up Focal loss
    gamma = 1
    floss = FocalLoss(gamma=gamma, reduction='mean')

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fcn=floss)
        val_acc = test(model, val_loader)
        logging.info(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        is_best_epoch = (val_acc > best_val_acc)
        if is_best_epoch:
            # Save the model
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            save_dir = "/ssl-jet-vol/semi-supervised-tests/trained_models/" + "_best_epoch_full.pt"
            torch.save(best_model_state, save_dir)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        val_acc_lst.append(val_acc)
        logging.info(
                    "Epoch #%d: Current validation metric: %.5f (best: %.5f)"
                    % (epoch, val_acc, best_val_acc)
                )
        np.savetxt("/ssl-jet-vol/semi-supervised-tests/trained_models/" + "_training_losses.txt", train_loss_lst)

    odir = "/ssl-jet-vol/semi-supervised-tests/plots"
    plot_acc(train_loss_lst, train_acc_lst, val_acc_lst, odir)

    out_lst, pred_lst, test_acc, test_mask_lst, supervision_mask_lst = test(model, val_loader, output_pred=True)
    # Apply the Softmax function to the output to obtain the predicted probabilities
    m = torch.nn.Softmax(dim=1)
    out_norm_lst = []  
    for out in out_lst:
        out_norm_lst.append(m(out))
    out_norm = torch.cat(out_norm_lst) # the predicted probabilities
    pred = torch.cat(pred_lst)  # the predicted classes

    # Plot the class balance
    classes = np.array([i for i in range(5)])
    PDG_CLASSES = ["electron", "muon", "photon", "charged_hadron", "neutral_hadron"]
    for i, data in enumerate(train_loader):
        data = data.cuda()
        train_mask = torch.ones_like(data.x[:, 0], dtype=torch.bool)
        print(i)
        if i == 0:
            labels_training = data.y[train_mask].cpu().numpy()
        else:
            labels_training = np.concatenate([labels_training, data.y[train_mask].cpu().numpy()]) 
        
    class_dict = plot_class_balance(classes, labels_training, PDG_CLASSES, odir)

    # Plot the accuracy for each class
    for i, data in enumerate(val_loader):
        data = data.cuda()
        supervision_mask = supervision_mask_lst[i]
        print(i)
        if i == 0:
            labels_val = data.y[supervision_mask].cpu().numpy()
        else:
            labels_val = np.concatenate([labels_val, data.y[supervision_mask].cpu().numpy()]) 
        print(f"labels: {labels_val.shape}")
    plot_class_balance_and_accuracy(class_dict, labels_val, PDG_CLASSES, pred, odir)

    # Plot the OvR ROC Curves
    classes = np.array([i for i in range(5)])
    for i, data in enumerate(val_loader):
        data = data.cuda()
    #     test_mask = test_mask_lst[i]
        supervision_mask = supervision_mask_lst[i]
        print(i)
        if i == 0:
            labels_test = data.y[supervision_mask].cpu().numpy()
        else:
            labels_test = np.concatenate([labels_test, data.y[supervision_mask].cpu()]) 
    roc_auc_ovr = plot_overlayed_roc_curve(classes, labels_test, out_norm[torch.cat(supervision_mask_lst)][:, classes], PDG_CLASSES, odir, label="all_classes", ncol=1)

    # Plot the ROC Curves for photons and charged hadrons
    classes = [2, 3]
    class_labels = ["photon", "charged_hadron"]
    plot_overlayed_roc_curve(classes, labels_test, out_norm[torch.cat(supervision_mask_lst)][:, classes], class_labels, odir, label="pch", ncol=1)

    # Plot the AUC and class balance
    plot_class_balance_and_AUC(class_dict, roc_auc_ovr, PDG_CLASSES, odir)

if __name__ == "__main__":
    main()
