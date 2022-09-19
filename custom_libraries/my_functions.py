import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def calculate_tpr_fpr(y_real, y_pred):
    """
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    """
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr
    
def get_all_roc_coordinates(y_real, y_proba):
    """
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    """
    tpr_list = [0]
    fpr_list = [0]
    resolution = 50
    
    for i in range(resolution):
        threshold = i / resolution
#         threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    """
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    """
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
def plot_overlayed_roc_curve(classes, labels, predictions, class_labels, odir, label = '', ax = None, figsize=(9, 9), ncol=2):
    """
    Plots overlayed ROC curves and returns a list of AUC.
    
    Args:
        classes: The classes used in classification. Must match with the predictions and labels.
                 i.e., if predictions consists of [..., 1, 3, 2, 1, 4, ...], then your classes cannot be ['photon', 'hadron', 'lepton']
        labels: The list of labels.
        predictions: The list of predicted classes. First dimension should match with the dimension of labels
        class_labels: List of actual names of the classes (instead of 0, 1, ...)
    Return:
        roc_auc_ovr: Dictionary of AUC, one for each class
    """
#     assert labels.size() == predictions[:, 0].size()
    if predictions.type() == 'torch.cuda.FloatTensor':
        predictions = predictions.cpu()
        
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    roc_auc_ovr = {}
    for i in range(len(classes)):
        c = classes[i]
        y_real = [1 if y == c else 0 for y in labels]
        y_proba = predictions[:, i]
        tpr, fpr = get_all_roc_coordinates(y_real, y_proba)
        
        # Calculates the ROC AUC
        roc_auc_ovr[c] = roc_auc_score(y_real, y_proba.detach())
        print(roc_auc_ovr[c])
        
        ax.plot(fpr, tpr, label = f"{class_labels[i]}: AUC_ovr = {roc_auc_ovr[c]:.3f}")

    # plot the 50/50 lines
    x = np.linspace(0, 1, 10)
    Y = x
    plt.plot(x, Y, color='k', linestyle='dashed')
    # set limits and labels on axes
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax.legend(loc='lower right', shadow=False, ncol=ncol)
    plt.title("ROC Curve OvR")
    plt.show()
    fig.savefig("%s/roc_curve_%s.pdf" % (odir, label))
    fig.savefig("%s/roc_curve_%s.png" % (odir, label))
    
    return roc_auc_ovr

def plot_class_balance(classes, labels, class_labels, odir, label = ''):
    """
    Plots a bar graph of # of data points for each class
    
    Args:
        classes: The classes used in classification. Must match with the labels.
                 i.e., if labels consists of [..., 1, 3, 2, 1, 4, ...], your classes cannot be ['photon', 'hadron', 'lepton']
        labels: The list of labels.
        class_labels: List of actual names of the classes (instead of 0, 1, ...)
    Return:
        d: dictionary of number of data points for each class
    """
    # initialize dictionary
    d = {i: 0 for i in classes}

    for data in labels:
        d[data] += 1
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(class_labels, list(d.values()))
    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of data points")
    plt.show()
    fig.savefig("%s/class_balance_%s.pdf" % (odir, label))
    fig.savefig("%s/class_balance_%s.png" % (odir, label))
    return d

def plot_class_balance_and_accuracy(class_dict, labels, class_labels, predictions, odir, label = '', width=0.8):
    '''
    Plots two bar graphs:
        a bar graph of accuracy for each class
        a bar graph of the # of data points AND the accuracy for each class
    
    Args:
        class_dict: dictionary of number of data points for each class. Can be obtained by calling plot_class_balance
        labels: list of labels. Accepted datatypes: numpy list, python list
        class_labels: List of actual names of the classes (instead of 0, 1, ...)
        predictions: The list of predicted classes.
        width: width of the bars. Default = 0.8
    '''
    classes = list(class_dict.keys())
    # calcualte the accuracy for each class
    dict = {i: [0, 0] for i in classes}
    for i in range(predictions.size()[0]):
        c = labels[i] # the true class
        dict[c][0] += 1 # the first element is the number of data points
        if c == predictions[i]:
            dict[c][1] += 1  # if the prediction matches the label, add 1 to the number of correctly predicted datapoints
    acc_lst = [elem[1] / elem[0] for elem in dict.values()]
    print(dict)
    
    # plot the accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(class_labels, acc_lst, color='tab:orange')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Accuracy')
    plt.show()

    # plot the accuracy and # of data points per class
    data_tuples = list(zip(list(class_dict.values()), acc_lst))
    df = pd.DataFrame(data_tuples, columns=['# of data points', 'Accuracy'])

    fig = plt.figure(figsize=(12, 6)) # Create matplotlib figure
    ax = fig.add_subplot(111) # Create matplotlib axes
    width = width
    _ = df.plot(kind= 'bar' , secondary_y= '# of data points' ,width=width, ax=ax, rot= 0) #TODO: use class_labels instead
    ax.set_xlabel('Classes')
    
    # to use class_labels for x-axis values
    ax.set_xticklabels(class_labels)
    
    plt.show()
    fig.savefig("%s/class_balance_and_accuracy_%s.pdf" % (odir, label))
    fig.savefig("%s/class_balance_and_accuracy_%s.png" % (odir, label))

def plot_class_balance_and_AUC(class_dict, roc_auc_ovr, class_labels, odir, label = '', figsize=(12, 6), width=0.8):
    '''
    Plots two bar graphs:
        a bar graph of AUC for each class
        a bar graph of the # of data points AND the AUC for each class
    
    Args:
        class_dict: dictionary of number of data points for each class. Can be obtained by calling plot_overlayed_roc_curve
        roc_auc_ovr: dictionary of AUC for each class.
        class_labels: list of actual names of the classes (instead of 0, 1, ...)
        width: width of the bars. Default = 0.8
    '''
    classes = list(class_dict.keys())
    
    # plot the AUC for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(class_labels, list(roc_auc_ovr.values()), color='b')
    ax.set_xlabel("Classes")
    ax.set_ylabel("AUC")
    plt.show()
    fig.savefig("%s/AUC_%s.pdf" % (odir, label))
    fig.savefig("%s/AUC_%s.png" % (odir, label))
    
    # plot the class balance AND AUC for each class
    data_tuples = list(zip(list(class_dict.values()), list(roc_auc_ovr.values())))
    df = pd.DataFrame(data_tuples, columns=["# of data points", "ROC_AUC"])
    fig = plt.figure(figsize=(12, 6)) # Create matplotlib figure
    ax = fig.add_subplot(111) # Create matplotlib axes
    width = width
    _ = df.plot(kind= 'bar' , secondary_y= '# of data points' ,width=width, ax=ax, rot= 0) #TODO: use class_labels instead
    ax.set_xlabel('classes')
    ax.set_xticklabels(class_labels)
    plt.show()
    fig.savefig("%s/class_balance_and_AUC_%s.pdf" % (odir, label))
    fig.savefig("%s/class_balance_and_AUC_%s.png" % (odir, label))
