# -*- coding: iso-8859-1 -*-
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import math
import pandas as pd
from utils import *
from scipy.stats import wasserstein_distance
from scipy.stats import entropy


def plotLoss(loss_train, loss_val, savePath):
    loss_train = loss_train[1:]
    loss_val = loss_val[1:]
    fig = plt.figure()#ÐÂ½¨Ò»ÕÅÍ¼
    plt.plot(loss_train, label='training loss')
    #plt.plot(history.history['val_loss'],label='val loss',marker=">",c="gray")
    plt.plot(loss_val,label='val loss',)
    plt.title('model loss', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(loc='best', fontsize=12)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    fig.savefig(savePath, dpi=300)
    plt.show()

def calR(y_pred, y):
    y_pred = np.squeeze(y_pred)
    y = np.squeeze(y)
    y_flat_list = y_pred.flatten().tolist()
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, y_flat_list)
    return r_value


def pltPredict_linregress(y_pred, y, save_path):
    y_pred = np.squeeze(y_pred)
    y = np.squeeze(y)
    print("pearson: ",scipy.stats.pearsonr(y_pred, y))

    # maxValue = math.ceil(max(np.max(y), np.max(y_pred)))
    # minValue = math.ceil(min(np.min(y), np.min(y_pred)))
    maxValue = 4
    minValue = -8
    y_flat_list = y_pred.flatten().tolist()
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, y_flat_list)
    line = slope * np.array(y) + intercept
    print(slope, r_value)

    fig = plt.figure()#ÐÂ½¨Ò»ÕÅÍ¼
    plt.xlabel('true logKD', fontsize=20)
    plt.ylabel('predicted logKD', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.scatter(y, y_pred, s=50, alpha=0.8)
    plt.plot(y, line, color='black', label='slope = {0:.4f}\n R = {1:.2f}'.format(slope, r_value), lw=2)


    plt.legend()
    plt.xlim(minValue, maxValue)
    plt.ylim(minValue, maxValue)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()

def rmseAndMae(yDic, savePath):
    with open(savePath, 'w') as file:
        for key, value in yDic.items():
            file.write(f"{key}\t{value}\n")

