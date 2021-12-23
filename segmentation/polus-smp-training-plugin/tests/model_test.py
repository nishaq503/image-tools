import unittest

import sys, os, json
import copy

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score, jaccard_score
from tqdm import tqdm

import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

polus_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/segmentation/polus-smp-training-plugin/"
sys.path.append(polus_dir)

from src.utils import LocalNorm
from src.utils import Dataset
from src.utils import MODELS
from src.training import initialize_model
from src.training import initialize_dataloader
from src.utils import METRICS
from src.utils import LOSSES

import torch

def getLogs(logCSV   : str,
            patience : int,
            log_dict : dict) -> dict:
    
    with open(logCSV, 'r') as csvfile:
        for row in list(csv.reader(csvfile))[-1*patience:]:
            for parameter in row:
                parameter = parameter.split(":")
                key = parameter[0].strip(" ")
                value = float(parameter[1].strip(" "))
                log_dict[key]['all'].append(value)
                log_dict[key]['avg'] += value/patience
                log_dict[key]['mini'] = np.minimum(log_dict[key]['mini'], value)
                log_dict[key]['maxi'] = np.maximum(log_dict[key]['maxi'], value)
    
    return log_dict

modelDir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/OTHER_output_smp/output_SMP_two/Unet-MCCLoss-resnet18-random-Adam"
images_test_dir = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/image/"
labels_test_dir = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/groundtruth_centerbinary_1pixelsmaller/"

num_images = os.listdir(images_test_dir)
num_labels = os.listdir(labels_test_dir)
assert num_images == num_labels

bestmodelPath = os.path.join(modelDir, "model.pth")
configPath    = os.path.join(modelDir, "config.json")
configObj     = open(configPath, 'r')
configDict    = json.load(configObj)

bestmodel = torch.load(bestmodelPath).cpu()
loss = configDict['lossName']

patience  = configDict['patience']
maxEpochs = configDict['maxEpochs']

metrics = list(metric() for metric in METRICS.values())
metric_outputs = {}
trainlog_dict = {}
validlog_dict = {}

for metric in metrics:
    metric_outputs[metric.__name__] = {'avg': 0, 'maxi': 0, 'mini': 1}
    trainlog_dict[metric.__name__] = {'all': [], 'avg': 0, 'maxi': 0, 'mini': 1}
    validlog_dict[metric.__name__] = {'all': [], 'avg': 0, 'maxi': 0, 'mini': 1}
metric_outputs[loss] = {'avg': 0, 'maxi': 0, 'mini': 1}
trainlog_dict[loss] = {'all': [], 'avg': 0, 'maxi': 0, 'mini': 1}
validlog_dict[loss] = {'all': [], 'avg': 0, 'maxi': 0, 'mini': 1}

traincsv_path = os.path.join(modelDir, "trainlogs.csv")
validcsv_path = os.path.join(modelDir, "validlogs.csv")

trainlog_dict = getLogs(logCSV = traincsv_path, patience = patience, log_dict = trainlog_dict)
validlog_dict = getLogs(logCSV = validcsv_path, patience = patience, log_dict = validlog_dict)

test_loader = tqdm(initialize_dataloader(
    images_dir=images_test_dir,
    labels_dir=labels_test_dir,
    pattern="nuclear_test_61{x}.tif",
    batch_size=1,
))

test_loader_len = len(test_loader)


# metrics = list(metric() for metric in METRICS.values())
for test in test_loader:
    test0 = test[0]
    test1 = test[1]
    pr_mask = bestmodel.predict(test0)
    for metric in metrics:
        metric_value = (METRICS[metric.__name__].forward(self=metric, y_pr=pr_mask, y_gt=test1).numpy())
        metric_outputs[metric.__name__]['avg'] += metric_value/test_loader_len
        metric_outputs[metric.__name__]['mini'] = np.minimum(metric_value, metric_outputs[metric.__name__]['mini'])
        metric_outputs[metric.__name__]['maxi'] = np.maximum(metric_value, metric_outputs[metric.__name__]['maxi'])
    
    metric_loss = (LOSSES[loss].forward(self=metric, y_pred=pr_mask, y_true=test1).numpy())
    metric_outputs[loss]['avg'] += metric_loss/test_loader_len
    metric_outputs[loss]['mini'] = np.minimum(metric_loss, metric_outputs[loss]['mini'])
    metric_outputs[loss]['maxi'] = np.maximum(metric_loss, metric_outputs[loss]['maxi'])
    
for metric in metrics:
    print(metric.__name__)
    metric_avg = metric_outputs[metric.__name__]['avg']
    print("METRIC", metric_outputs[metric.__name__])
    print("TRAIN LOG", trainlog_dict[metric.__name__])
    print(" ")
    assert metric_avg > trainlog_dict[metric.__name__]['mini']
    assert metric_avg > trainlog_dict[metric.__name__]['avg'] - (trainlog_dict[metric.__name__]['avg'] * .1)
     
    
print(loss)
print("METRIC ", metric_outputs[loss])
print("TRAIN LOG", trainlog_dict[loss])
print(" ")