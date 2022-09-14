import json
import math
import os
import pickle
import sys

import keras
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
import numpy as np
import tensorflow as tf

# import model and load it
model_file = sys.argv[1]

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

# load in the dataset
data = tf.keras.datasets.cifar100.load_data(label_mode="fine")

# split dataset
(x_train, y_train), (x_test, y_test) = data

fpr, tpr, roc_thresholds = metrics.roc_curve(x_test, y_test, pos_label=1)

with open("api/ml/metrics/roc.json", "w") as fd:
    json.dump(
        {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        },
        fd,
        indent=4,
    )