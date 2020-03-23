import ember
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

data_dir = "/home/cuckoo/Desktop/ember/ember2018/"
feature_version = 2

if not (os.path.exists(os.path.join(data_dir, f"X_train_{feature_version}.dat")) 
        and os.path.exists(os.path.join(data_dir, f"y_train_{feature_version}.dat"))):
    print("Creating vectorized features")
    ember.create_vectorized_features(data_dir, feature_version=feature_version)

#_ = ember.create_metadata(data_dir)

#emberdf = ember.read_metadata(data_dir)
X_test, y_test = ember.read_vectorized_features(data_dir, subset="test", feature_version=feature_version)
#X_train, y_train = ember.read_vectorized_features(data_dir, subset="train", feature_version=3)
with open(os.path.join(data_dir, f"SGDR_model_{feature_version}.pkl"), 'rb') as f:
    model = pickle.load(f)
    y_test_pred = model.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, y_test_pred))

