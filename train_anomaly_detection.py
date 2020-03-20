import numpy as np
import pandas as pd
import pyod
import suod
from suod.models.base import SUOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn


def get_data(n_normal_samples, n_anomaly_samples, dimensions):
    mu = [50]*dimensions
    cov = np.eye(dimensions)
    normal_data = np.random.multivariate_normal(mu, cov, 1000) 
    anomaly_data = np.random.exponential(scale=50, size=(10, 100))
    dataset = np.concatenate((normal_data, anomaly_data))
    return dataset

def vote(predictions):
    n_models = predictions.shape[1]
    majority = int(n_models/2) + 1
    voted_predictions = predictions.sum(axis=1)
    voted_labels = [1 if x>=majority else 0 for x in voted_predictions]
    return voted_labels

def train():
    dataset = get_data(1000, 10, 100)
    contamination = 0.01
    base_estimators = [
        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        PCA(contamination=contamination),
        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination)]
    model = SUOD(base_estimators=base_estimators, n_jobs=6,  
                rp_flag_global=True,  
                bps_flag=True,  
                approx_flag_global=False, 
                contamination=contamination)
    model.fit(dataset)  
    model.approximate(dataset)  
    predicted_labels = model.predict(dataset)
    voted_labels = vote(predicted_labels)
    true_labels = [0]*1000 + [1]*10
    auc_score = roc_auc_score(voted_labels, true_labels)
    print("The resulted area under the ROC curve score is {}".format(auc_score))
    mlflow.log_metric("auc_score", auc_score)
    mlflow.sklearn.log_model(model, "anomaly_model")


if __name__ == "__main__":
    train()