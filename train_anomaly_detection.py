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
from utils import get_data, vote


def train():
    dataset = get_data(1000, 10, 100)
    contamination = 0.01
    with mlflow.start_run():
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
        mlflow.sklearn.log_model(model, "anomaly_model", conda_env="conda.yaml")


if __name__ == "__main__":
    train()