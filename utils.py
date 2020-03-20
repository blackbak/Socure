import numpy as np
import pandas as pd


def get_data(n_normal_samples, n_anomaly_samples, dimensions):
    mu = [50]*dimensions
    cov = np.eye(dimensions)
    normal_data = np.random.multivariate_normal(mu, cov, n_normal_samples) 
    anomaly_data = np.random.exponential(scale=50, size=(n_anomaly_samples, 100))
    dataset = np.concatenate((normal_data, anomaly_data))
    columns = []
    for i in range(1, 101):
        columns.append("col {}".format(i))
    dataset_df = pd.DataFrame(dataset, columns=columns)
    return dataset_df


def vote(predictions):
    n_models = predictions.shape[1]
    majority = int(n_models/2) + 1
    voted_predictions = predictions.sum(axis=1)
    voted_labels = [1 if x>=majority else 0 for x in voted_predictions]
    return voted_labels
