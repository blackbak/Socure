import requests
import numpy as np
import pandas as pd
import argparse
from utils import get_data, vote


parser = argparse.ArgumentParser()
parser.add_argument("--n_normal", type=int)
parser.add_argument("--n_anomaly", type=int)
args = parser.parse_args()


def main():
    test_data = get_data(args.n_normal, args.n_anomaly, 100)
    http_data = test_data.to_json(orient="split")
    headers = {
    'Content-Type': 'application/json',
    }
    response = requests.post("http://localhost:1234/invocations", headers=headers, data= http_data)
    predictions = np.array(response.json())
    voted_labels = vote(predictions)
    print("The predictions of the anomaly ensemble model through voting is:")
    print(voted_labels)


if __name__ == "__main__":
    main()