# Socure

This is a repo showcasing an anomaly and fraud detection pipeline. Notebooks contain analysis over data and thought process from a fraud detection point of view. Python scripts are focused on deployment and testing.

Assuming that Anaconda 3 is installed in the system. We are going to train an anomaly detection model, store it and serve it as an API with mlflow. At the end we are going to test the model.

Step 1: Create a conda environment with all the dependencies.

```bash
conda env create -f conda.yaml -n socure
conda activate socure
```

Step 2: Train the model and store its version with mlflow. The model is going to be saved at the folder ./mlruns/

```bash
python train_anomaly_detection.py
```

Step 3: Serve the model with mlflow. Since this is a showcase we are going to serve it locally and without containers. Conda environments would be our "containarization". URI is the directory of the model and should look like ./mlruns/..../anomaly_model 

```bash
mlflow  models serve -m URI --port 1234
```

Step 4: Test the model running the scoring python script. Open as seperate terminal window in order to do so. We can control with arguments how many normal and anomalous points we want to test. The normal points are scored first and then the anomalous ones.

```bash
conda activate socure
python score_test.py  --n_normal 3 --n_anomaly 2
```

You could always track the activity of mlflow with the ui:

```bash
mlflow ui
```

By default is available on localhost:5000
