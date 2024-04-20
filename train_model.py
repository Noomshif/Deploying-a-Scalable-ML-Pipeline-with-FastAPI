import os

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# TODO: load the cencus.csv data
# Correct path setup when script is in the root directory
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, 'data', 'census.csv')

# Load the data
try:
    data = pd.read_csv(data_path)
    logging.info("Data loaded successfully.")
except FileNotFoundError:
    logging.error(f"Failed to load data from {data_path}")
    exit(1)  # Stop execution if file is not found

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
    # use the train dataset 
    # use training=True
    # do not need to pass encoder and lb as input
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    for slice_value in sorted(test[col].unique()):
        count = test[test[col] == slice_value].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test, col, slice_value, cat_features, "salary", encoder, lb, model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)