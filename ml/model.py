import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.linear_model import LogisticRegression

# TODO: add necessary import

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
   # TODO: implement the function
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
    pass


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function
    preds = model.predict(X)
    return preds
    pass

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    pass

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model
    pass


def performance_on_categorical_slice(
        data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and value.

    This function filters the data for a specific value in a specified column, then processes
    this data slice using one-hot encoding for the categorical features and label binarization
    for the label. This can be used for performance evaluation in either training or
    inference/validation settings.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `X`. If None, an empty array will be returned for y.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : trained model
        Model used for prediction.

    Returns
    -------
    precision : float
        Precision of the predictions.
    recall : float
        Recall of the predictions.
    fbeta : float
        F1 score of the predictions.
    """
    # Filter data for the slice
    data_slice = data[data[column_name] == slice_value]

    # Process the data slice
    X_slice, y_slice, _, _ = process_data(
        data_slice, categorical_features=categorical_features, label=label, training=False, encoder=encoder, lb=lb
    )

    # Perform inference
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

