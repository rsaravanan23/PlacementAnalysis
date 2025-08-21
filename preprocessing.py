import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Loads data from CSV"""
    return pd.read_csv(filepath)


def get_target(data):
    """Returns the placement column as a NumPy array"""
    return data['placement'].values


def scale_features(data, feature_names):
    """Scales and returns the selected features using StandardScaler"""
    X = data[feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
