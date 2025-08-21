from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def split_data(X, y, test_size=0.2, random_state=42):
    """spits the data into an 80:20 train and test split"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """logistic regression training"""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    return accuracy, precision, recall