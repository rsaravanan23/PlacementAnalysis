# PlacementAnalysis
A logistic regression-based tool that analyzes the impact of academic performance and IQ on student placement outcomes

## Project Components

### preprocessing.py
- **Purpose**: Handles the loading, preprocessing, and feature scaling of the placement dataset
- **Functionality**:
  - Loads the placement dataset Kaggle .
  - Renames relevant columns for consistency.
  - Converts placement status into a binary target variable.
  - Applies feature scaling using StandardScaler.

### predictor.py
- **Purpose**: Implements model training and evaluation logic using logistic regression
- **Functionality**:
  - Splits feature and target data into training and test sets.
  - Trains a logistic regression model on the training data.
  - Evaluates the model on test data.
  - Computes and returns classification metrics: accuracy, precision, and recall.


## Installation and Usage

To use the PlacementAnalysis model, follow these steps: 

1. Clone Repo:
   ```
   git clone https://github.com/rsaravanan23/PlacementAnalysis.git
    ```

2. Install required packages:
  '''
  pip install -r requirements.txt
  '''

## Project Findings and Insights

In this project, we trained and evaluated three logistic regression models to predict student placement outcomes based on various metrics. The models used different combinations of features derived from the placement dataset. All models were implemented using Scikit-Learn, and evaluated using standard classification metrics — accuracy, precision, and recall.

### Model Metrics

- **Only IQ Model Metrics**:
  - Accuracy: 40.0%
  - Precision: 40.0%
  - Recall: 40.0%

- **Only College GPA Model Metrics**:
  - Accuracy: 95.0%
  - Precision: 91.0%
  - Recall: 100.0%

- **Both IQ and College GPA Model Metrics**:
  - Accuracy: 85.0%
  - Precision: 89.0%
  - Recall: 80.0%

## Credits

Credits to sameerprogrammer for providing the Kaggle dataset that made this project a reality
