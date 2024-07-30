import os

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import common
def load_train_data():
    print(f"proc split data")
    data = common.get_preprocess_data()
    data.set_index('CUSTOMER_ID', inplace=True)
    relevant_columns = [
        'HOUSE',
        'OVERCHARGE',
        'DATA',
        'INCOME',
        'OVER_15MINS_CALLS_PER_MONTH',
        'REPORTED_SATISFACTION_very_sat',
        'LEFTOVER',
        'HANDSET_PRICE',
        'AVERAGE_CALL_DURATION',
        'TIME_CLIENT',
        'REPORTED_SATISFACTION_very_unsat'
    ]

    # Assuming `df` is your dataframe and `target` is the name of the target column
    X = data[relevant_columns]
    y = data['CHURNED_STAY']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_datasets_to_csv(X_train, X_test, y_train, y_test, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define file paths
    X_train_path = os.path.join(directory, 'X_train.csv')
    X_test_path = os.path.join(directory, 'X_test.csv')
    y_train_path = os.path.join(directory, 'y_train.csv')
    y_test_path = os.path.join(directory, 'y_test.csv')

    # Save datasets to CSV
    pd.DataFrame(X_train).to_csv(X_train_path, index=False)
    pd.DataFrame(X_test).to_csv(X_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

def model_train(X_train, y_train):
    xgb_clf = xgb.XGBClassifier()
    xgb_clf = xgb_clf.fit(X_train, y_train)
    return xgb_clf


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_data()
    save_datasets_to_csv(X_train, X_test, y_train, y_test, common.PROCESSED_DATA_PATH)
    xgb_clf = model_train(X_train, y_train)
    common.persist_model(xgb_clf, common.MODEL_PATH+"model_version_1", "model_version_1")