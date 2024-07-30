import pickle
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap

# Using INI configuration file
from configparser import ConfigParser

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

config = ConfigParser()
config.read(CONFIG_PATH)
RAW_DATA_PATH = str(config.get("PATHS", "RAW_DATA_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
PROCESSED_DATA_PATH = str(config.get("PATHS", "PROCESSED_DATA_PATH"))

def get_preprocess_data():
    train_data = pd.read_csv(RAW_DATA_PATH+"/training.csv", delimiter=',')
    no_nan_data = train_data
    no_nan_data.dropna(subset=['HOUSE'], inplace=True)  # Remove rows where 'HOUSE' is NaN
    no_nan_data.dropna(subset=['LESSTHAN600k'], inplace=True)  # Remove rows where 'HOUSE' is NaN
    scaler = StandardScaler()
    norm_data = no_nan_data
    norm_data[
        ['DATA', 'INCOME', 'OVERCHARGE', 'LEFTOVER', 'HOUSE', 'REVENUE', 'HANDSET_PRICE', 'OVER_15MINS_CALLS_PER_MONTH',
         'TIME_CLIENT', 'AVERAGE_CALL_DURATION']] = scaler.fit_transform(norm_data[['DATA', 'INCOME', 'OVERCHARGE',
                                                                                    'LEFTOVER', 'HOUSE', 'REVENUE',
                                                                                    'HANDSET_PRICE',
                                                                                    'OVER_15MINS_CALLS_PER_MONTH',
                                                                                    'TIME_CLIENT',
                                                                                    'AVERAGE_CALL_DURATION']])
    # Mapping dictionary
    mapping = {'zero': 0, 'one': 1}
    TF_mapping = {False: 0, True: 1}
    # Apply the mapping
    norm_data['COLLEGE'] = norm_data['COLLEGE'].map(mapping)
    norm_data['LESSTHAN600k'] = norm_data['LESSTHAN600k'].map(TF_mapping)

    df = pd.DataFrame(norm_data)

    # Specify the columns to be converted to dummy variables
    columns_to_dummify = ['CONSIDERING_CHANGE_OF_PLAN', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL', 'CHURNED']

    # Apply get_dummies on the specified columns
    dummies = pd.get_dummies(df[columns_to_dummify])

    # Drop the original columns if you want to replace them with dummy variables
    df = df.drop(columns_to_dummify, axis=1)

    # Join the dummy variables back to the original DataFrame
    df = df.join(dummies)

    return df

# Propose DISCOUNT based on churn probability
def propose_discount(probability):
    if probability > 0.75:
        return '20%'
    elif probability > 0.5:
        return '15%'
    elif probability > 0.25:
        return '10%'
    else:
        return '5%'


def plots(fpr, tpr, roc_auc_value):
    # Create a figure
    plt.figure()

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the figure
    save_path = '../reports/figures/auc_curve.svg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='svg')

    # Show the figure
    plt.show()


# #model explainity using shap
# def explain_shap_model(model, X):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X)
#
#     # Convert SHAP values to DataFrame for better readability
#     shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
#     # Summary plot with feature names
#     shap.summary_plot(shap_values, X)

# Function to persist the model
def persist_model(model, path, model_name):
    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wb") as file:
            # Assuming you are using joblib to save the model
            import joblib
            joblib.dump(model, file)
        print(f"Model {model_name} saved to {path}")
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_model(model_path, model_name):
    full_path = os.path.join(model_path, model_name)
    print(f"Loading the model from {full_path}")
    with open(full_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully")
    return model


if __name__ == '__main__':
    # Example usage
    #xgb_clf = xgb.XGBClassifier()

    # Debugging path
    print("MODEL_PATH:", MODEL_PATH)
    print("RAW_DATA_PATH",RAW_DATA_PATH)
    print("PROCESSED_DATA_PATH", PROCESSED_DATA_PATH)

    #persist_model(xgb_clf, MODEL_PATH+"model_name", "xgb_model")

    # Example usage for loading the model
    #loaded_model = load_model(MODEL_PATH, "model_version_1")
