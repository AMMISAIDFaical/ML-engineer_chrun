import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import common

def evaluate_model(model, X_test,y_test):
    print(f"Evaluating the model")
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc}")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_value = auc(fpr, tpr)

    #plot
    common.plots(fpr, tpr, roc_auc_value)

    #saving results csv
    customer_ids = X_test.index

    # Predict probabilities and labels
    churn_probabilities = model.predict_proba(X_test)[:, 1]
    churn_labels = model.predict(X_test)
    real_churn_labels = ['LEAVE' if label == 0 else 'STAY' for label in y_test["CHURNED_STAY"]]
    churn_labels = ['LEAVE' if label == 0 else 'STAY' for label in churn_labels]

    # Determine CLIENT_TO_CONTACT based on churn probability threshold (e.g., 0.5)
    client_to_contact = ['YES' if prob > 0.5 else 'NO' for prob in churn_probabilities]

    discounts = [common.propose_discount(prob) for prob in churn_probabilities]
    # Create the DataFrame
    results_df = pd.DataFrame({
        'CUSTOMER_ID': customer_ids,
        'THE GROUND TRUTH CHRUNED': real_churn_labels,
        'THE PRED CHURNED': churn_labels,
        'CHURN_PROBABILITY': churn_probabilities,
        'CLIENT_TO_CONTACT': client_to_contact,
        'DISCOUNT': discounts
    })

    #saving the resulted df
    results_df.to_csv('../data/results/results_df.csv', sep=',', encoding='utf-8', index=False, header=True)

    return roc_auc_value,results_df



if __name__ == "__main__":
    X_test = pd.read_csv('../data/processed/X_test.csv', delimiter=',')
    y_test = pd.read_csv('../data/processed/y_test.csv', delimiter=',')
    model = common.load_model(common.MODEL_PATH, "model_version_1")
    roc_auc_value,results_df = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {roc_auc_value:.2f}")
    print()
    print(results_df)