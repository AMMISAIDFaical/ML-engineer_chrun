from sklearn.metrics import log_loss, roc_auc_score
import mlflow
from xgboost import XGBClassifier

from ..train import load_train_data


def mlflow_track_train(X_train, X_test, y_train, y_test):

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("rexel_chrun_case_study")

    params = {
        "objective": "binary:logistic",  # Defines the learning task and objective
        "learning_rate": 0.1,  # Step size shrinkage
        "max_depth": 6,  # Maximum depth of a tree
        "n_estimators": 100,  # Number of boosting rounds (trees)
        "seed": 42  # Random seed for reproducibility
    }

    # Start MLflow run
    with mlflow.start_run():
        # Train the model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

        # Calculate metrics
        loss = log_loss(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        # Log metrics to MLflow
        mlflow.log_metric("log_loss", loss)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.set_tag("Training Infos, xgb cls chrun")

        # log the model itself
        mlflow.xgboost.log_model(model, "model")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_data()
    mlflow_track_train(X_train, X_test, y_train, y_test)
