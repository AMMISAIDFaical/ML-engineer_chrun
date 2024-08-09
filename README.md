# Rexel_chrun_case_study

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predict customer churn for TELCO Inc and recommend personalized discounts to maximize future profit using a provided dataset.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
    └── results        <- result csv on the test set.
│    
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for rexel_chrun_case_study
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── rexel_chrun_case_study                <- Source code for use in this project.
    │
    ├── common.py      <- Script provides helper functions for trian and eval
    │   
    ├── train.py       <- Script to train models and then use trained models and save them
    │
    └── evaluate.py       <- Script to eval and save results 
```

--------

# TELCO Churn Prediction Project

## Task Description
This project aims to predict customer churn for TELCO Inc., a telecommunications company. The primary objectives are:
- Rank customers by their probability of churning.
- Determine which customers should be contacted to minimize churn.
- Recommend personalized discounts to maximize future profits.

The results are provided in a CSV file containing churn predictions, recommended customer contacts, and suggested discounts.

## Project Structure
- **mlflow/**: Directory for MLflow experiment tracking and model versioning.
- **__init__.py**: Initialization file to make the directory importable as a module.
- **common.py**: Utility functions for model loading, plotting, and discount recommendation.
- **config.ini**: Configuration file for project parameters and MLflow settings.
- **mlflow_exp.py**: Script for managing MLflow experiments, tracking metrics, and saving models.
- **predict.py**: Script for evaluating the model, generating predictions, and saving results.
- **train.py**: Script for training the churn prediction model and saving it for future use.

## Prerequisites
- Python 3.8 or later
- pip
- MLflow

## Installation
1. **Clone the Project Repository**:
   ```sh
   git clone https://github.com/AMMISAIDFaical/ML-engineer_chrun.git

Here's your content formatted in Markdown:

```markdown
## Install Dependencies

```sh
pip install -r requirements.txt
```

## Set Up MLflow

1. Ensure MLflow is installed and set up correctly.
2. Configure the `config.ini` file with the appropriate settings for MLflow tracking.

## Setup and Usage

### Configuration

- Edit the `config.ini` file to set up paths and parameters for MLflow and data files.

### Data Preparation

- Place the training and test data CSV files in the appropriate directory.

### Train the Churn Prediction Model

```sh
python train.py
```

This script will:
- Load and preprocess the data.
- Train the model using the training dataset.
- Save the trained model for future use.

### Predict Churn and Recommend Actions

```sh
python predict.py
```

This script will:
- Load the test data.
- Load the trained model.
- Generate predictions on the test data.
- Evaluate the model using ROC AUC score.
- Save the results in a CSV file.

### Output CSV

The output file `cv_grid_xgb_results.csv` will be saved in `../data/results/` and will contain:
- `CUSTOMER_ID`: Unique identifier for each customer.
- `CHURN_Ground_truth_Label`: Actual churn status (LEAVE or STAY).
- `CHURN_PREDS`: Predicted churn status (LEAVE or STAY).
- `CHURN_PROBABILITY`: Probability of churn.
- `CLIENT_TO_CONTACT`: Whether the customer should be contacted (YES or NO).
- `DISCOUNT`: Recommended discount to offer.
- `Rank`: Rank based on the probability of churn.

## Notes

- Ensure that dataset files are correctly formatted and placed in the specified directories.
- The project includes scripts to train, evaluate, and make predictions using the churn prediction model.
- Modify `config.ini` and scripts as needed to fit your specific setup and requirements.

## Requirements

List of required Python packages:

```txt
pandas
scikit-learn
numpy
xgboost
mlflow
matplotlib
```

## Metrics

The model's performance is evaluated using the ROC AUC metric.
```
