# Getting Started

This document provides instructions to set up the project on a clean installation, including downloading and preparing the data.

## Prerequisites

Before you begin, ensure you have the following software installed on your machine:

- Python 3.6 or higher
- [pip](https://pip.pypa.io/en/stable/installation/) for managing Python packages
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for version control

## Installation

1. **Clone the Repository:**

   Start by cloning the project repository to your local machine using Git:

   ```bash
   git clone https://github.com/AMMISAIDFaical/ML-engineer_chrun.git
   cd ML-engineer_chrun
   
## Set Up Virtual Environment

1. Create and activate a Python virtual environment to manage dependencies:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install Dependencies

Install the required Python packages using the requirements.txt file:
    
    ```
    pip install -r requirements.txt
    
## Configuration

The project uses a configuration file (config.ini) to manage paths for raw, processed data, and models. Here is an example configuration: **config.ini**

[PATHS]
RAW_DATA_PATH=../data/raw
MODEL_PATH=../models/
PROCESSED_DATA_PATH =../data/processed

## Data Preparation
   Download Raw Data:
   Obtain the raw data files and place them in the directory specified in RAW_DATA_PATH in the configuration file. 
   Ensure the files are named correctly, e.g., training.csv.

## Preprocess Data
   Use the provided script to preprocess the raw data. This includes cleaning and normalizing the data. 
   Run the following command:

   python -c "from common import get_preprocess_data; df = get_preprocess_data()"
   This script will load the raw data, clean it, normalize it using StandardScaler, and create dummy variables for categorical features.

### Save Processed Data

   - Save the processed data to the PROCESSED_DATA_PATH specified in your configuration:
     ```
     df.to_csv(f"{PROCESSED_DATA_PATH}/processed_data.csv", index=False)
   
## Model Training
   - Train the Model:
     Implement your model training in train.py and run it to train your models with the processed data. 
     Make sure to adjust the script according to your needs.

   python -m rexel_chrun_case_study.train.py

## Persist the Model:

   - After training, you can save the model using the persist_model function provided in your code. For example:
   from common import persist_model
   persist_model(model, MODEL_PATH, "my_model_name")

## Model Inference

Load and Use the Model:

   - To make predictions using the trained model, load it using the load_model function:
     from <module-name> import load_model 
     model = load_model(MODEL_PATH, "my_model_name")

## Notes
Ensure all necessary paths in the configuration file are set correctly. Adjust the paths, filenames, and commands
as necessary for your project setup. 

### Instructions

- Ensure that all paths and filenames in the `config.ini` file are correctly configured for your environment.
- Modify the paths and any other specifics to match your project's setup and naming conventions.
