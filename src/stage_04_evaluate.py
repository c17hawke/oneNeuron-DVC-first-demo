import argparse
import os
import joblib
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, save_json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

STAGE = "Evaluate" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def evaluate_metrics(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    return rmse, mae, r2

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    test_data_path = os.path.join(split_data_dir_path, artifacts["TEST"])

    test_df = pd.read_csv(test_data_path)

    target_col = "quality"
    test_y = test_df[target_col]
    test_X = test_df.drop(target_col, axis=1)

    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    model_file_path = os.path.join(model_dir_path, model_name)

    lr = joblib.load(model_file_path)

    predicted_values = lr.predict(test_X)
    rmse, mae, r2 = evaluate_metrics(
        actual_values=test_y,
        predicted_values=predicted_values
    )

    scores = {
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2
    }

    scores_file_path = config["scores"]
    save_json(scores_file_path, scores)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e