import argparse
import os
import shutil
import joblib
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.linear_model import ElasticNet

STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    train_data_path = os.path.join(split_data_dir_path, artifacts["TRAIN"])

    alpha = params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    random_state = params["base"]["random_state"]

    train_df = pd.read_csv(train_data_path)

    target_col = "quality"
    train_y = train_df[target_col]
    train_X = train_df.drop(target_col, axis=1)

    model_dir = artifacts["MODEL_DIR"]
    model_name = artifacts["MODEL_NAME"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, model_name)

    lr = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )

    lr.fit(train_X, train_y)

    joblib.dump(lr, model_file_path)
    logging.info(f"model is trained and saved at: {model_file_path}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e