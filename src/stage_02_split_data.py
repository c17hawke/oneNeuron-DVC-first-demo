import argparse
import os
from posixpath import split
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from src.utils.common import create_directories, read_yaml

STAGE = "SPLIT data stage" ## <<< change stage name 

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
    raw_local_dir = artifacts["RAW_LOCAL_DIR"]
    raw_local_file = artifacts["RAW_LOCAL_FILE"]
    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    raw_local_filepath = os.path.join(raw_local_dir_path, raw_local_file)

    df = pd.read_csv(raw_local_filepath)

    test_size = params["base"]["test_size"]
    random_state = params["base"]["random_state"]

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    logging.info(f"splitting of data in training and test files at test_size: {test_size}")
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    create_directories([split_data_dir_path])

    train_data_path = os.path.join(split_data_dir_path, artifacts["TRAIN"])
    test_data_path = os.path.join(split_data_dir_path, artifacts["TEST"])

    for data, data_path in (train, train_data_path), (test, test_data_path):
        data.to_csv(data_path, sep=",", index=False)
        logging.info(f"data is saved at: {data_path}")


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