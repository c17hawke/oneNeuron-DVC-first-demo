import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random


STAGE = "GET DATA" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
    

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    remote_data_URL = config["data_source"]

    df = pd.read_csv(remote_data_URL, sep=";")
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    raw_local_dir = artifacts["RAW_LOCAL_DIR"]
    raw_local_file = artifacts["RAW_LOCAL_FILE"]

    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)

    create_directories([raw_local_dir_path])

    raw_local_filepath = os.path.join(raw_local_dir_path, raw_local_file)

    df.to_csv(raw_local_filepath, sep=",", index=False)
    logging.info(f"raw data is saved at: {raw_local_filepath}")


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