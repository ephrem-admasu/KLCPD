

from protein_data.prepare_data import get_coordinates
from utils.klcpd_main import get_reduced_data, train_and_pred_dataset, save_preds
import torch
import numpy as np
from dotenv import load_dotenv
import os


def train(dataset_name, svd_method, components):

    load_dotenv()
    preload_model_name = f"PRE_LOAD_{dataset_name}_MODEL"
    # convert to boolean as env variable are always string
    PRE_LOAD_PROTEIN_1FME_MODEL = os.getenv(preload_model_name, False) == 'True'
    data = get_coordinates(dataset_name)
    data_reduced = get_reduced_data(data, components, svd_method)
    preds = train_and_pred_dataset(data_reduced, dataset_name.lower(), svd_method, components, preload_model=PRE_LOAD_PROTEIN_1FME_MODEL)
    save_preds(data_reduced, preds, svd_method, dataset_name.lower())

if __name__ == "__main__":
    dataset_name = 'PROTEIN_1BYZ'
    svd_method = 'random'
    components = 2
    train(dataset_name, svd_method, components)