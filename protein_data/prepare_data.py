import torch
import pandas as pd
import numpy as np
import os
from enum import Enum
import scipy.io as sio
# load env variables
from dotenv import load_dotenv
load_dotenv()

DATASET_DIR_PATH = os.environ['DATASET_DIR_PATH']
CODAR_FILE_PATH = f"{DATASET_DIR_PATH}/codar.mat"


class DatasetName(Enum):
    PROTEIN_19HT = '1h9t_traj.xyz', [6570, 8993, 3]
    PROTEIN_1FME = '1fme_md.xyz', [1000, 504, 3]
    PROTEIN_1BYZ = '1byz_md.xyz', [1000, 504, 3]


def check_valid_dataset(dataset_name):
    assert isinstance(
        dataset_name, str) and dataset_name in DatasetName.__members__, 'Dataset name must be a string and one of the following: PROTEIN_19HT, PROTEIN_1FME'

    return True


def get_details(dataset_name):
    check_valid_dataset(dataset_name)
    data = DatasetName[dataset_name].value

    return data


def check_pkl_file(file_name):
    file_arr = file_name.split('.')
    pkl_file_name = '_'.join(file_arr) + '.pkl'
    files = os.listdir(DATASET_DIR_PATH)

    return pkl_file_name in files, f'{DATASET_DIR_PATH}/{pkl_file_name}'


def get_coordinates(dataset_name):
    file_name, shape = get_details(dataset_name)
    is_present, pickle_file_name = check_pkl_file(file_name)

    if is_present:
        coordinates = torch.load(pickle_file_name)
        return coordinates

    
    # df = pd.read_fwf(f'{DATASET_DIR_PATH}/{file_name}')
    # raw_df = df[[df.columns[1], df.columns[2], df.columns[3]]]
    
    
    # # remove NaN
    # raw_df = raw_df.dropna()
    
    df = df.read_csv(f'{DATASET_DIR_PATH}/{file_name}', sep=' ')
    
    coordinates = []
    for i in range(df.shape[0]):
        _, x, y, z = df.iloc[i, :].values
        coordinates.append([float(x), float(y), float(z)])

    

        
    # coordinates = raw_df.to_numpy(dtype=np.float32)
    D = df.shape[0]
    D1 = D//shape[1]
    coordinates = coordinates[:(D1*shape[1])]
    coordinates = np.array(coordinates)

    shape_ = [D1, shape[1], 3]
    coordinates = coordinates.reshape(shape_)

    # reshape to series of atoms coordinates for each frame
    (x1, y1, z1) = coordinates.shape
    coordinates = coordinates.reshape(x1, y1*z1, order='C')
    # save as pickle
    torch.save(coordinates, pickle_file_name)

    return coordinates


def get_codar_coordinates():
    """Get coordinates from codar.mat file with added noise"""
    codar_file = sio.loadmat(CODAR_FILE_PATH)
    (x1, y1, z1) = codar_file['trace'].shape
    buffer_set = codar_file['trace'].copy()
    for i in range(0, 51):
        test_set = buffer_set
        D_filt = np.random.normal(1, .5, size=(9033, 3))
        test_set[i+49] += D_filt

    return test_set.reshape(x1, y1*z1, order='C')


if __name__ == '__main__':
    dataset_name = 'PROTEIN_1BYZ'
    # X = get_coordinates(dataset_name)
    X = get_codar_coordinates()
    print(X.shape)
