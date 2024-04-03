import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv

from protein_data.prepare_data import DatasetName

load_dotenv()

DATASET_DIR_PATH = os.getenv('DATASET_DIR_PATH')

DATASET_NAME = 'PROTEIN_1FME'

dataset_name, shape = DatasetName[DATASET_NAME].value

nrows, ncols = shape[0]*shape[1], shape[2]

f = open(os.path.join(DATASET_DIR_PATH, dataset_name))

print(nrows)

coords = []
counter = 0
for row in f:
    if len(row.split()) != 4:
        continue
    symbol, x, y, z = row.split()
    coords.append([float(x), float(y), float(z)])
    counter = counter + 1
    if counter > nrows:
        break

f.close()

print(len(coords))

# pd.DataFrame(coords).to_csv(os.path.join(DATASET_DIR_PATH, dataset_name), index = False, header = False)
