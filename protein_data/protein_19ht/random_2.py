from ...utils.klcpd_main import get_reduced_data, train_and_pred_dataset, save_preds
from protein_data.prepare_data import get_coordinates
from dotenv import load_dotenv
import os
load_dotenv()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# convert to boolean as env variable are always string
PRE_LOAD_PROTEIN_19HT_MODEL = os.getenv('PRE_LOAD_PROTEIN_19HT_MODEL', False) == 'True'
dataset_name = 'PROTEIN_19HT'
data = get_coordinates(dataset_name)
svd_method = 'random'
components = 2
data_reduced = get_reduced_data(data, components, svd_method)
preds = train_and_pred_dataset(data_reduced, dataset_name.lower(), svd_method, 
                               components, preload_model=PRE_LOAD_PROTEIN_19HT_MODEL)
save_preds(data_reduced, preds, svd_method, dataset_name.lower())