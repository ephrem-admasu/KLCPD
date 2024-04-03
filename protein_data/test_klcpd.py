# import numpy as np
from utils.klcpd_main import get_reduced_data, train_and_pred_dataset, save_preds
from prepare_data import get_codar_coordinates

# data = np.random.rand(100,4)
dataset_name = 'CODAR'
data = get_codar_coordinates()
svd_method = 'random'
components = 3
data_reduced = get_reduced_data(data, components, svd_method)
preds = train_and_pred_dataset(data_reduced, dataset_name.lower(), svd_method, components)
save_preds(data_reduced, preds, svd_method, dataset_name.lower())



































