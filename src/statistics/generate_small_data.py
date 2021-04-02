''' The file that is used to extract the samples with the driver gene across all cancers from the sample_id_sbs.organ.csv
file '''

import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'my_utilities')))
from my_utilities import my_config as cfg

use_columns = ['organ'] + cfg.GENE_NAMES
# we only need to extract those gene and their caner type to calculate their occurrence in each type in all samples
data = pd.read_csv(cfg.DATA_PATH, usecols=use_columns, low_memory=True)

data = data.fillna(0)
print(data.columns.tolist())
data.to_csv('small_data/small_data3.csv')
