"""

This file is used to provide the hyper parameters needed

training episodes : EPOCH
learning rate : LEARNING_RATE
sbs signatures : SBS_NAMES

data path : C_V_DATA_PATH
fold numbers : CROSS_VALIDATION_COUNT

"""
# train the hyper parameters

import pandas as pd

EPOCH = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 8

C_V_DATA_PATH = '../../data/cross_valid'

CROSS_VALIDATION_COUNT = 6
SBS_NAMES = ['SBS4','SBS5','SBS1','SBS39','SBS36','SBS2','SBS13','SBS10b','SBS9','SBSPON','SBS3','SBS6','SBS30','SBSN','SBS10a',
             'SBS15','SBS26','SBS29','SBS17b','SBS87','SBS16','SBS18','SBS52','SBS8','SBS7b','SBS40','SBS50','SBS24','SBS27','SBS42',
             'SBS86','SBS57','SBS33','SBS90','SBS17a','SBS55','SBS22','SBS54','SBS48','SBS58','SBS28','SBS7a','SBS7d','SBS7c','SBS38',
             'SBS84','SBS35','SBS14','SBS44']


