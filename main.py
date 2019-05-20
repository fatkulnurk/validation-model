import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def read_data():
    fields = ['a', 'b', 'c', 'd', 'label']
    return pd.read_csv('data/6. iris.data.csv', names=fields)


kf = KFold(n_splits=10)
