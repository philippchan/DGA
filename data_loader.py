import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer

class DGADataLoader:
    def __init__(self, data_path='./CSV/', random_state=41):
        self.data_path = data_path
        self.random_state = random_state
        self.transformer = PowerTransformer()
    
    def load_all_datasets(self):
        """Load all DGA datasets"""
        datasets = {}
        
        # TC10 dataset
        tc10 = pd.read_csv(f'{self.data_path}tc10.csv', header=0)
        datasets['tc10'] = {
            'data': tc10.iloc[:, :7],
            'labels': tc10.iloc[:, -1],
            'name': 'TC 10'
        }
        
        # Mirowski dataset
        mirowski = pd.read_csv(f'{self.data_path}mirowski.csv', header=0)
        datasets['mirowski'] = {
            'data': mirowski.iloc[:, :7],
            'labels': mirowski.iloc[:, -1],
            'name': 'Mirowski'
        }
        
        # Hong Kong dataset
        hk = pd.read_csv(f'{self.data_path}hk.csv', header=0)
        datasets['hk'] = {
            'data': hk.iloc[:, :7],
            'labels': hk.iloc[:, -1],
            'name': 'Hong Kong'
        }
        
        # Generate SMOTE dataset
        smote = SMOTE(sampling_strategy='minority', random_state=self.random_state)
        smote_data, smote_labels = smote.fit_resample(
            datasets['tc10']['data'], datasets['tc10']['labels']
        )
        datasets['smote'] = {
            'data': smote_data,
            'labels': smote_labels,
            'name': 'SMOTE'
        }
        
        return datasets
    
    def get_transformed_data(self, data):
        """Apply power transformation"""
        return pd.DataFrame(self.transformer.fit_transform(data))