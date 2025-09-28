import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
import warnings

class DGADataLoader:
    def __init__(self, data_path='./CSV/', random_state=41):
        self.data_path = data_path
        self.random_state = random_state
        
    def _validate_data(self, data, dataset_name):
        """Validate dataset for common issues"""
        print(f"Validating {dataset_name} dataset...")
        
        # Check for negative values (physically impossible for gas concentrations)
        negative_mask = data < 0
        if negative_mask.any().any():
            negative_count = negative_mask.sum().sum()
            warnings.warn(f"Found {negative_count} negative values in {dataset_name}. Setting to 0.")
            data = data.clip(lower=0)
        
        # Check for missing values
        if data.isnull().any().any():
            missing_count = data.isnull().sum().sum()
            warnings.warn(f"Found {missing_count} missing values in {dataset_name}. Filling with 0.")
            data = data.fillna(0)
        
        # Check for infinite values
        if np.isinf(data).any().any():
            inf_count = np.isinf(data).sum().sum()
            warnings.warn(f"Found {inf_count} infinite values in {dataset_name}. Replacing with large finite values.")
            data = data.replace([np.inf, -np.inf], 1e6)
        
        return data
    
    def _create_gas_ratios(self, data):
        """Create domain-specific gas ratio features"""
        ratios = pd.DataFrame(index=data.index)
        
        # Rogers Ratios (standard in DGA analysis)
        ratios['CH4_H2'] = data['CH_4'] / (data['H_2'] + 1e-6)  # Avoid division by zero
        ratios['C2H6_CH4'] = data['C_2H_6'] / (data['CH_4'] + 1e-6)
        ratios['C2H4_C2H6'] = data['C_2H_4'] / (data['C_2H_6'] + 1e-6)
        ratios['C2H2_C2H4'] = data['C_2H_2'] / (data['C_2H_4'] + 1e-6)
        
        # Other important ratios
        ratios['CO_CO2'] = data['CO'] / (data['CO_2'] + 1e-6)
        ratios['C2H4_C2H2'] = data['C_2H_4'] / (data['C_2H_2'] + 1e-6)
        
        # Total combustible gas
        ratios['TCG'] = data[['H_2', 'CH_4', 'C_2H_2', 'C_2H_4', 'C_2H_6', 'CO']].sum(axis=1)
        
        # Replace infinities with large values
        ratios = ratios.replace([np.inf, -np.inf], 1e6)
        
        return ratios
    
    def load_all_datasets(self):
        """Load all DGA datasets with validation and feature engineering"""
        datasets = {}
        
        # TC10 dataset
        tc10 = pd.read_csv(f'{self.data_path}tc10.csv', header=0)
        tc10_data = tc10.iloc[:, :7]
        tc10_data = self._validate_data(tc10_data, 'TC10')
        tc10_ratios = self._create_gas_ratios(tc10_data)
        
        datasets['tc10'] = {
            'data': pd.concat([tc10_data, tc10_ratios], axis=1),
            'original_data': tc10_data.copy(),
            'ratios': tc10_ratios.copy(),
            'labels': tc10.iloc[:, -1],
            'name': 'TC 10',
            'description': 'TC10 dataset with Rogers ratios and additional features'
        }
        
        # Mirowski dataset
        mirowski = pd.read_csv(f'{self.data_path}mirowski.csv', header=0)
        mirowski_data = mirowski.iloc[:, :7]
        mirowski_data = self._validate_data(mirowski_data, 'Mirowski')
        mirowski_ratios = self._create_gas_ratios(mirowski_data)
        
        datasets['mirowski'] = {
            'data': pd.concat([mirowski_data, mirowski_ratios], axis=1),
            'original_data': mirowski_data.copy(),
            'ratios': mirowski_ratios.copy(),
            'labels': mirowski.iloc[:, -1],
            'name': 'Mirowski',
            'description': 'Mirowski dataset with Rogers ratios and additional features'
        }
        
        # Hong Kong dataset
        hk = pd.read_csv(f'{self.data_path}hk.csv', header=0)
        hk_data = hk.iloc[:, :7]
        hk_data = self._validate_data(hk_data, 'Hong Kong')
        hk_ratios = self._create_gas_ratios(hk_data)
        
        datasets['hk'] = {
            'data': pd.concat([hk_data, hk_ratios], axis=1),
            'original_data': hk_data.copy(),
            'ratios': hk_ratios.copy(),
            'labels': hk.iloc[:, -1],
            'name': 'Hong Kong',
            'description': 'Hong Kong dataset with Rogers ratios and additional features'
        }
        
        # Note: SMOTE will be applied during cross-validation, not here
        # to avoid data leakage
        
        print("Dataset summary:")
        for name, dataset in datasets.items():
            n_pos = dataset['labels'].sum()
            n_neg = len(dataset['labels']) - n_pos
            print(f"- {dataset['name']}: {len(dataset['labels'])} samples ({n_pos} positive, {n_neg} negative), {dataset['data'].shape[1]} features")
        
        return datasets
    
    def get_dataset_info(self, dataset):
        """Get detailed information about a dataset"""
        info = {
            'name': dataset['name'],
            'samples': len(dataset['labels']),
            'features': dataset['data'].shape[1],
            'positive_samples': dataset['labels'].sum(),
            'negative_samples': len(dataset['labels']) - dataset['labels'].sum(),
            'imbalance_ratio': dataset['labels'].mean(),
            'feature_names': list(dataset['data'].columns)
        }
        return info