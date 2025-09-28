import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, average_precision_score
)

class DGAModelEvaluator:
    def __init__(self, random_state=41):
        self.random_state = random_state
        self.scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'precision': 'precision', 
            'recall': 'recall',
            'f1': 'f1',
            'average_precision': 'average_precision'
        }
    
    def evaluate_models(self, checkpoints):
        """Evaluate multiple models using cross-validation"""
        results_list = []
        
        for checkpoint in checkpoints:
            X, y = checkpoint['data'], checkpoint['labels']
            cv_results = cross_validate(
                checkpoint['pipeline'], X, y, cv=5, scoring=self.scoring
            )
            
            results_list.append({
                'Architecture': checkpoint['architecture'],
                'Dataset': checkpoint['dataset'],
                'Preprocessing': checkpoint['preprocessing'],
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'ROC AUC': np.mean(cv_results['test_roc_auc']),
                'Precision': np.mean(cv_results['test_precision']),
                'Recall': np.mean(cv_results['test_recall']),
                'PR AUC': np.mean(cv_results['test_average_precision']),
                'F1 Score': np.mean(cv_results['test_f1'])
            })
        
        return pd.DataFrame(results_list)
    
    def train_test_evaluate(self, pipeline, X, y, test_size=0.3):
        """Train and evaluate on holdout set"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'average_precision': average_precision_score(y_test, y_prob)
        }