import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, average_precision_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

class DGAModelEvaluator:
    def __init__(self, random_state=41, n_splits=5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'precision': 'precision', 
            'recall': 'recall',
            'f1': 'f1',
            'average_precision': 'average_precision'
        }
    
    def evaluate_models(self, checkpoints, use_smote=False, return_models=False):
        """Evaluate multiple models using stratified k-fold cross-validation"""
        results_list = []
        trained_models = []
        
        for checkpoint in checkpoints:
            X, y = checkpoint['data'], checkpoint['labels']
            pipeline = checkpoint['pipeline']
            
            # Add SMOTE to pipeline if requested
            if use_smote:
                if isinstance(pipeline, ImbPipeline):
                    # Check if SMOTE is already in the pipeline
                    has_smote = any('smote' in step[0].lower() for step in pipeline.steps)
                    if not has_smote:
                        # Insert SMOTE after preprocessing but before classifier
                        preprocessor_steps = []
                        classifier_step = None
                        
                        for name, step in pipeline.steps:
                            if hasattr(step, 'predict') or hasattr(step, 'predict_proba'):
                                classifier_step = (name, step)
                            else:
                                preprocessor_steps.append((name, step))
                        
                        if classifier_step:
                            new_steps = preprocessor_steps + [('smote', SMOTE(random_state=self.random_state)), classifier_step]
                            pipeline = ImbPipeline(new_steps)
            
            # Use stratified k-fold to maintain class distribution
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            try:
                cv_results = cross_validate(
                    pipeline, X, y, cv=cv, scoring=self.scoring, 
                    return_train_score=False, return_estimator=return_models
                )
                
                # Calculate mean and std of metrics
                results = {
                    'Architecture': checkpoint['architecture'],
                    'Dataset': checkpoint['dataset'],
                    'Preprocessing': checkpoint['preprocessing'],
                    'Accuracy': f"{np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}",
                    'ROC AUC': f"{np.mean(cv_results['test_roc_auc']):.4f} ± {np.std(cv_results['test_roc_auc']):.4f}",
                    'Precision': f"{np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}",
                    'Recall': f"{np.mean(cv_results['test_recall']):.4f} ± {np.std(cv_results['test_recall']):.4f}",
                    'PR AUC': f"{np.mean(cv_results['test_average_precision']):.4f} ± {np.std(cv_results['test_average_precision']):.4f}",
                    'F1 Score': f"{np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}"
                }
                
                results_list.append(results)
                
                if return_models:
                    trained_models.append({
                        'checkpoint': checkpoint,
                        'models': cv_results['estimator'],
                        'cv_results': cv_results
                    })
                    
            except Exception as e:
                warnings.warn(f"Failed to evaluate {checkpoint['architecture']} on {checkpoint['dataset']}: {str(e)}")
                continue
        
        if return_models:
            return pd.DataFrame(results_list), trained_models
        else:
            return pd.DataFrame(results_list)
    
    def train_test_evaluate(self, pipeline, X, y, test_size=0.3, use_smote=False):
        """Train and evaluate on holdout set with proper preprocessing"""
        # Split first to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Apply SMOTE only to training data if requested
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Clone pipeline to avoid fitting issues
        from sklearn.base import clone
        fitted_pipeline = clone(pipeline)
        
        # Fit and predict
        fitted_pipeline.fit(X_train, y_train)
        y_pred = fitted_pipeline.predict(X_test)
        y_prob = fitted_pipeline.predict_proba(X_test)[:, 1] if hasattr(fitted_pipeline, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_test, y_prob),
                'average_precision': average_precision_score(y_test, y_prob)
            })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        return metrics, fitted_pipeline
    
    def feature_importance_analysis(self, pipeline, X, y, feature_names, n_repeats=10):
        """Calculate feature importance using permutation importance"""
        from sklearn.base import clone
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Fit pipeline
        fitted_pipeline = clone(pipeline)
        fitted_pipeline.fit(X_train, y_train)
        
        # Calculate permutation importance
        result = permutation_importance(
            fitted_pipeline, X_test, y_test, 
            n_repeats=n_repeats, random_state=self.random_state, scoring='f1'
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df, fitted_pipeline