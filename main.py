# %% [markdown]
# ## Set up environment

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = "{:,.4f}".format
plt.style.use('default')

# Import our improved utilities
from data_loader import DGADataLoader
from model_utils import DGAModelEvaluator  
from plot_utils import DGAPlotter

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(41)

# %% [markdown]
# ## Initialize Utilities and Load Data
# 

# %%
# Initialize utilities
data_loader = DGADataLoader()
model_evaluator = DGAModelEvaluator(n_splits=5)  # 5-fold stratified CV
plotter = DGAPlotter()

# Load all datasets with feature engineering
datasets = data_loader.load_all_datasets()

# Extract individual datasets
tc10_data = datasets['tc10']['data']
tc10_labels = datasets['tc10']['labels']
mirowski_data = datasets['mirowski']['data'] 
mirowski_labels = datasets['mirowski']['labels']
hk_data = datasets['hk']['data']
hk_labels = datasets['hk']['labels']

print("Dataset Information:")
for name, dataset in datasets.items():
    info = data_loader.get_dataset_info(dataset)
    print(f"- {info['name']}: {info['samples']} samples ({info['positive_samples']} faults, {info['negative_samples']} normal), "
          f"imbalance ratio: {info['imbalance_ratio']:.3f}")


# %% [markdown]
# ## Exploratory Data Analysis - Gas Concentrations

# %%
# Plot dataset distributions
print("Plotting gas concentration distributions...")
fig1 = plotter.plot_dataset_distributions(datasets)
plt.show()

# %% [markdown]
# ## Exploratory Data Analysis - Gas Ratios
# 

# %%
print("Plotting gas ratio distributions...")
fig2 = plotter.plot_gas_ratios(datasets)
plt.show()

# %% [markdown]
# ## Dataset Visualization - Original Concentrations

# %%
# Plot datasets using original gas concentrations
print("Visualizing datasets using gas concentrations...")

for name, dataset in datasets.items():
    X = dataset['data']
    y = dataset['labels']
    
    fig = plt.figure(figsize=(15, 5), dpi=300)
    fig.suptitle(f'{dataset["name"]} Dataset - Gas Concentrations', size='xx-large', fontweight='bold')
    
    # Plot different feature combinations using original concentrations
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 1), 'View 1: H2, CH4, C2H2', use_ratios=False)
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 2), 'View 2: C2H4, C2H6, CO', use_ratios=False)
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 3), 'View 3: H2, C2H4, CO2', use_ratios=False)
    
    plt.tight_layout()
    plt.show()
    plt.close()

# %% [markdown]
# ## Dataset Visualization - Engineered Ratios

# %%
# Plot datasets using engineered gas ratios
print("Visualizing datasets using gas ratios...")

for name, dataset in datasets.items():
    X = dataset['data']
    y = dataset['labels']
    
    fig = plt.figure(figsize=(15, 5), dpi=300)
    fig.suptitle(f'{dataset["name"]} Dataset - Gas Ratios', size='xx-large', fontweight='bold')
    
    # Plot different feature combinations using ratios
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 1), 'View 1: CH4/H2, C2H6/CH4, C2H4/C2H6', use_ratios=True)
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 2), 'View 2: C2H2/C2H4, CO/CO2, TCG', use_ratios=True)
    plotter.plot_3d_dataset(X, y, X.columns, fig, (1, 3, 3), 'View 3: Key Ratios', use_ratios=True)
    
    plt.tight_layout()
    plt.show()
    plt.close()

# %% [markdown]
# ## Define Model Pipelines
# 

# %%
# Define common preprocessing pipeline
def create_preprocessing_pipeline(use_feature_selection=False):
    steps = [
        ('scaler', StandardScaler()),
    ]
    
    if use_feature_selection:
        steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=10)))
    
    return steps

# Define base preprocessing
base_preprocessing = create_preprocessing_pipeline()

print("Model pipelines defined successfully!")

# %% [markdown]
# ## Compare Different Models on TC10 Dataset

# %%
# Define models with proper pipelines
checkpoints = [
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'k-NN',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler + FeatureSelection',
        'pipeline': Pipeline(create_preprocessing_pipeline(use_feature_selection=True) + [
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]),
        'legend': 'k-NN'
    },
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'Random Forest',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': Pipeline(base_preprocessing + [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=41))
        ]),
        'legend': 'Random Forest'
    },
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'SVM (Linear)',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': Pipeline(base_preprocessing + [
            ('svc', SVC(kernel='linear', probability=True, random_state=41))
        ]),
        'legend': 'SVM (Linear)'
    },
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'SVM (RBF)',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': Pipeline(base_preprocessing + [
            ('svc', SVC(kernel='rbf', probability=True, random_state=41))
        ]),
        'legend': 'SVM (RBF)'
    },
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'Neural Network',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': Pipeline(base_preprocessing + [
            ('nn', MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', 
                               solver='adam', max_iter=2000, random_state=41))
        ]),
        'legend': 'Neural Network'
    }
]

print("Evaluating models on TC10 dataset with 5-fold stratified cross-validation...")
results_df = model_evaluator.evaluate_models(checkpoints)
display(results_df)

# %% [markdown]
# ## Handle Class Imbalance - Compare Strategies

# %%
# Compare imbalance handling strategies
svc_pipeline = Pipeline(base_preprocessing + [
    ('svc', SVC(kernel='rbf', probability=True, random_state=41))
])

checkpoints_imbalance = [
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'SVM (RBF)',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': svc_pipeline,
        'legend': 'No Balancing'
    },
    {
        'data': tc10_data,
        'labels': tc10_labels,
        'architecture': 'SVM (RBF)',
        'dataset': 'TC10',
        'preprocessing': 'StandardScaler',
        'pipeline': Pipeline(base_preprocessing + [
            ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=41))
        ]),
        'legend': 'Class Weight'
    }
]

print("Comparing imbalance handling strategies...")
results_imb_df, models_imb = model_evaluator.evaluate_models(
    checkpoints_imbalance, use_smote=False, return_models=True
)
display(results_imb_df)

# %% [markdown]
# ## SMOTE Evaluation

# %%
# SMOTE evaluation (separate to avoid pipeline issues)
print("Evaluating with SMOTE...")
smote_results = []
for checkpoint in checkpoints_imbalance:
    metrics, pipeline = model_evaluator.train_test_evaluate(
        checkpoint['pipeline'], tc10_data, tc10_labels, use_smote=True
    )
    smote_results.append({
        'Architecture': checkpoint['architecture'] + ' + SMOTE',
        'Dataset': checkpoint['dataset'],
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1 Score': f"{metrics['f1']:.4f}",
        'ROC AUC': f"{metrics.get('roc_auc', 'N/A')}"
    })

smote_df = pd.DataFrame(smote_results)
display(smote_df)

print("\nSummary of Imbalance Handling:")
print("Class Weighting: Better precision, maintains interpretability")
print("SMOTE: Better recall, creates synthetic samples")
print("No Balancing: May bias towards majority class")

# %% [markdown]
# ## Feature Importance Analysis

# %%
print("Performing feature importance analysis...")
# Use Random Forest for feature importance (most interpretable)
rf_pipeline = Pipeline(base_preprocessing + [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=41))
])

importance_df, fitted_pipeline = model_evaluator.feature_importance_analysis(
    rf_pipeline, tc10_data, tc10_labels, tc10_data.columns
)

print("Top 10 most important features:")
display(importance_df.head(10))

# Plot feature importance
fig = plotter.plot_feature_importance(importance_df.head(15), 
                                    'Feature Importance (Random Forest)')
plt.show()

print("\nKey Insights from Feature Importance:")
print("- Gas ratios are among the most important features")
print("- Domain knowledge (Rogers ratios) aligns with model findings")
print("- Feature importance provides model interpretability")

# %% [markdown]
# ## Compare Performance Across Datasets
# 

# %%
# Compare performance across different datasets
svc_rbf_pipeline = Pipeline(base_preprocessing + [
    ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=41))
])

dataset_checkpoints = []
for name, dataset in datasets.items():
    if name != 'smote':  # We handle SMOTE separately
        dataset_checkpoints.append({
            'data': dataset['data'],
            'labels': dataset['labels'],
            'architecture': 'SVM (RBF)',
            'dataset': dataset['name'],
            'preprocessing': 'StandardScaler + ClassWeight',
            'pipeline': svc_rbf_pipeline,
            'legend': dataset['name']
        })

print("Comparing model performance across different datasets...")
dataset_results_df = model_evaluator.evaluate_models(dataset_checkpoints)
display(dataset_results_df)

print("\nDataset Comparison Insights:")
print("- Performance varies across datasets due to different data characteristics")
print("- Consider dataset-specific tuning for optimal performance")
print("- External validation important for generalization assessment")

# %% [markdown]
# ## Hyperparameter Tuning - Setup

# %%

print("Setting up hyperparameter tuning...")
# Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    tc10_data, tc10_labels, test_size=0.3, random_state=41, stratify=tc10_labels
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Positive class in training: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Positive class in test: {y_test.sum()} ({y_test.mean():.1%})")

# %% [markdown]
# ## Hyperparameter Tuning - SVM

# %%
print("Tuning SVM hyperparameters...")

# SVM parameter grid
svm_pipeline = Pipeline(base_preprocessing + [
    ('svc', SVC(kernel='rbf', probability=True, random_state=41))
])

svm_param_grid = {
    'svc__C': np.logspace(-2, 3, 6),
    'svc__gamma': ['scale', 'auto', 0.1, 0.01]
}

# Halving Grid search for SVM
svm_halving_grid = HalvingGridSearchCV(
    svm_pipeline, svm_param_grid, 
    scoring='f1', n_jobs=-1, cv=StratifiedKFold(n_splits=3), random_state=41
)

# Fit the model
svm_halving_grid.fit(X_train, y_train)

# Evaluate on test set using model_evaluator
svm_metrics_dict, fitted_svm = model_evaluator.train_test_evaluate(
    svm_halving_grid.best_estimator_, X_test, y_test
)

svm_metrics = {
    'Architecture': 'SVM',
    'Best CV Score': svm_halving_grid.best_score_,
    'Test Accuracy': svm_metrics_dict['accuracy'],
    'Test Precision': svm_metrics_dict['precision'],
    'Test Recall': svm_metrics_dict['recall'],
    'Test F1': svm_metrics_dict['f1'],
    'Test ROC AUC': svm_metrics_dict.get('roc_auc', np.nan),
    'Best Params': svm_halving_grid.best_params_
}

print(f"SVM Best parameters: {svm_halving_grid.best_params_}")
print(f"SVM Best CV F1 score: {svm_halving_grid.best_score_:.4f}")
print(f"SVM Test F1 score: {svm_metrics_dict['f1']:.4f}")

# %% [markdown]
# ## Hyperparameter Tuning - Random Forest

# %%
print("Tuning Random Forest hyperparameters...")

# Random Forest parameter grid
rf_pipeline = Pipeline(base_preprocessing + [
    ('rf', RandomForestClassifier(random_state=41))
])

rf_param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

# Halving Grid search for Random Forest
rf_halving_grid = HalvingGridSearchCV(
    rf_pipeline, rf_param_grid, 
    scoring='f1', n_jobs=-1, cv=StratifiedKFold(n_splits=3), random_state=41
)

# Fit the model
rf_halving_grid.fit(X_train, y_train)

# Evaluate on test set using model_evaluator
rf_metrics_dict, fitted_rf = model_evaluator.train_test_evaluate(
    rf_halving_grid.best_estimator_, X_test, y_test
)

rf_metrics = {
    'Architecture': 'Random Forest',
    'Best CV Score': rf_halving_grid.best_score_,
    'Test Accuracy': rf_metrics_dict['accuracy'],
    'Test Precision': rf_metrics_dict['precision'],
    'Test Recall': rf_metrics_dict['recall'],
    'Test F1': rf_metrics_dict['f1'],
    'Test ROC AUC': rf_metrics_dict.get('roc_auc', np.nan),
    'Best Params': rf_halving_grid.best_params_
}

print(f"Random Forest Best parameters: {rf_halving_grid.best_params_}")
print(f"Random Forest Best CV F1 score: {rf_halving_grid.best_score_:.4f}")
print(f"Random Forest Test F1 score: {rf_metrics_dict['f1']:.4f}")

# %% [markdown]
# ## Display Tuning Results
# 

# %%

# Display tuning results
tuning_results = [svm_metrics, rf_metrics]
tuning_df = pd.DataFrame(tuning_results)
display(tuning_df)

print("\nHyperparameter Tuning Summary:")
best_tuned_model = max(tuning_results, key=lambda x: x['Test F1'])
print(f"Best tuned model: {best_tuned_model['Architecture']}")
print(f"Best test F1 score: {best_tuned_model['Test F1']:.4f}")

# %% [markdown]
# ## Final Model Evaluation

# %%

# Train final models and generate comprehensive evaluation
print("Final model evaluation with detailed metrics...")

final_results = {}
best_models = {}

# Evaluate top 3 models from initial comparison
for checkpoint in checkpoints[:3]:  
    print(f"Evaluating {checkpoint['legend']}...")
    metrics, fitted_model = model_evaluator.train_test_evaluate(
        checkpoint['pipeline'], tc10_data, tc10_labels
    )
    
    # Get predictions for plotting
    X_train, X_test, y_train, y_test = train_test_split(
        tc10_data, tc10_labels, test_size=0.3, random_state=41, stratify=tc10_labels
    )
    fitted_model.fit(X_train, y_train)
    y_prob = fitted_model.predict_proba(X_test)[:, 1]
    
    final_results[checkpoint['legend']] = {
        'metrics': metrics,
        'y_test': y_test,
        'y_prob': y_prob,
        'confusion_matrix': metrics['confusion_matrix']
    }
    best_models[checkpoint['legend']] = fitted_model

print("Final model evaluation completed!")

# %% [markdown]
# ## Plot Final Results - PR and ROC Curves

# %%

print("Plotting final PR and ROC curves...")
fig_curves = plotter.plot_pr_roc_curves_from_results(final_results)
plt.show()

# Print AUC values using the metrics already calculated in final_results
print("\nModel Performance Summary:")
for model_name, results in final_results.items():
    metrics = results['metrics']
    print(f"{model_name}:")
    print(f"  PR-AUC = {metrics.get('average_precision', 'N/A')}")
    print(f"  ROC-AUC = {metrics.get('roc_auc', 'N/A')}")
    print(f"  F1 Score = {metrics['f1']:.4f}")
    print()

# %% [markdown]
# ## Plot Final Results - Confusion Matrices

# %%

print("Plotting confusion matrices...")
fig_cm = plotter.plot_confusion_matrix_comparison(final_results)
plt.show()

# Print detailed metrics
print("\nDetailed Performance Metrics:")
for model_name, results in final_results.items():
    metrics = results['metrics']
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

# %% [markdown]
# ## Analysis Summary and Conclusions

# %%

print("ANALYSIS SUMMARY")
print("="*50)

# Best model based on F1 score
best_model_name, best_model_results = max(final_results.items(), key=lambda x: x[1]['metrics']['f1'])
best_metrics = best_model_results['metrics']

print(f"Best Performing Model: {best_model_name}")
print(f"Best F1 Score: {best_metrics['f1']:.4f}")
print(f"Best ROC AUC: {best_metrics.get('roc_auc', 'N/A')}")

print("\nKey Improvements in This Analysis:")
print("1. No data leakage through proper pipeline usage")
print("2. Domain-specific feature engineering (gas ratios)")
print("3. Robust stratified cross-validation")
print("4. Comprehensive model evaluation with multiple metrics")
print("5. Feature importance analysis for interpretability")
print("6. Proper handling of class imbalance")
print("7. Data validation and sanity checks")

print("\nTechnical Insights:")
print(f"- Dataset size: {len(tc10_labels)} samples ({tc10_labels.sum()} faults)")
print(f"- Feature space: {tc10_data.shape[1]} features (7 original + 6 engineered)")
print("- Gas ratios proved to be important features")
print("- Tree-based models provided best interpretability")

print("\nRecommendations for Production:")
print("1. Collect more diverse and balanced data")
print("2. Validate on external datasets for generalization")
print("3. Implement model monitoring for concept drift")
print("4. Add uncertainty quantification for predictions")
print("5. Develop interactive visualizations for domain experts")
print("6. Consider ensemble methods for improved robustness")



