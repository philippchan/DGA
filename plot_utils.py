import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns

class DGAPlotter:
    def __init__(self, figsize=(10, 5), dpi=150, random_state=41):
        self.figsize = figsize
        self.dpi = dpi
        self.random_state = random_state
        plt.style.use('default')  # More neutral style
        sns.set_palette("colorblind")  # Colorblind-friendly palette
    
    def plot_dataset_distributions(self, datasets):
        """Plot feature distributions for all datasets"""
        n_datasets = len(datasets)
        fig, axes = plt.subplots(n_datasets, 1, figsize=(12, 4*n_datasets), dpi=self.dpi)
        
        if n_datasets == 1:
            axes = [axes]
        
        for ax, (name, dataset) in zip(axes, datasets.items()):
            # Plot original gas concentrations
            original_data = dataset['original_data']
            melted_data = original_data.melt(var_name='Gas', value_name='Concentration')
            
            sns.boxplot(data=melted_data, x='Gas', y='Concentration', ax=ax)
            ax.set_title(f'{dataset["name"]} - Gas Concentration Distributions')
            ax.tick_params(axis='x', rotation=45)
            ax.set_yscale('log')  # Log scale for concentration values
        
        plt.tight_layout()
        return fig
    
    def plot_gas_ratios(self, datasets):
        """Plot engineered gas ratio features"""
        n_datasets = len(datasets)
        ratio_columns = ['CH4_H2', 'C2H6_CH4', 'C2H4_C2H6', 'C2H2_C2H4', 'CO_CO2']
        
        fig, axes = plt.subplots(n_datasets, len(ratio_columns), 
                               figsize=(20, 4*n_datasets), dpi=self.dpi)
        
        if n_datasets == 1:
            axes = [axes]
        
        for i, (name, dataset) in enumerate(datasets.items()):
            ratios = dataset['ratios'][ratio_columns]
            
            for j, ratio_col in enumerate(ratio_columns):
                ax = axes[i][j] if n_datasets > 1 else axes[j]
                
                # Plot distribution by class
                for label in [0, 1]:
                    mask = dataset['labels'] == label
                    ax.hist(ratios[ratio_col][mask], alpha=0.7, 
                           label=f'Class {label}', bins=20)
                
                ax.set_title(f'{dataset["name"]} - {ratio_col}')
                ax.set_xlabel(ratio_col)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_pr_roc_curves_from_results(self, results_dict):
        """Plot PR and ROC curves from pre-computed results"""
        fig, axs = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        for model_name, results in results_dict.items():
            y_test = results['y_test']
            y_prob = results['y_prob']
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            # ROC curve  
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            axs[0].plot(recall, precision, label=f'{model_name} (AP={pr_auc:.3f})', linewidth=2)
            axs[1].plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2)
        
        # PR curve settings
        axs[0].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axs[0].set_xlabel('Recall', fontsize=10)
        axs[0].set_ylabel('Precision', fontsize=10)
        axs[0].legend(loc='best', fontsize=9)
        axs[0].grid(True, alpha=0.3)
        
        # ROC curve settings
        axs[1].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axs[1].set_xlabel('False Positive Rate', fontsize=10)
        axs[1].set_ylabel('True Positive Rate', fontsize=10)
        axs[1].legend(loc='lower right', fontsize=9)
        axs[1].grid(True, alpha=0.3)
        axs[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df, title='Feature Importance'):
        """Plot feature importance from permutation importance analysis"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=True)
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['importance_mean'], xerr=importance_df['importance_std'],
               align='center', alpha=0.7, ecolor='black', capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Permutation Importance (F1 Score Decrease)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_dataset(self, X, y, axis_labels, fig, subplot_num, ax_title='', use_ratios=False):
        """Create 3D scatter plot of dataset features"""
        nrows, ncols, index = subplot_num
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
        
        # Select features to plot (prioritize ratios if requested)
        if use_ratios and 'CH4_H2' in X.columns:
            feature_indices = ['CH4_H2', 'C2H2_C2H4', 'CO_CO2']
        else:
            # Use original gas concentrations
            feature_indices = X.columns[:3]
        
        # Plot by class
        scatter1 = ax.scatter(X[feature_indices[0]][y == 1], 
                            X[feature_indices[1]][y == 1], 
                            X[feature_indices[2]][y == 1], 
                            c='red', label='Fault (1)', alpha=0.7, s=50)
        scatter0 = ax.scatter(X[feature_indices[0]][y == 0], 
                            X[feature_indices[1]][y == 0], 
                            X[feature_indices[2]][y == 0], 
                            c='blue', label='Normal (0)', alpha=0.7, s=50)
        
        ax.set_xlabel(f'{feature_indices[0]}')
        ax.set_ylabel(f'{feature_indices[1]}')
        ax.set_zlabel(f'{feature_indices[2]}')
        ax.set_title(f'{ax_title}', fontweight='bold')
        ax.legend()
        
        return ax
    
    def plot_confusion_matrix_comparison(self, results_dict):
        """Plot confusion matrices for multiple models"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4), dpi=self.dpi)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, results) in zip(axes, results_dict.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Normal', 'Predicted Fault'],
                       yticklabels=['Actual Normal', 'Actual Fault'])
            ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        
        plt.tight_layout()
        return fig