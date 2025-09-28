import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split

class DGAPlotter:
    def __init__(self, figsize=(10, 5), dpi=150, random_state=41):
        self.figsize = figsize
        self.dpi = dpi
        self.random_state = random_state
        plt.style.use('ggplot')
    
    def plot_pr_roc_curves(self, checkpoints):
        """Plot PR and ROC curves for multiple models"""
        fig, axs = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        for checkpoint in checkpoints:
            X = checkpoint['data']
            y = checkpoint['labels']
            pipeline = checkpoint['pipeline']
            legend = checkpoint['legend']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            # ROC curve  
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            axs[0].plot(recall, precision, label=f'{legend}, AP={pr_auc:.4f}')
            axs[1].plot(fpr, tpr, label=f'{legend}, AUC={roc_auc:.4f}')
        
        # PR curve settings
        axs[0].set_title('PR Curve')
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].legend(loc='best')
        # axs[0].set_xlim(0.5, 1.1)
        # axs[0].set_ylim(0.5, 1.1)
        
        # ROC curve settings
        axs[1].set_title('ROC Curve')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_dataset(self, X, y, axis_labels, fig, subplot_num, ax_title=''):
        """Create 3D scatter plot of dataset features"""
        nrows, ncols, index = subplot_num
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
        
        ax.scatter(X[X.columns[0]][y == 1], 
                  X[X.columns[1]][y == 1], 
                  X[X.columns[2]][y == 1], 
                  c='r', label='positive', alpha=0.6)
        ax.scatter(X[X.columns[0]][y == 0], 
                  X[X.columns[1]][y == 0], 
                  X[X.columns[2]][y == 0], 
                  c='g', label='negative', alpha=0.6)
        
        ax.set_xlabel(f'${axis_labels[0]}$')
        ax.set_ylabel(f'${axis_labels[1]}$')
        ax.set_zlabel(f'${axis_labels[2]}$')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_title(f'{ax_title}')
        
        return ax