import numpy as np
import torch
import texttable as tt


def test_error(y_pred,y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return np.mean(y_pred!=y_true)


def confusion_matrix(y_pred, y_true, mask=None):
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    size = y_true.size(0) if mask is None else mask.sum().item()
    sum_and_int = lambda x: x.sum().long().item()
    to_percentage = lambda l: [f'{float(y)* 100. / size :.2f}%' for y in l]
    
    y_pred_binary = (y_pred > 0.5).float()
    if mask is None:
        mask = torch.ones_like(y_pred)
        
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)
    mask = mask.float()
    
    true_positives = sum_and_int(y_true * y_pred_binary * mask)
    false_positives = sum_and_int((1 - y_true) * y_pred_binary * mask)
    
    true_negatives = sum_and_int((1 - y_true) * (1 - y_pred_binary) * mask)
    false_negatives = sum_and_int(y_true * (1 - y_pred_binary) * mask)
    

    total = true_positives + false_positives + true_negatives + false_negatives
    
    # Show the confusion matrix
    table = tt.Texttable()
    table.header(['Real/Pred', 'Positive', 'Negative', ''])
    table.add_row(['Positive'] + to_percentage([true_positives, false_negatives, true_positives + false_negatives]))
    table.add_row(['Negative'] + to_percentage([false_positives, true_negatives, false_positives + true_negatives]))
    table.add_row([''] + to_percentage([true_positives + false_positives, false_negatives + true_negatives, total]))
    
    return table



def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:            #If no figure handle is provided, it opens the current figure
        ax = plt.gca()
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)    #30 points in the grid axis
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)                 # We create a grid with the x,y coordinates defined above
    
    # From the grid to a list of (x,y) values. 
    # Check Numpy help for ravel()
    
    xy = np.vstack([X.ravel(), Y.ravel()]).T 
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    # In levels we provide a list of floating point numbers indicating 
    #the level curves to draw, in increasing order; e.g., to draw just the zero contour pass
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, marker='*', color="orange")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
 
  
    
    

    
# return decision boundary for LR
def get_lr_boundary(coef, intercept, x1, x2):
    w1, w2 = coef
    y1 = -(w1/w2)*x1 - intercept/w2
    y2 = -(w1/w2)*x2 - intercept/w2
    return [[x1,x2],[y1,y2]]