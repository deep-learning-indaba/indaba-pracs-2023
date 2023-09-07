
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # generating synthetic data


def generate_synthetic_data(n_samples, plot_data=False):

    """
        From: https://github.com/mbilalzafar/fair-classification
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) 
        and 1.0 means it's in non-protected group (e.g., male).
    """
    seed = 123
    disc_factor = math.pi / 4.0 # this variable determines the initial discrimination in the data -- decrease it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(size=n_samples, random_state=seed) # fix seed
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

    # join the positive and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0,n_samples*2))
    
    random.seed(seed)
    random.shuffle(perm)

    X = X[perm]
    y = y[perm]
    
    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)


    """ Generate the sensitive feature here """
    x_control = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        
        r = random.uniform(0.0, 1.0) # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            x_control.append(1.0) # 1.0 means its male
        else:
            x_control.append(0.0) # 0.0 -> female

    x_control = np.array(x_control)


    #x_control = {"s1": x_control} # all the sensitive features are stored in a dictionary
    X = np.concatenate((X,x_control.reshape(-1,1)), axis=1)
    data = np.concatenate((X,y.reshape(-1,1)),axis = 1)
    return pd.DataFrame(data=data, columns = ["x1","x2","sensitive","y"]), X, y

def plot_data(X,y,X_sens,ax=None, name=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
    num_to_draw = 200 # we will only draw a small number of points to avoid clutter
    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    x_control_draw = X_sens[:num_to_draw]
    X_s_0 = x_draw[x_control_draw == 0.0]
    X_s_1 = x_draw[x_control_draw == 1.0]
    y_s_0 = y_draw[x_control_draw == 0.0]
    y_s_1 = y_draw[x_control_draw == 1.0]
    ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='.', s=100, facecolors='none',linewidth=1.5, label= "Prot. Positive")
    ax.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='blue', marker='.', s=100, facecolors='none',linewidth=1.5, label = "Prot. Negative")
    ax.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', s=100,  label = "Non-prot. Positive")
    ax.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='blue', marker='o', s=100, label = "Non-prot. Negative")


    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.legend(handles=[p1, p2], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
#     ax.title.set_text('{} dataset'.format(name), fontsize=20)
    if name is not None: 
        fig.suptitle('{} dataset'.format(name), fontsize=20)
    plt.xlabel("score X1", fontsize=20)
    plt.ylabel("score X2", fontsize=20)
    plt.xlim((-12,10))
    plt.ylim((-7,8))
    plt.show()
