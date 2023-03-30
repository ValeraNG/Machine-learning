import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples = 50, centers = 2, random_state = 0, cluster_std = 0.60)

# ===================================
index = 1
for _x, _y in zip(X, y):
    print(index, ')\t', _y, ': ', _x)
    index += 1
# ===================================

model = SVC(kernel = 'linear', C = 1E10) 
model.fit(X, y) 

# ======================================================================
print('Accuracy of linear kernel:', accuracy_score(y, model.predict(X)))
# ======================================================================

def plot_svc_decision_function(model, ax = None, plot_support = True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    
    Y, X = np.meshgrid(y, x) 
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors = 'k', levels = [-1, 0, 1], alpha = 0.5, linestyles = ['--', '-', '--'])

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'autumn')
plot_svc_decision_function(model);
plt.show()