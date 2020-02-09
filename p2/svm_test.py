import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


#%%
# load data
train_feats = np.load('train_feats.npy')
train_targs = np.zeros(200)
train_targs[0:100] = 1

#%%
# cross-validation
X, y = shuffle(train_feats, train_targs, random_state=1939)  # 打乱
clf = SVC()
scores = cross_val_score(clf, X, y, cv=4, scoring='accuracy')
print(scores)
print('Avg accuracy:', scores.mean())
