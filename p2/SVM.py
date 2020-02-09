#%%
import numpy as np
from sklearn.svm import SVC


#%%
def SVM(x):
    # train
    train_feats = np.load('train_feats.npy')
    train_targs = np.zeros(200)
    train_targs[0:100] = 1
    clf = SVC()
    clf.fit(train_feats, train_targs)
    return clf.predict(x)


#%%
# 测试用
test_feats = np.load('B_feat.npy')
result = SVM(test_feats)
np.save('B', result)
