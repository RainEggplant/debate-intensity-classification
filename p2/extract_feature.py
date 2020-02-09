#%%
import os
import librosa
import librosa.feature
import numpy as np
import scipy.stats
from sklearn import preprocessing


#%%
TRAIN_PATH = '../dataset/train'
TEST_PATH = '../dataset/test'
ALPHA = 0.97  # 预加重系数
N_MFCC = 16


#%%
def collect_audio(mode):
    if mode == 'train':
        pos_folder = os.path.join(TRAIN_PATH, 'positive')
        pos_wav_paths = [os.path.join(pos_folder, str(i), 'audio.wav') for i in range(100)]
        neg_folder = os.path.join(TRAIN_PATH, 'negative')
        neg_wav_paths = [os.path.join(neg_folder, str(i), 'audio.wav') for i in range(100)]
        return pos_wav_paths, neg_wav_paths
    else:
        test_wav_paths = [os.path.join(TEST_PATH, str(i), 'audio.wav') for i in range(100)]
        return test_wav_paths


def calc_mfccs(wav_paths):
    mfccs = []
    for wavfile in wav_paths:
        y, fs = librosa.load(wavfile)
        y_emp = np.append(y[0], y[1:] - ALPHA * y[:-1])  # 对信号进行预加重
        mfcc = librosa.feature.mfcc(y_emp, n_mfcc=N_MFCC)
        mfccs.append(mfcc)
    return mfccs


def get_features(mfccs):
    mfcc_features = []
    for mfcc in mfccs:
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_skew = scipy.stats.skew(mfcc, axis=1)
        mfcc_d1 = librosa.feature.delta(mfcc)
        mfcc_d1_mean = np.mean(np.power(mfcc_d1, 2), axis=1)
        feature = np.hstack((mfcc_std, mfcc_mean, mfcc_skew, mfcc_d1_mean))
        mfcc_features.append(feature)

    return np.array(mfcc_features)


#%%
pos_wav_paths, neg_wav_paths = collect_audio('train')
test_wav_paths = collect_audio('test')

#%%
# 逐文件计算 MFCC, 需要较长时间
train_mfccs = calc_mfccs(pos_wav_paths + neg_wav_paths)
test_mfccs = calc_mfccs(test_wav_paths)

#%%
# 提取 MFCC 中的特征
train_feats = get_features(train_mfccs)
test_feats = get_features(test_mfccs)

#%%
# 对特征进行标准化
scaler = preprocessing.StandardScaler().fit(train_feats)
train_feats_scaled = scaler.transform(train_feats)
test_feats_scaled = scaler.transform(test_feats)

#%%
# 保存特征到文件
np.save('train_feats', train_feats_scaled)
np.save('B_feat', test_feats_scaled)
