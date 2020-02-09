from sklearn import decomposition
import scipy.io.wavfile as wav
import os.path as path
import numpy as np
import librosa
import librosa.feature
import numpy as np
import scipy.stats
from sklearn import preprocessing

ALPHA = 0.97
N_MFCC = 16


def get_image_feature(data_dir, negative_num=None, positive_num=None, test_num=None):
    features = []
    if test_num != None:
        for i in range(test_num):
            image = np.load(path.join(data_dir,  str(i),
                                      "feat.npy"))
            features.append(image)
    else:
        for i in range(negative_num):
            image = np.load(path.join(data_dir, "negative",
                                      str(i), "feat.npy"))
            features.append(image)
        for i in range(positive_num):
            image = np.load(
                path.join(data_dir, "positive", str(i), "feat.npy"))
            features.append(image)
    return features


def get_audio_feature(data_dir, negative_num=None, positive_num=None, test_num=None):
    features = []
    if test_num != None:
        if path.exists("audio_test_feat.npy"):
            return np.load("audio_test_feat.npy")
        for i in range(test_num):
            y, fs = librosa.load(
                path.join(data_dir,  str(i),  "audio.wav"))
            y_emp = np.append(y[0], y[1:] - ALPHA * y[:-1])
            mfcc = librosa.feature.mfcc(y_emp, n_mfcc=N_MFCC)
            features.append(get_feature(mfcc))
    else:
        if path.exists("audio_train_feat.npy"):
            return np.load("audio_train_feat.npy")
        for i in range(negative_num):
            y, fs = librosa.load(path.join(data_dir, "negative",
                                           str(i),
                                           "audio.wav"))
            y_emp = np.append(y[0], y[1:] - ALPHA * y[:-1])
            mfcc = librosa.feature.mfcc(y_emp, n_mfcc=N_MFCC)
            features.append(get_feature(mfcc))
        for i in range(positive_num):
            y, fs = librosa.load(path.join(data_dir, "positive", str(i),
                                           "audio.wav"))
            y_emp = np.append(y[0], y[1:] - ALPHA * y[:-1])
            mfcc = librosa.feature.mfcc(y_emp, n_mfcc=N_MFCC)
            features.append(get_feature(mfcc))

    features = np.array(features)
    np.save(path.join(data_dir, "audio_feat.npy"), features)
    return features


def get_feature(mfcc):
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_skew = scipy.stats.skew(mfcc, axis=1)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d1_mean = np.mean(np.power(mfcc_d1, 2), axis=1)
    feature = np.hstack((mfcc_std, mfcc_mean, mfcc_skew, mfcc_d1_mean))
    return feature
