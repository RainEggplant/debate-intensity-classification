import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


class ImageDataset(Dataset):

    def __init__(self, image_feat, negative_num=None, positive_num=None, test_num=None, transform=None):
        self.image_feat = image_feat
        self.test_num = test_num
        self.negative_num = negative_num
        self.positive_num = positive_num
        self.transform = transform

    def __len__(self):
        return self.negative_num + self.positive_num if self.test_num == None else self.test_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.test_num != None:
            image = self.image_feat[idx]
            sample = image.reshape(-1).astype('float32'), 0.5
        else:
            if idx >= self.negative_num:
                image = self.image_feat[idx]
                sample = image.reshape(-1).astype('float32'), 1.0
            else:
                image = self.image_feat[idx]
                sample = image.reshape(-1).astype('float32'), 0.0

        if self.transform:
            sample = self.transform(sample)

        return sample


class AudioDataset(Dataset):

    def __init__(self, train_feat, test_feat, negative_num=None, positive_num=None, test_num=None, transform=None):
        scaler = preprocessing.StandardScaler().fit(train_feat)
        train_feats_scaled = scaler.transform(train_feat)
        test_feats_scaled = scaler.transform(test_feat)
        self.train_feat = train_feats_scaled
        self.test_feat = test_feats_scaled
        self.test_num = test_num
        self.negative_num = negative_num
        self.positive_num = positive_num
        self.transform = transform

    def __len__(self):
        return self.negative_num + self.positive_num if self.test_num == None else self.test_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.test_num != None:
            audio = self.test_feat[idx]
            sample = audio.reshape(-1).astype('float32'), 0.5
        else:
            if idx >= self.negative_num:
                audio = self.train_feat[idx]
                sample = audio.reshape(-1).astype('float32'), 1.0
            else:
                audio = self.train_feat[idx]
                sample = audio.reshape(-1).astype('float32'), 0.0

        if self.transform:
            sample = self.transform(sample)

        return sample
