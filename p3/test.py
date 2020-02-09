import sys
import torch
from skorch import NeuralNetBinaryClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from common.net import AudioNet
from common.dataset import AudioDataset
from common.preprocess import get_audio_feature, get_image_feature

if len(sys.argv) <= 2:
    print("Please specify train & test dataset directories!")
    sys.exit(1)

train_dir = sys.argv[1]
test_dir = sys.argv[2]

# Use CUDA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))


# Prepare data

train_feat = get_audio_feature(
    train_dir, negative_num=100, positive_num=100)
test_feat = get_audio_feature(
    test_dir, test_num=100)

test_dataset = AudioDataset(
    train_feat=train_feat, test_feat=test_feat, test_num=100)

net = NeuralNetBinaryClassifier(
    module=AudioNet,
    device=device,
)
net.initialize()
net.load_params(f_params='debate_weights.pkl')

result = net.predict(test_dataset)

result = result.reshape(-1)
np.save("C.npy", result)
