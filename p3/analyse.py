import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard, Checkpoint, TrainEndCheckpoint, EarlyStopping, LoadInitState, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch import NeuralNetBinaryClassifier
import numpy as np
from thop.profile import profile

from common.net import AudioNet
from common.dataset import AudioDataset
from common.preprocess import get_audio_feature

if __name__ == '__main__':

    if len(sys.argv) <= 2:
        print("Please specify train & test dataset directories!")
        sys.exit(1)

    train_dir = sys.argv[1]
    test_dir = sys.argv[2]

    run = 'debate_train_audio_0'

    writer = SummaryWriter('runs/'+run)

    # Use CUDA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using " + torch.cuda.get_device_name(device))

    # Prepare data

    print("Prepare data")

    train_feat = get_audio_feature(
        train_dir, negative_num=100, positive_num=100)
    test_feat = get_audio_feature(
        test_dir, test_num=100)

    train_dataset = AudioDataset(
        train_feat=train_feat, test_feat=test_feat, negative_num=100, positive_num=100)

    cp_from_final = Checkpoint(
        dirname='checkpoints/' + run, fn_prefix='from_train_end_')
    cp = Checkpoint(
        dirname='checkpoints/' + run)
    train_end_cp = TrainEndCheckpoint(dirname='checkpoints/'+run)
    load_state = LoadInitState(train_end_cp)

    net = NeuralNetBinaryClassifier(
        module=AudioNet,
        criterion=nn.BCEWithLogitsLoss,
        max_epochs=5000,
        lr=0.01,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        batch_size=160,
        device=device,
        callbacks=[
            ('tensorboard', TensorBoard(writer)),
            ('cp', cp),
            ('train_end_cp', train_end_cp),
            # ("load_state", load_state),
            ('early_stoping', EarlyStopping(patience=5)),
            ('lr_scheduler', LRScheduler(
                policy=ReduceLROnPlateau, monitor='valid_loss')),
        ],
    )

    print("Begin training")

    try:
        y_train = np.concatenate((
            np.zeros((100, )), np.ones((100, )))).astype('float32')
        net.fit(train_dataset, y_train)
    except KeyboardInterrupt:
        net.save_params(f_params=run+'.pkl')

    net.save_params(f_params=run + '.pkl')

    print("Finish training")

    inputs = torch.randn(160, 64).to(device)
    total_ops, total_params = profile(net.module_, (inputs,), verbose=False)
    print("%s | %s | %s" % ("Model", "Params(k)", "FLOPs(M)"))
    print("%s | %.2f | %.2f" %
          ("net.name", total_params / (1000), total_ops / (1000 ** 2)))
