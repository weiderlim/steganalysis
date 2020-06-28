import albumentations as A
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sklearn
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from albumentations.pytorch.transforms import ToTensorV2
from catalyst.data.sampler import BalanceClassSampler
from datetime import datetime
from efficientnet_pytorch import EfficientNet
from glob import glob
from skimage import io
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler

from modules.dataset_retriever import DatasetRetriever
from modules.dataset_submission_retriever import DatasetSubmissionRetriever
from modules.fitter import Fitter
from modules.train_global_config import TrainGlobalConfig

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b3')
    net._fc = nn.Linear(in_features=1536, out_features=4, bias=True)
    return net

def run_training(fitter):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )
    fitter.fit(train_loader, val_loader)

if __name__ == "__main__":

    # ARGS
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-d", "--datapath", type=str, default="./alaska2-image-steganalysis", help="Path to root data dir. Default: %(default)s")
    parser.add_argument("-c", "--checkpoint", type=str, help="Resume from checkpoint.")
    parser.add_argument("-s", "--skip_training", action='store_true', help="Skip training and evaluate test set.")
    options = parser.parse_args()

    # SETUP
    SEED = 42
    seed_everything(SEED)

    # LOAD DATA
    dataset = []
    for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
        for path in glob(os.path.join(options.datapath, 'Cover/*.jpg')):
            dataset.append({
                'kind': kind,
                'image_name': path.split('/')[-1],
                'label': label
            })
    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)
    gkf = GroupKFold(n_splits=5) # Split into 5 sets, validate on set 0, train on sets 1-4
    dataset.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
        dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
    fold_number = 0
    train_dataset = DatasetRetriever(
        options.datapath,
        kinds=dataset[dataset['fold'] != fold_number].kind.values,
        image_names=dataset[dataset['fold'] != fold_number].image_name.values,
        labels=dataset[dataset['fold'] != fold_number].label.values,
        transforms=get_train_transforms(),
    )
    validation_dataset = DatasetRetriever(
        options.datapath,
        kinds=dataset[dataset['fold'] == fold_number].kind.values,
        image_names=dataset[dataset['fold'] == fold_number].image_name.values,
        labels=dataset[dataset['fold'] == fold_number].label.values,
        transforms=get_valid_transforms(),
    )

    # TRAINING
    net = get_net().cuda()
    device = torch.device('cuda:0')
    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    if options.checkpoint is not None:
        # checkpoint = torch.load('../input/alaska2-public-baseline/best-checkpoint-033epoch.bin')
        # net.load_state_dict(checkpoint['model_state_dict'])
        print("Resuming from checkpoint {}".format(options.checkpoint))
        fitter.load(options.checkpoint)
    if not options.skip_training:
        run_training(fitter)

    # TEST
    net.eval() # Switch model to evaluation/inference mode
    dataset = DatasetSubmissionRetriever(
        options.datapath,
        image_names=np.array([path.split('/')[-1] for path in glob(os.path.join(options.datapath, 'Test/*.jpg'))]),
        transforms=get_valid_transforms(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    result = {'Id': [], 'Label': []}
    for step, (image_names, images) in enumerate(data_loader):
        print(step, end='\r')
        
        y_pred = net(images.cuda()) # 4-class classifcation: [5, 2, 4, 12]
        one_hot_classification = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy() # [0.2, 0.05, 0.015, 0.8]
        prob_is_cover = one_hot_classification[:,0]
        y_pred = 1 - prob_is_cover
        
        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    submission = pd.DataFrame(result)
    submission.to_csv('submission.csv', index=False)
    submission.head()
    # submission['Label'].hist(bins=100);
