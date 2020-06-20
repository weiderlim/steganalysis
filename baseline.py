import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sklearn
import time
import torch

from albumentations.pytorch.transforms import ToTensorV2
from datetime import datetime
from glob import glob
from skimage import io
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSample

from dataset_retriever import DatasetRetriever
from dataset_submission_retriever import DatasetSubmissionRetriever
from fitter import Fitter
from train_global_config import TrainGlobalConfig

# RUN PARAMS

DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'

# SETUP

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# LOAD DATA

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

dataset = []
for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })
random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

# Split into 5 set, validate on set 0, train on sets 1-4
gkf = GroupKFold(n_splits=5)
dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
fold_number = 0

train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != fold_number].kind.values,
    image_names=dataset[dataset['fold'] != fold_number].image_name.values,
    labels=dataset[dataset['fold'] != fold_number].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == fold_number].kind.values,
    image_names=dataset[dataset['fold'] == fold_number].image_name.values,
    labels=dataset[dataset['fold'] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

image, target = train_dataset[0]
numpy_image = image.permute(1,2,0).cpu().numpy()
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))    
# ax.set_axis_off()
# ax.imshow(numpy_image);

import warnings
warnings.filterwarnings("ignore")

from efficientnet_pytorch import EfficientNet

def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net

net = get_net().cuda()

from catalyst.data.sampler import BalanceClassSampler

def run_training():
    device = torch.device('cuda:0')

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

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
#     fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)

# checkpoint = torch.load('../input/alaska2-public-baseline/best-checkpoint-033epoch.bin')
# net.load_state_dict(checkpoint['model_state_dict'])
run_training()
net.eval() # Switch model to evaluation/inference mode

# checkpoint.keys()

dataset = DatasetSubmissionRetriever(
    image_names=np.array([path.split('/')[-1] for path in glob('../input/alaska2-image-steganalysis/Test/*.jpg')]),
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