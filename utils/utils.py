import logging
import os 
import random 
import numpy as np 
import torch 
import math 
import torchvision.datasets as datasets
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data.distributed import DistributedSampler

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def seed_all(seed=1004):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
    

#for ImageNet
class Transform():
     def __init__(self,transform):
        self.transform=transform
     def __call__(self,image):
        return self.transform(image=image)["image"]
     
def open_img(img_path):
     img=Image.open(img_path).convert('RGB')
     return np.array(img)

def get_training_dataloader(traindir, sampler=None, batch_size=16, num_workers=2, shuffle=True, args=None, rank=None):

    train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])

    ImageNet_training = datasets.ImageFolder(traindir, transform=Transform(train_transforms), loader=open_img)


    ImageNet_train_sampler = DistributedSampler(dataset=ImageNet_training, seed=args.seed, num_replicas=args.world_size, rank=rank)
    
    ImageNet_training_loader = DataLoader(
        ImageNet_training,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        sampler=ImageNet_train_sampler
        )
   
    return ImageNet_training_loader, ImageNet_train_sampler


def get_test_dataloader(valdir, sampler=None, batch_size=16, num_workers=2, shuffle=True, args=None, rank=None):
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    ImageNet_test = datasets.ImageFolder(
        valdir,
        transform=Transform(val_transforms),
        loader=open_img
    )
    
    ImageNet_test_loader = DataLoader(
        ImageNet_test,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
    )
    
    return ImageNet_test_loader
