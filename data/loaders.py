import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import warnings
import torchvision.datasets as datasets
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data.distributed import DistributedSampler
import numpy as np

warnings.filterwarnings('ignore')


def build_cifar(use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())
    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./cifar_data',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='./cifar_data',
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./cifar_data',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='./cifar_data',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset






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
