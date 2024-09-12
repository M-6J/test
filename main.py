import argparse
import os
import torch.nn.parallel
import torch.optim
from models.DTA_SNN import *
from data.augmentations import rand_bbox
from data.loaders import build_cifar
from utils.utils import *
import numpy as np
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Dual Temporal-channel-wise Attention for Spiking Neural Networks')


parser.add_argument('--DTA_ON',
                    default=True,
                    type=bool,
                    help='using DTA')
parser.add_argument('--DS',
                    default='',
                    type=str,
                    help='cifar10, cifar100')

parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number')

parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    metavar='N')

parser.add_argument('--learning_rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training')

parser.add_argument('--time_step',
                    default=6,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 6)')
parser.add_argument('--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs',
                    default=250,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
args = parser.parse_args()



def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    r = np.random.rand(1)

    for  i,(images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        labels = labels.to(device)
        images = images.to(device)
        if args.beta > 0 and r < args.cutmix_prob:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, target_a) * lam + criterion(mean_out,target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':

    seed_all(args.seed)

    if args.DS == 'cifar10':
        num_CLS = 10
        save_ds_name = 'CIFAR10'
        train_dataset, val_dataset = build_cifar(use_cifar10=True)
    
    elif args.DS == 'cifar100': 
        num_CLS = 100
        save_ds_name = 'CIFAR100'
        train_dataset, val_dataset = build_cifar(use_cifar10=False)
        
    if args.DS == 'cifar10' or 'cifar100':
        DP_model = dta_msresnet18(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA_ON) #### 
        DP_model = torch.nn.DataParallel(DP_model).to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(DP_model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=5e-5)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    
    logger = get_logger(f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}.log')
    logger.info('start training!')

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(DP_model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch +1, args.epochs, loss, acc ))
        scheduler.step()
        facc = test(DP_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch+1, args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(DP_model.module.state_dict(), f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}.pth.tar')

        logger.info('Best Test acc={:.3f}'.format(best_acc ))
        logger.info('\n')

