import warnings
warnings.filterwarnings(action='ignore')
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
import torch.cuda.amp
from utils.utils import *
from data.loaders import *
from models.DTA_SNN import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main_worker(rank, args):
    seed_all(args.seed)
    dist.init_process_group('nccl', rank = rank, world_size=args.world_size)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    args.device = 'cuda'
    device = torch.device(args.device)
    
    if rank is not None:
        print("Use GPU: {} for training".format(rank))

    print('ImageNet')
    
    DDP_model = dta_msresnet_34(num_classes=1000, time_step=args.time_step, DTA_ON=args.DTA_ON)
    
    
    DDP_model.to(device)
    DDP_model = nn.SyncBatchNorm.convert_sync_batchnorm(DDP_model)
    DDP_model = DDP(DDP_model, device_ids=[rank], broadcast_buffers=False)
    
    if rank == 0:
        if args.resume: 
            logger = get_logger(f'resume-ImageNet-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}.log')
            logger.info('start training!')
        else:        
            logger = get_logger(f'ImageNet-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}.log')
            logger.info('start training!')
    else: logger = None
    
    best_acc = 0.5
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params = DDP_model.parameters(),lr = args.learning_rate ,momentum=0.9,weight_decay=1e-5)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    training_loader, train_sampler = get_training_dataloader(
        traindir="./ImageNet_dataset/train",
        num_workers=args.workers,
        batch_size=int(args.batch_size / args.world_size),
        shuffle=False,
        sampler=1, # to enable sampler for DDP
        args = args,
        rank = rank
    )
    
    test_loader = get_test_dataloader(
        valdir="./ImageNet_dataset/val",
        num_workers=args.workers,
        batch_size=int(args.batch_size / args.world_size),
        shuffle=False,
        args=args,
        rank=rank
    )
                                               
    scaler = torch.cuda.amp.GradScaler()
    
    if args.resume is not False:
        checkpoint = torch.load(args.pt_path, map_location=torch.device(rank))
        args.start_epoch = checkpoint['epoch'] + 1
        DDP_model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % int(args.start_epoch))
    
    for epoch in range(args.start_epoch, args.epochs):
        
        train_sampler.set_epoch(epoch)
        train(device, rank, DDP_model, training_loader, criterion, optimizer, args, scaler, logger, epoch)
        train_scheduler.step()
        
        if rank == 0:
            curr_acc = eval_training(device, test_loader, DDP_model, criterion, args, logger, epoch)
        
            if best_acc < curr_acc:
                best_acc = curr_acc
                checkpoint = {'epoch': epoch+1,
                          'model_state_dict': DDP_model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': train_scheduler.state_dict()}
                torch.save(DDP_model.module.state_dict(), f'ImageNet-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}.pth.tar')

            logger.info('Best Test acc={:.3f}'.format(best_acc ))
            logger.info('\n')
        else: 
            pass


def train(device, rank, model, training_loader, criterion, optimizer, args, scaler, logger, epoch):
    running_loss = 0

    model.train()
    correct = 0.0
    num_sample = 0
    
    for batch_index, (images, labels) in enumerate(training_loader):
        labels = labels.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        num_sample += images.size()[0]
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, labels)
            _, predicted = mean_out.cpu().max(1)
            correct += float(predicted.eq(labels.cpu()).sum().item())
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if rank == 0:
        acc = correct/num_sample* 100
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch+1, args.epochs, running_loss, acc))



@torch.no_grad()
def eval_training(device, test_loader, model, criterion, args, logger, epoch):

    model.eval()

    running_loss = 0.0
    correct = 0.0
    real_batch = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            real_batch += images.size()[0]

            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)

            outputs = model(images)
            mean_out = outputs.mean(1)
            
            loss = criterion(mean_out, labels)
            criterion(mean_out,labels)
            running_loss += loss.item()

            _, predicted = mean_out.cpu().max(1)
            correct += float(predicted.eq(labels.cpu()).sum().item())

    acc = correct / len(test_loader.dataset) * 100

    logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch+1, args.epochs, running_loss, acc))
    
    return acc

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DTA_ON',
                    default=True,
                    type=bool,
                    help='using DTA')
    
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number')

    parser.add_argument('--batch_size',
                        default=128,
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
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--world_size', 
                        default=4, 
                        type=int,
                        help='world size')
    
    parser.add_argument('--resume', 
                        default=False, 
                        type=bool,
                        help = 'O X pre-trained')
    
    parser.add_argument('--pt_path',
                        type=str,
                        help = 'resume path')

    args = parser.parse_args()

    mp.spawn(main_worker,nprocs=args.world_size,args=(args,),join=True)


    
    