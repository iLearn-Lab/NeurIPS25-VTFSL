import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', default='') 
parser.add_argument('--dataset', type=str, default='miniImageNet') 
parser.add_argument('--augment', default='crop_aug')    
parser.add_argument('--repeat_aug', action='store_false')
parser.add_argument('--mixup_active', action='store_false')
parser.add_argument('--mixup', type=float, default=0.3)
parser.add_argument('--cutmix', type=float, default=1.0)
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
parser.add_argument('--mixup-prob', type=float, default=1.0)
parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
parser.add_argument('--mixup-mode', type=str, default='batch')
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default='8', choices=[16,8,4,2,1])
parser.add_argument('--image_size', type=int,default=224) 
parser.add_argument('--ef_epoch', type=int,default=50)
parser.add_argument('--episode', type=int,default=600)
parser.add_argument('--backbone', type=str, default='visformer-t')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--epoch', type=int,default=800, choices=[800,300]) 
parser.add_argument('--resume', type=str, default='') 
parser.add_argument('--gpu', default='0') 
args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from timm.scheduler import CosineLRScheduler
from timm.optim import AdamW
from timm.data import Mixup

from data.datasets import Datasets
from data.dataloader import EpisodeSampler, RepeatSampler
from model import visformer
import utils


def main(args):
    utils.set_seed(args.seed)
    svname = args.name      
    if svname is None:
        svname = '{}'.format(args.backbone)
    save_path = os.path.join('./checkpoint', args.dataset, svname)
    # utils.ensure_path(save_path)       
    utils.set_log_path(save_path)
    
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard')) 
    utils.log('set gpu: {} '.format(args.gpu)) 
    utils.log(vars(args)) 
    
    # dataloader
    # train_loader
    args_dict = vars(args)
    train_dataset = Datasets(args.dataset, split='train', **args_dict)
    if args.repeat_aug:
        repeat_sampler = RepeatSampler(train_dataset, batch_size=args.batch_size, repeat=2)
        train_loader = DataLoader(train_dataset, batch_sampler=repeat_sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset),len(train_dataset.dataset.classes)))
    utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    num_classes = len(train_dataset.dataset.classes)

    # val_loader
    eval_fs = True #False
    ef_epoch = args.ef_epoch
    n_way = 5
    n_shots = [1, 5]
    fs_loaders = []
    fs_dataset=Datasets(args.dataset,split='test',**args_dict)
    for n_shot in n_shots:
        episode_sampler = EpisodeSampler(fs_dataset.dataset.targets, 
                                        args.episode, 
                                        n_way, 
                                        n_shot + 15)
        fs_loader = DataLoader(fs_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers)
        fs_loaders.append(fs_loader)
    utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0].shape, len(fs_dataset),len(fs_dataset.dataset.classes)))
    utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

    mixup_fn = None
    

    # Model # 
    model = visformer.visformer_tiny(num_classes=num_classes)
    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    model=model.cuda()

    # Optimizer #
    optimizer = AdamW(model.parameters(), 
                  betas=(0.9, 0.999), 
                  eps=1.e-8, 
                  lr=args.lr, 
                  weight_decay=5.e-2)
    # lr_scheduler=None
    lr_scheduler = CosineLRScheduler(optimizer, 
                                    warmup_lr_init=1.e-6, 
                                    t_initial=args.epoch, 
                                    k_decay=0.1, 
                                    warmup_t=5)
    
    # Other parameters #
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')
    else:
        start_epoch = 1
    max_epoch = args.epoch
    save_epoch = 100
    max_va1 = 0.
    max_va5 = 0.
    timer_used = utils.Timer() 
    timer_epoch = utils.Timer()
    
    
    # Train and Val #
    for epoch in range(start_epoch, max_epoch+1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta']
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]
        aves = {k: utils.Averager() for k in aves_keys}
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train #
        model.train()
        if args.mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=num_classes)
            for data, label in train_loader:    
                data, label = data.cuda(), label.cuda()
                # Mixup
                data, label = mixup_fn(data, label)

                logits,_= model(data) 
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc_mix(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)
                # logits = None; loss = None
        else:
            for data, label in train_loader:    
                data, label = data.cuda(), label.cuda()

                logits,_= model(data) 
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)
            
        
        # Eval #
        if eval_fs and (epoch % ef_epoch == 0 or epoch==1):
            model.eval()
            for i,n_shot in enumerate(n_shots):
                for episode in fs_loaders[i]:
                    image = episode[0].cuda()  # way * (shot+15)   
                    labels = torch.arange(n_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda() 
                    with torch.no_grad():
                        _, im_features = model(image)

                        im_features = im_features.view(n_way, n_shot + 15, -1) 
                        sup_im_features, que_im_features = im_features[:, :n_shot], im_features[:, n_shot:]
                        sup_im_features = sup_im_features.mean(dim=1)
                        que_im_features = que_im_features.contiguous().view(n_way * 15, -1)
                        
                        logits = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                        acc = utils.compute_acc(logits, labels)

                    aves['fsa-' + str(n_shot)].add(acc)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1)
        
        # key's value to item() 
        for k, v in aves.items():
            aves[k] = v.item()

        # time of a epoch ,sum epochs and max epochs
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        # log of train loss and train acc
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        # log of val acc         
        if eval_fs and ((epoch % ef_epoch == 0 or epoch==1)):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        # svaed checkpoint
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(checkpoint,os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        if eval_fs and aves['fsa-' + str(1)] > max_va1:
            max_va1 = aves['fsa-' + '1']
            torch.save(checkpoint, os.path.join(save_path, 'best-{}shot.pth'.format(1)))
        if eval_fs and aves['fsa-' + str(5)] > max_va5:
            max_va5 = aves['fsa-' + '5']
            torch.save(checkpoint, os.path.join(save_path, 'best-{}shot.pth'.format(5)))
        if epoch==299 or epoch ==799:
            log_str='best 1-shot: {} and best 5-shot: {}'.format(max_va1,max_va5)
            
           
        writer.flush()

if __name__ == '__main__':
    

    main(args)


