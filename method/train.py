import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--shot', type=int, default=5)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--dataset', type=str, default='CIFAR-FS')
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=600)
parser.add_argument('--feat-size', type=int, default=384)
parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='visformer')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr1', type=float, default=1e-6)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--epoch', type=int,default=100)
parser.add_argument('--stage', type=float, default=3)
parser.add_argument('--num_workers', type=int, default=8, choices=[16,8,4,2,1])
parser.add_argument('--t', type=float, default=0.2)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from timm.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


from data.dataloader import EpisodeSampler
from model.visformer import visformer_tiny
import utils

def kernel_fn(x, y, type='rbf', sigma=1.0, degree=3, coef0=1.0):
    if type == 'linear':
        return x @ y.T
    elif type == 'poly':
        return (x @ y.T + coef0) ** degree
    elif type == 'rbf':
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B1, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (B2, 1)
        dist = x_norm - 2 * x @ y.T + y_norm.T      # (B1, B2)
        return torch.exp(-dist / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel type: {type}")

def volume_computation_with_kernel(anchor, *inputs, kernel_type='rbf', sigma=1.0, degree=3, coef0=1.0):
    """
    Volume computation in kernel-induced feature space.

    Args:
    - anchor: Tensor of shape (B1, D)
    - *inputs: list of tensors of shape (B2, D)
    - kernel_type: 'linear', 'poly', or 'rbf'
    Returns:
    - volume matrix: (B1, B2)
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]
    n_modalities = 1 + len(inputs)

    # Compute kernel matrices
    aa = kernel_fn(anchor, anchor, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
    aa_diag = torch.diagonal(aa, dim1=0, dim2=1).unsqueeze(1).expand(-1, batch_size2)

    l_inputs = [kernel_fn(anchor, input, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
                for input in inputs]

    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot = kernel_fn(input1, input2, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
            dot_diag = torch.diagonal(dot, dim1=0, dim2=1).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_diag)
        input_dot_products.append(row)

    # Construct Gram matrix for each anchor pair
    G = torch.stack(
        [torch.stack([aa_diag] + l_inputs, dim=-1)] +
        [torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1)
         for i in range(len(inputs))],
        dim=-2
    )  # (B1, B2, n_modalities, n_modalities)

    # Compute determinant for each Gram matrix
    gram_det = torch.det(G.float())
    res = torch.sqrt(torch.clamp(gram_det, min=1e-8))  # numerical stability
    return res

def main(config):
    # seed
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    svname = args.name+'{}-shot'.format(args.shot) 
    args.work_dir = os.path.join('./save', args.dataset, svname)
    utils.ensure_path(args.work_dir) 
    utils.set_log_path(args.work_dir)
    utils.log(vars(args))
    
    writer = SummaryWriter(os.path.join(args.work_dir, 'tensorboard'))  

    # Dataset
    if args.dataset in ['miniImageNet', 'tieredImageNet']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CIFAR-FS', 'FC100']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224_cifar)
    elif args.dataset in ['FG-CUB', 'FG-Cars','FG-Dogs']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CD-CUB','Places', 'Plantae']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)

    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset),len(train_dataset.classes)))
    utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),len(val_dataset.classes)))

    generated_cache = torch.load('data/valid/val_{}_{}shot.pt'.format(args.dataset,str(args.shot)))  
    
    # Dataloader
    n_episodes = int(len(train_dataset) / (args.way * (args.shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.targets,n_episodes, args.way, args.shot + 15, fix_seed=False)
    train_loader = DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers, pin_memory=True)
    val_sampler = EpisodeSampler(val_dataset.targets, args.episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,num_workers=args.num_workers, pin_memory=True)

     # text
    val_idx_to_class = val_dataset.class_to_idx
    idx_to_class = train_dataset.class_to_idx
    if args.dataset == 'FG-CUB':
        val_idx_to_class = {k: v.split(".", 1)[-1] for v,k in val_idx_to_class.items()}
        idx_to_class = {k: v.split(".", 1)[-1] for v,k in idx_to_class.items()}
    elif args.dataset == 'FG-Dogs':
        val_idx_to_class = {k: v.split("-", 1)[-1] for v,k in val_idx_to_class.items()}
        idx_to_class = {k: v.split("-", 1)[-1] for v,k in idx_to_class.items()}
    else:
        val_idx_to_class = {k: v for v, k in val_idx_to_class.items()}
        idx_to_class = {k: v for v, k in idx_to_class.items()}


    semantic = torch.load('./semantic/{}_semantic_clip_cot.pth'.format(args.dataset))['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    #image
    generated_cache = torch.load('data/training/train_{}_{}shot.pt'.format(args.dataset,str(args.shot)))

    # backbone
    if args.backbone == 'visformer':
        model = visformer_tiny(num_classes=len(train_dataset.classes))
    else:
        raise ValueError(f'unknown model: {args.model}')
    
    # load
    if args.backbone == 'visformer':
        text_dim = args.semantic_size
        feature_dim = args.feat_size
        if 2 <= args.stage < 3:
            feature_dim = 192
        model.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
        model.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        model.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                                torch.nn.Sigmoid(),
                                                torch.nn.Linear(feature_dim, feature_dim),
                                                torch.nn.Sigmoid(),)
    
    # resume
    if args.resume:
        init = args.resume
    else:
        init = './checkpoint/visformer-{}.pth'.format(args.dataset)
        checkpoint = torch.load(init,map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model = model.to(device)
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # optimizer
    optim_params_id = [id(param) for param in model.t2i.parameters()]
    optim_params_id += [id(param) for param in model.t2i2.parameters()]
    optim_params_id += [id(model.log_tau)]


    optim_params = [param for param in model.parameters() if id(param) in optim_params_id]
    other_params = [param for param in model.parameters() if id(param) not in optim_params_id]
    
    # low lr of backbone
    optimizer = AdamW([{'params': optim_params, 'lr':args.lr, 'weight_decay': 1e-4},
                        {'params': other_params, 'lr': args.lr1}], weight_decay=1e-4)
    lr_scheduler=None

    # train
    save_epoch = 25
    max_va = 0.
    ef_epoch = 1
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    if args.resume:
        checkpoint = torch.load(args.resume,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')
    else:
         start_epoch = 1
    for epoch in range(start_epoch, args.epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        model.train()
        for episode in train_loader:
            image = episode[0].to(device)
            glabels = episode[1].to(device)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).to(device)

            generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
            generated_labels = generated_labels.contiguous().view(-1)
            support_list = [generated_cache[int(class_id)] for class_id in generated_labels]
            synthetic_support = torch.cat(support_list, dim=0).to(device)
            synthetic_support = synthetic_support.view(args.way * args.shot, 3, 224, 224)

            image = image.view(args.way, args.shot+15, *image.shape[1:])
            sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
            sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

            glabels = glabels.view(args.way, args.shot+15)[:, :args.shot]
            glabels = glabels.contiguous().view(-1)

            text_features = torch.stack([semantic[idx_to_class[l.item()]] for l in glabels]).to(device)
            _, gen_support = model.fusion(sup, text_features, args)
            _, syn_support = model(synthetic_support)
    
            text_features = model.t2i(text_features)
            temp = model.log_tau.exp()
            text_features_norm = F.normalize(text_features, dim=-1)
            gen_support_norm = F.normalize(gen_support, dim=-1)
            # volume = volume_computation_with_kernel(text_features_norm, gen_support_norm, kernel_type='rbf', sigma=0.5)
            syn_support_norm = F.normalize(syn_support, dim=-1)
            #  kernelized volume
            volume = volume_computation_with_kernel(
                text_features_norm,
                gen_support_norm,
                syn_support_norm,
                kernel_type='rbf',  
                sigma=0.5          
            )
            volume = volume / temp
            volumeT = volume.T / temp
            targets = torch.arange(volume.shape[0]).to(device)
            align_loss = (
                F.cross_entropy(-volume, targets, label_smoothing=0.2) +
                F.cross_entropy(-volumeT, targets, label_smoothing=0.2)
            ) / 2   


            gen_support = gen_support.view(args.way, args.shot, -1).mean(dim=1)
            _, query = model(que)
            logits = F.normalize(query, dim=-1) @ F.normalize(gen_support, dim=-1).t()
            loss_cls = F.cross_entropy(logits/ args.t, labels)

            loss = loss_cls + align_loss
            acc = utils.compute_acc(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
        
        # eval
        if epoch % ef_epoch == 0 or epoch== 1:
            ks = np.arange(0, 101) * 0.01
            P_acc = {}
            model.eval()
            with torch.no_grad():
                for episode in val_loader:
                    image = episode[0].to(device)
                    glabels = episode[1].to(device)
                    labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).to(device)

                    generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
                    generated_labels = generated_labels.contiguous().view(-1)
                    support_list = [generated_cache[int(class_id)] for class_id in generated_labels]
                    generated_support = torch.cat(support_list, dim=0).to(device)

                    image = image.view(args.way, args.shot+15, *image.shape[1:])
                    sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
                    sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                    glabels = glabels.view(args.way, args.shot+15)[:, :args.shot]
                    glabels = glabels.contiguous().view(-1)
                    text_features = torch.stack([semantic[val_idx_to_class[l.item()]] for l in glabels]).to(device)

                    sup_reshaped = sup.view(args.way, args.shot, 3, 224, 224)
                    gen_reshaped = generated_support.view(args.way, args.shot, 3, 224, 224)
                    merged = torch.cat([sup_reshaped, gen_reshaped], dim=1)
                    sup1 = merged.view(args.way * 2 * args.shot, 3, 224, 224)
                    _, support = model(sup1)
                    support = support.view(args.way, args.shot* 2, -1).mean(dim=1)

                    _, gen_support = model.fusion(sup, text_features, args)
                    gen_support = gen_support.view(args.way, args.shot, -1).mean(dim=1)
                    _, query = model(que)

                    logits = F.normalize(query, dim=-1) @ F.normalize(gen_support, dim=-1).t()
                    acc = utils.compute_acc(logits, labels)
                    aves['va'].add(acc)

                    for f in ks:
                        com_proto = f * support + (1 - f) * gen_support
                        logits = F.normalize(query, dim=-1) @ F.normalize(com_proto, dim=-1).t()
                        acc = utils.compute_acc(logits, labels)
                        if str(f) in P_acc:
                            P_acc[str(f)].append(acc)
                        else:
                            P_acc[str(f)] = []
                            P_acc[str(f)].append(acc)

        # post #
        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1)

        # key's value to item()
        for k, v in aves.items():
            aves[k] = v.item()
        
        # time of a epoch ,sum epochs and max epochs
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * args.epoch)

        # log train loss and acc  
        log_str = 'epoch {}, train {:.4f}|{:.2f}%'.format(epoch, aves['tl'], aves['ta'] * 100)
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)
        
        # log val acc 
        if epoch % ef_epoch == 0 or epoch==1:
            log_str += ', val {}: {:.4f}'.format(args.shot,aves['va'])
            writer.add_scalars('acc', {'val': aves['va']}, epoch)
            max_acc = {
                    'k': 0,
                    'acc': 0,
                }
            for k, v in P_acc.items():
                P_acc[k] = utils.count_95acc(np.array(v))
                if P_acc[k][0] > max_acc['acc']:
                    max_acc['acc'] = P_acc[k][0]
                    max_acc['k'] = k
            log_str += ', {:.4f} (k = {})'.format( max_acc['acc'], max_acc['k'])
        
        # save checkpoint
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'k': max_acc['k'],
            }
        
        if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(checkpoint,os.path.join(args.work_dir, 'epoch-{}.pth'.format(epoch)))
            
        if aves['va'] > max_va or max_acc['acc'] > max_va :
            if aves['va'] > max_acc['acc']:
                max_va = aves['va']
                torch.save(checkpoint, os.path.join(args.work_dir, 'best.pth'))
            else:
                max_va = max_acc['acc']
                torch.save(checkpoint, os.path.join(args.work_dir, 'best.pth'))
        
        
        log_str += '| MAX: {:.4f}'.format(max_va)
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        writer.flush()


if __name__ == '__main__':

    main(args)

