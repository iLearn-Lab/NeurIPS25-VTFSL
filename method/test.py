import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--dataset', type=str, default='CIFAR-FS')
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=2000)
parser.add_argument('--feat-size', type=int, default=384)
parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='visformer')
parser.add_argument('--aug_support', type=int, default=1)
parser.add_argument('--stage', type=float, default=3)
parser.add_argument('--num_workers', type=int, default=16, choices=[16,8,4,2,1])
parser.add_argument('--gpu', default='1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from data.dataloader import EpisodeSampler
from model.visformer import visformer_tiny
import utils

def get_grouped_few_shot_images(val_dataset, target_classes, shot):
    all_labels = torch.tensor(val_dataset.targets)
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(all_labels):
        if label.item() in target_classes.tolist():
            class_to_indices[label.item()].append(idx)

    for c in target_classes:
        assert len(class_to_indices[c.item()]) >= shot, f"class {c.item()} lack {shot} samples"

    sampled_per_class = {}
    for c in target_classes:
        inds = np.random.choice(class_to_indices[c.item()], size=shot, replace=False)
        sampled_per_class[c.item()] = list(inds)

    grouped_indices = []
    grouped_labels = []
    for c in target_classes:
        grouped_indices.extend(sampled_per_class[c.item()])
        grouped_labels.extend([c.item()] * shot)

    final_dataset = Subset(val_dataset, grouped_indices)
    final_loader = DataLoader(final_dataset, batch_size=len(grouped_indices), shuffle=False)
    images, _ = next(iter(final_loader))
    labels = torch.tensor(grouped_labels)

    return images, labels


def main(config):
    # seed
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    svname = args.name+'{}-shot'.format(args.shot) 
    args.work_dir = os.path.join('./save', args.dataset, svname)
    utils.set_log_path(args.work_dir)

    # Dataset
    if args.dataset in ['miniImageNet', 'tieredImageNet']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CIFAR-FS', 'FC100']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224_cifar)
    elif args.dataset in ['FG-CUB', 'FG-Cars','FG-Dogs']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CD-Cars','CropDiseases', 'EuroSAT', 'ISIC', 'ChestX','Places', 'Plantae']:
        args.train = f'/dataset/{args.dataset}/base'
        args.val = f'/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    # 加载缓存
    generated_cache = torch.load('data/testing/test_{}_{}shot.pt'.format(args.dataset,str(args.shot)))  # class_id -> [shot, 3, 224, 224]

    if args.aug_support == 1:
        utils.log('fs dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),len(val_dataset.classes)),'test')
    else:
        utils.log('fs dataset: {} (x{}), {}'.format(val_dataset[0][0][0].shape, len(val_dataset),len(val_dataset.classes)),'test')

    # Dataloader
    val_sampler = EpisodeSampler(val_dataset.targets, args.episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,num_workers=args.num_workers, pin_memory=True)


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
    # init = args.work_dir+'/'+'best.pth'
    init = f'checkpoint/{args.dataset}-{args.shot}-shot.pth'
    checkpoint = torch.load(init,map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    utils.log('num params: {}'.format(utils.compute_n_params(model)),'test')
    model = model.to(device)

    # test 
    model.eval()
    va_lst = []
    A_acc = []
    f = checkpoint['k']
    
    with torch.no_grad():
        for episode in tqdm(val_loader,desc='fs-' + str(args.shot), leave=False):
            if args.aug_support ==1:
                image = episode[0].to(device)
                glabels = episode[1].to(device)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, args.query).view(-1).to(device)

                generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
                generated_labels = generated_labels.contiguous().view(-1)
                support_list = [generated_cache[int(class_id)] for class_id in generated_labels]
                generated_support = torch.cat(support_list, dim=0).to(device)

                image = image.view(args.way, args.shot+args.query, *image.shape[1:])
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
                va_lst.append(acc)

                com_proto = f * support + (1 - f) * gen_support
                logits = F.normalize(query, dim=-1) @ F.normalize(com_proto, dim=-1).t()
                acc = utils.compute_acc(logits, labels)
                A_acc.append(acc)

    
    va_lst = utils.count_95acc(np.array(va_lst))
    A_acc = utils.count_95acc(np.array(A_acc))
    log_str = 'test epoch : acc = {:.2f} +- {:.2f} (%)'.format(va_lst[0] * 100, va_lst[1] * 100)
    log_str += ' | {:.2f} +- {:.2f} (%,k = {:.2f})'.format(A_acc[0] * 100, A_acc[1] * 100, f)
    utils.log(log_str,'test')

if __name__ == '__main__':

    main(args)

