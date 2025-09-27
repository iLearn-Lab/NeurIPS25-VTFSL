import math
import os
import random
import warnings
import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False

transform_train = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])

transform_val = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])

resize = 224
transform_train_224_cifar = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_val_224_cifar = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_train_224 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

transform_val_224 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])

Normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train_cifar = transforms.Compose([
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Normalize
])

transform_val_cifar = transforms.Compose([
    transforms.ToTensor(),
    Normalize
])

#FG dataset
mean = ([x / 255.0 for x in [125.3, 123.0, 113.9]])
std = ([x / 255.0 for x in [63.0, 62.1, 66.7]])
norm_params = {"mean": mean, 
            "std": std}
norm = transforms.Normalize(**norm_params)

transform_train_fg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm  
])

transform_val_fg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm
])

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_acc_mix(logits, label, reduction='mean'):
    if len(label.shape) > 1:
        label = torch.argmax(label, dim=1)
    
    correct_predictions = (torch.argmax(logits, dim=1) == label).float()
    
    if reduction == 'mean':
        return correct_predictions.mean().item()
    elif reduction == 'sum':
        return correct_predictions.sum().item()
    else:
        return correct_predictions

def count_95acc(accuracies):
    acc_avg = np.mean(np.array(accuracies))
    acc_ci95 = 1.96 * np.std(np.array(accuracies)) / np.sqrt(len(accuracies))
    return acc_avg, acc_ci95

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        raise FileExistsError(f"Error: The path '{path}' already exists. Please provide a different path.")
    else:
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    


def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='train'):
    filename += '.txt'
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)

def cluster(data, n_clusters=64, num=600):
    x = []
    label = []
    for k, v in data.items():
        x.extend(v)
        label.append(k)
    data = 0
    y = np.arange(len(label)).repeat(len(v))
    x = np.array(x)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x, y)
    k_center = kmeans.cluster_centers_
    k_label = kmeans.labels_
    center = {}
    for k in range(len(label)):
        labels = k_label[k * num:(k + 1) * num]
        counts = np.bincount(labels)
        index = np.argmax(counts)
        center[label[k]] = k_center[index]
    return center


class MultiTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        out = []
        for trans in self.trans:
            out.append(trans(x))
        return out

transform = transforms.Compose([
                    transforms.Resize(int(224 * 1.1)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform = MultiTrans([transform]*3 + [aug]*(10-3))


def convert_raw(dataset,x):
    if dataset == 'miniImageNet':
        norm_params = {"mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]}
    else:
        norm_params = {"mean": [0.5071, 0.4866, 0.4409],
                        "std": [0.2009, 0.1984, 0.2023]}

    mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
    std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
    return x * std + mean

def visualize_dataset(dataset, name, writer, n_samples=16, step=0):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(convert_raw(dataset,dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo), global_step=step)
    writer.flush()

transform_cam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
    # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))  # Differs from ImageNet standard!
])
