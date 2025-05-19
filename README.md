# VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning


# Method

We propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment mechanism.

1. **VT-FSL Intuition**

<img src='img/fig1.png'>

2. **VT-FSL Overview**

<img src='img/fig2.png'>

## Standard Few-Shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| FC100 |    57.99 ± 0.40     |    67.68 ± 0.38    |
|  CIFAR-FS  |     88.67 ± 0.33     |     91.45 ± 0.46    |
| MiniImageNet |    83.66 ± 0.31     |     88.38 ± 0.25    |
| TieredImageNet |      88.02 ± 0.34     |     91.97 ± 0.27    |

## Fine-grained Few-Shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| CUB |    91.08 ± 0.28     |    94.63 ± 0.19    |
|  Dogs  |     86.58 ± 0.30     |     90.69 ± 0.25     |
| Cars |    92.95 ± 0.24     |     96.62 ± 0.15    |

## Cross Domain Few-shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| CUB      |    66.86 ± 0.47     |    81.02 ± 0.36     |
|  Places  |    73.68 ± 0.41     |    81.52 ± 0.33    |
|  Plantae |    45.90 ± 0.40     |    61.54 ± 0.38    |


# Requirements

- [PyTorch and torchvision](https://pytorch.org)

- Install packages:
```python
pip instal -r requirements.txt
```


# Datasets

- Dataset: [Google Cloud](https://drive.google.com/drive/folders/1xmXqS5AZAJpifyg2uyTnTXgWXeK9HthG?usp=drive_link) 
- Please download the dataset you need and then put the xxx.tar.gz in ./dataset directory:
```python
cd ./dataset
tar -xvzf xxx.tar.gz
```


# Reproducing Results with Pretrained Checkpoints
To directly reproduce the results reported in the paper using our trained models:

1. Download Weights

- Pre-training and Meta-tuning weights: [Google Cloud](https://drive.google.com/drive/folders/1qvXTPBmiw3Rb4Aj2uopvYKri4fm7-X2I?usp=sharing)
- Please download checkpoints and then put them into ./checkpoints directory.

2. Run Inference
Run the evaluation script with the desired settings. For example, to evaluate on the miniImageNet dataset with a 5-way 5-shot configuration:
```python
python method/test.py \
    --dataset miniImageNet \
    --way 5 \
    --shot 5 \
    --episode 2000 \
    --image_size 224 \
    --gpu 0

```
This will evaluate the pretrained model on 2000 few-shot episodes using the specified configuration. 

3. Expected Results
You should observe performance consistent with the results reported in our paper. If results slightly vary, it may be due to sampling randomness; we recommend running with a fixed seed or averaging over multiple runs.



# Training from Scratch
If you prefer to train the model from scratch instead of using the provided pretrained weights, follow the two-stage training process described below. We provide example scripts for the miniImageNet dataset.

1. Pre-train the Feature Extractor
Run the following command to pre-train the visual backbone on the base split of miniImageNet:
```python
python method/pretrain.py \
    --dataset miniImageNet \
    --batch_size 512 \
    --image_size 224 \
    --backbone visformer-t \
    --lr 5e-4 \
    --epoch 800 \
    --gpu 0

```

2. Meta-tune with VT-FSL
After pretraining, meta-tune the model for few-shot learning using the episodic training strategy.
- For 5-way 1-shot setting:
```python
python method/train.py \
    --dataset miniImageNet \
    --way 5 \
    --shot 1 \
    --image_size 224 \
    --backbone visformer-t \
    --lr 5e-4 \
    --epoch 100 \
    --t 0.2 \
    --gpu 0

```

- For 5-way 5-shot setting:
```python
python method/train.py \
    --dataset miniImageNet \
    --way 5 \
    --shot 5 \
    --image_size 224 \
    --backbone visformer-t \
    --lr 5e-4 \
    --epoch 100 \
    --t 0.2 \
    --gpu 0

```
After training, the model checkpoints will be automatically saved for evaluation.














