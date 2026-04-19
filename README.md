
<a id="top"></a>
<div align="center">
    <div>
  <h1 align="center">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</h1>
  Wenhao&#160;Li<sup>1, 2</sup>,
  Qiangchang&#160;Wang<sup>1*</sup>,
  Xianjing&#160;Meng<sup>1</sup>,
  Zhibin&#160;Wu<sup>1</sup>,
  Yilong&#160;Yin<sup>1*</sup>
  
  </div>
  <sup>1</sup>School of Software, Shandong University &#160&#160&#160</span> <sup>2</sup>Shenzhen Loop Area Institute&#160&#160&#160</span>
  <br />
  <sup>*</sup>Corresponding author&#160;&#160;</span>
  <br/>
  <p>
  <a href="https://openreview.net/pdf/521c2d2f24d89aa54641a2bb0276e4fd1a5cebf4.pdf"><img src="https://img.shields.io/badge/NeurIPS-2025-blue.svg?style=flat-square" alt="NeurIPS 2025"></a>
  <a href="https://arxiv.org/abs/2509.25033"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2025-b31b1b.svg?style=flat-square"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/Platform-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <img src="https://img.shields.io/badge/Language-Python-green.svg?style=flat-square" alt="Language">
  <img src="https://img.shields.io/npm/l/express.svg?style=flat-square" alt="License">
  <a href="https://peacelwh.github.io/"><img src="https://img.shields.io/badge/Author_Page-blue.svg?style=flat-square" alt="Author Page"></a>
</p>
 
</div>



## ***NeurIPS 2025***

<!--
# [VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning]()
![](https://img.shields.io/badge/Platform-PyTorch-blue)
![](https://img.shields.io/badge/Language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)
-->

This repo is the official code implementation of the NeurIPS 2025 paper "VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning". [[arXiv]](https://arxiv.org/abs/2509.25033) [[paper]](https://openreview.net/pdf/521c2d2f24d89aa54641a2bb0276e4fd1a5cebf4.pdf)

We propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment mechanism.

## **Poster**

<img src='img.png'>

### Standard Few-Shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| MiniImageNet |    83.66 ± 0.31     |     88.38 ± 0.25    |
|  CIFAR-FS  |     88.67 ± 0.33     |     91.45 ± 0.46    |
| TieredImageNet |      88.02 ± 0.34     |     91.97 ± 0.27    |
| FC100 |    57.99 ± 0.40     |    67.68 ± 0.38    |

### Fine-grained Few-Shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| CUB |    91.08 ± 0.28     |    94.63 ± 0.19    |
|  Dogs  |     86.58 ± 0.30     |     90.69 ± 0.25     |
| Cars |    92.95 ± 0.24     |     96.62 ± 0.15    |

### Cross Domain Few-shot Classification Results
|  Dataset  | 1-Shot 5-Way | 5-Shot 5-Way |  
|:--------:|:------------:|:------------:|
| CUB      |    66.86 ± 0.47     |    81.02 ± 0.36     |
|  Places  |    73.68 ± 0.41     |    81.52 ± 0.33    |
|  Plantae |    45.90 ± 0.40     |    61.54 ± 0.38    |


## Usage
### Requirements

- [PyTorch and torchvision](https://pytorch.org)

- Install packages:
```python
pip instal -r requirements.txt
```


### Datasets

- Download link: [Hugging Face](https://huggingface.co/datasets/iLearn-Lab/NeurIPS25-VTFS)  or [Google Cloud](https://drive.google.com/drive/folders/1xmXqS5AZAJpifyg2uyTnTXgWXeK9HthG?usp=drive_link) 
- Please download the dataset you need and then put the xxx.tar.gz in ./dataset directory:
```python
cd ./dataset
tar -xvzf xxx.tar.gz
```

### Synthetic Images
- Download link: [Hugging Face](https://huggingface.co/datasets/iLearn-Lab/NeurIPS25-VTFS)  or [Google Cloud](https://drive.google.com/drive/folders/18hCmtaQ-JDCCpcnwSlQSyZitt1Ja2rVy) 
- Please download the directory you need and then put them in ./data directory:


### Reproducing Results with Pretrained Checkpoints
To directly reproduce the results reported in the paper using our trained models:

1. Pre-training and meta-tuning checkpoints

- Download link: [Hugging Face](https://huggingface.co/datasets/iLearn-Lab/NeurIPS25-VTFS)  or [Google Cloud](https://drive.google.com/drive/folders/1qvXTPBmiw3Rb4Aj2uopvYKri4fm7-X2I) 
- Please download the checkpoints you need and then put them into ./checkpoints directory.

2. Run Inference
Run the evaluation script with the desired settings. For example, to evaluate on the miniImageNet dataset with a 5-way 5-shot configuration:
```python
python test.py \
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

### Coming Soon
We are actively working on releasing **more pretrained weights** across additional datasets (e.g., tieredImageNet, CUB, FC100), as well as the **generated class descriptions and synthetic images** used in VT-FSL. These resources will be made publicly available to further support reproducibility and research on multimodal few-shot learning.
Stay tuned for updates!



### Training from Scratch
If you prefer to train the model from scratch instead of using the provided pretrained weights, follow the two-stage training process described below. We provide example scripts for the miniImageNet dataset.

1. Pre-train the Feature Extractor
Run the following command to pre-train the visual backbone on the base split of miniImageNet:
```python
python pretrain.py \
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
python train.py \
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
python train.py \
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


## Citation
If you find this repo helpful in your research or work, please cite the following paper.
```
@article{li2025vt,
  title={VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning},
  author={Li, Wenhao and Wang, Qiangchang and Meng, Xianjing and Wu, Zhibin and Yin, Yilong},
  journal={arXiv preprint arXiv:2509.25033},
  year={2025}
}
```











