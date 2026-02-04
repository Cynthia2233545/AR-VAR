# AR-VAR: Anatomy-Aware Visual Autoregressive Models for Virtual Tumor CT Contrast Enhancement
## Installation
### 1. Clone the Repository
First, clone the official code repository to your local machine via the following Git command:
```bash
git clone [https://github.com/YourUsername/AR-VAR](https://github.com/Cynthia2233545/AR-VAR).git
cd AR-VAR
```

### 2. Create a Virtual Environment
We strongly recommend using Anaconda to create an isolated virtual environment, which can avoid dependency conflicts with other projects.
```bash
conda create -n arvar python=3.10
conda activate arvar
```
### 3. Install dependencies
We strongly recommend using Anaconda to create an isolated virtual environment, which can avoid dependency conflicts with other projects.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## 3. Data Preparation
Our model requires paired 2D slices in `.npy` format. **The filenames of the condition (NCCT) and target (CECT) must correspond exactly to each other** (one-to-one mapping is required for successful training).

### 3.1 Required Directory Structure
Please organize your dataset in the following directory structure strictly (the root directory is the `AR-VAR` project folder):
```bash
AR-VAR/
└── datasets/
├── train_condition/ # Place Non-Contrast CT (NCCT) slices here (input conditions)
│ ├── case001_slice001.npy
│ ├── case001_slice002.npy
│ └── ...
└── train/class_0 # Place Contrast-Enhanced CT (CECT) slices here (training targets)
├── case001_slice001.npy <-- Filename must match corresponding NCCT slice
├── case001_slice002.npy
└── ...
```
## 4. Model Zoo & Pre-trained Weights
AR-VAR is built upon the **VAR (Visual Autoregressive Modeling)** framework. For faster model convergence and better performance (especially for low-data scenarios), **you can optionally use the official pre-trained VAR weights** to initialize the base model.

### 1. Pre-trained Weight Details
Please download the required pre-trained weights from the official VAR HuggingFace Repository.

| Component | File Name | Description | Download Link |
| :-------- | :-------- | :---------- | :------------ |
| VAE | `vae_ch160v4096z32.pth` | The VQ-VAE model for image tokenization (core component for feature encoding) | [Official VAR VAE Weights](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) |
| VAR Model | `var_d16.pth` | Pre-trained VAR model with depth-16 architecture (base autoregressive model) | [Official VAR Depth-16 Weights](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) |


## Train AR-VAR
```bash
torchrun --nproc_per_node=7 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --data_path=/path/to/imagenet --condition_path=/path/to/condition/extract/above \
  --vae_ckpt=/path/to/pretrained/vae/ckpt --pretrained_var_ckpt=/path/to/pretrained/var/ckpt \
  --tblr=0.0001 --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 
 ```
## Inference
```bash
python inference.py --vae_ckpt=/path/to/pretrained/vae/ckpt --var_ckpt=/path/to/pretrained/var/ckpt \
  --car_ckpt=/path/to/car/ckpt --img_path=/path/to/original/image/to/extract/condition \
  --save_path=/path/to/save/image --cls=3 --type=ct
 ```
