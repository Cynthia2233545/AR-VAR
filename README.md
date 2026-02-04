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
### Data Preparation
Our model works with paired Non-Contrast CT (NCCT) and Contrast-Enhanced CT (CECT) images.
### 1. Dataset Structure
Please organize your data as follows. We recommend converting 3D NIfTI volumes into 2D slices (e.g., .png or .npy) for training efficiency, or use our provided dataloader for on-the-fly slicing.
## Train AR-VAR
```bash
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --data_path=/path/to/imagenet --condition_path=/path/to/condition/extract/above \
  --vae_ckpt=/path/to/pretrained/vae/ckpt --pretrained_var_ckpt=/path/to/pretrained/var/ckpt \
  --tblr=0.0001 --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 
 ```
## Inference
```bash
python inference.py --vae_ckpt=/path/to/pretrained/vae/ckpt --var_ckpt=/path/to/pretrained/var/ckpt \
  --car_ckpt=/path/to/car/ckpt --img_path=/path/to/original/image/to/extract/condition \
  --save_path=/path/to/save/image --cls=3 --type=hed
 ```
