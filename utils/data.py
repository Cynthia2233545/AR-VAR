import os

import PIL.Image as PImage
import numpy as np
from torchvision.transforms import InterpolationMode, transforms

from utils.condition_datasets import ConditionDatasetFolder, IMG_EXTENSIONS


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, condition_path: str, pn: str, final_reso: int,
    hflip=False, mid_reso=1.125
):
    # build augmentations
    # 创建数据预处理函数
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
                             transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
                             # transforms.RandomCrop((final_reso, final_reso)),  # move to Dataset
                             # transforms.ToTensor(), normalize_01_into_pm1,     # move to Dataset
                         ], [
                             transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
                             # transforms.CenterCrop((final_reso, final_reso)),
                             # transforms.ToTensor(), normalize_01_into_pm1,     # move to Dataset
                         ]

    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

    # build dataset
    # 读取数据集
    train_set = ConditionDatasetFolder(root=os.path.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS,
                                       transform=train_aug, train=True, condition_path=condition_path, pn=pn)
    val_set = ConditionDatasetFolder(root=os.path.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS,
                                     transform=val_aug, train=False, condition_path=condition_path, pn=pn)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    return num_classes, train_set, val_set


def pil_loader(path):
    # with open(path, 'rb') as f:
    #     img: PImage.Image = PImage.open(f).convert('RGB')
    # print("-----------------------------------------------------------------",path)
    img = np.load(path).astype(np.float32)

    # img = (img - img.min()) / (img.max() - img.min())
    start_x = (img.shape[0] - 256) // 2
    start_y = (img.shape[1] - 256) // 2

    # # 截取 256x256
    # img = img[start_x:start_x + 256, start_y:start_y + 256]
    # 使用 expand_dims 增加一个维度，使其成为 (1, 512, 512)
    img_cropped = np.expand_dims(img, axis=2)
    # 沿着通道维度重复3次，得到 (1, 3, 512, 512)
    image_stacked = np.repeat(img_cropped, 3, axis=2)
    # print("-----------------------------------------------------------------",image_stacked.shape)
    image_stacked = image_stacked.astype(np.uint8)  # 转换为 uint8 类型
    from PIL import Image
    # 使用 PIL 将 NumPy 数组转换为 Image 对象
    image_stacked = Image.fromarray(image_stacked)
    
    return image_stacked


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
