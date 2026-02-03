import os
import os.path
import sys
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension."""
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension."""
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class)."""
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class ConditionDatasetFolder(VisionDataset):
    def __init__(
        self, 
        root: str, 
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train=False,
        condition_path=None,
        pn=None,
        unet_cache_dir: str = '/data/tangyingxin/CAR/unet_features',  # 修改为UNet特征缓存目录
    ) -> None:
        # 添加调试信息
        print(f"初始化 ConditionDatasetFolder - root: {root}, condition_path: {condition_path}, pn: {pn}")
        
        # 确保所有参数正确传递
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # 1. 查找类别
        classes, class_to_idx = self.find_classes(self.root)
        
        # 2. 生成样本列表
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        
        # 3. 初始化属性
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.train = train
        self.condition_path = condition_path
        var_pn = pn.split('_')
        self.pn = [int(i) for i in var_pn]
        
        # 4. 验证文件配对
        self.validate_file_pairs()
        
        # 5. 初始化UNet特征相关组件
        self.unet_cache_dir = unet_cache_dir  # 使用UNet特征缓存目录
        os.makedirs(unet_cache_dir, exist_ok=True)
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """修复 find_classes 方法实现"""
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """添加 make_dataset 方法实现"""
        return self._make_dataset(directory, class_to_idx, extensions, is_valid_file)

    @staticmethod
    def _make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """静态方法实现 make_dataset 逻辑"""
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
            
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
    
    def validate_file_pairs(self):
        """验证每个目标图像都有对应的条件图像"""
        missing_files = []
        valid_pairs = []
        
        for path, _ in self.samples:
            # 直接从目标图像路径提取文件名
            file_name = os.path.basename(path)
            # 构建条件图像的完整路径（train_condition/同名文件）
            condition_path = os.path.join(self.condition_path, file_name)
            
            if not os.path.exists(condition_path):
                missing_files.append((path, condition_path))
            else:
                valid_pairs.append((path, condition_path))
        
        if missing_files:
            error_msg = f"Found {len(missing_files)} missing condition files:\n"
            for i, (target, condition) in enumerate(missing_files[:5]):  # 显示最多5个缺失文件
                error_msg += f"  Sample {i+1}:\n    Target: {target}\n    Condition: {condition}\n"
            if len(missing_files) > 5:
                error_msg += f"... and {len(missing_files)-5} more files"
            raise FileNotFoundError(error_msg)
        
        print(f"✓ All {len(self.samples)} target images have matching condition images.")
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, control_tensors, target, unet_feature) where target is class_index of the target class.
        """
        # 记录当前索引
        self._current_idx = index

        def get_control_for_each_scale(control_image, scale):
            res = []
            for pn in scale:
                # 使用LANCZOS插值保留细节，增大分辨率
                res.append(control_image.resize((pn*32, pn*32), Image.LANCZOS))
            return res

        path, target = self.samples[index]
        # 直接从目标图像路径提取文件名
        file_name = os.path.basename(path)
        # 构建条件图像的完整路径（train_condition/同名文件）
        condition_path = os.path.join(self.condition_path, file_name)
        
        # 使用自定义loader加载图像
        sample = self.loader(path)
        condition_sample = self.loader(condition_path)  # 加载低剂量图像
        
        # 确保条件图像和目标图像进行相同的预处理和裁剪
        if self.transform:
            # 训练时应用相同的随机裁剪
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(
                    sample, output_size=(256, 256)
                )
                sample = F.crop(sample, i, j, h, w)
                condition_sample = F.crop(condition_sample, i, j, h, w)
            # 验证时使用中心裁剪
            else:
                sample = self.transform(sample)
                condition_sample = self.transform(condition_sample)
        
        # 生成控制图像
        control_images = get_control_for_each_scale(condition_sample, self.pn)

        if self.target_transform is not None:
            target = self.target_transform(target)

        post_trans = transforms.Compose([transforms.ToTensor(), normalize_01_into_pm1])
        sample_tensor = post_trans(sample)

        # ============== 使用UNet分割特征 ==============
        # 构建UNet特征缓存路径
        rel_path = os.path.relpath(path, self.root)
        cache_dir = os.path.join(self.unet_cache_dir, os.path.dirname(rel_path))
        os.makedirs(cache_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(path))[0]
        unet_cache_path = os.path.join(cache_dir, f"{file_name}.pt")
        
        # 从缓存加载UNet特征（假设缓存文件已存在）
        if not os.path.exists(unet_cache_path):
            raise FileNotFoundError(f"UNet特征缓存文件不存在: {unet_cache_path}")
        
        unet_feature = torch.load(unet_cache_path)
        
        # 仅保留原始控制特征
        control_tensors = []
        for ci, pn in zip(control_images, self.pn):
            base_control = post_trans(ci.resize((pn*16, pn*16)))
            control_tensors.append(base_control)
        
        # 返回UNet特征
        return sample_tensor, control_tensors, target, unet_feature
    
    def __len__(self) -> int:
        return len(self.samples)


class CT_DatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            train=False,
            condition_path=None,
            pn=None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.train = train
        self.condition_path = condition_path
        var_pn = pn.split('_')
        self.pn = [int(i) for i in var_pn]
        
        # 验证文件配对
        self.validate_file_pairs()

    def validate_file_pairs(self):
        """验证每个目标图像都有对应的条件图像并随机显示几组结果"""
        missing_files = []
        valid_pairs = []
        
        for path, _ in self.samples:
            parts = os.path.normpath(path).split(os.sep)
            condition_path = os.path.join(self.condition_path, parts[-3], parts[-2], parts[-1])
            if not os.path.exists(condition_path):
                missing_files.append((path, condition_path))
            else:
                valid_pairs.append((path, condition_path))
        
        if missing_files:
            error_msg = f"找到 {len(missing_files)} 个缺失的条件图像:\n"
            for target, condition in missing_files[:5]:  # 只显示前5个缺失的文件
                error_msg += f"目标: {target} 缺少条件: {condition}\n"
            if len(missing_files) > 5:
                error_msg += f"...以及其他 {len(missing_files)-5} 个文件"
            raise FileNotFoundError(error_msg)
        
        print(f"✓ 验证通过: 所有 {len(self.samples)} 个目标图像都有匹配的条件图像")
        
        # 随机显示5组匹配结果
        print("\n随机显示5组匹配的目标图像和条件图像：")
        import random
        random.shuffle(valid_pairs)
        for i, (target, condition) in enumerate(valid_pairs[:5]):
            print(f"[示例 {i+1}]")
            print(f"  目标: {target}")
            print(f"  条件: {condition}")
        print()

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class)."""
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset."""
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, control_tensors, target) where target is class_index of the target class.
        """

        def get_control_for_each_scale(control_image, scale):
            res = []
            for pn in scale:
                # 使用LANCZOS插值保留细节，增大分辨率
                res.append(control_image.resize((pn*32, pn*32), Image.LANCZOS))
            return res

        path, target = self.samples[index]
        parts = os.path.normpath(path).split(os.sep)
        condition_path = os.path.join(self.condition_path, parts[-3], parts[-2],
                                      parts[-1])  # condition's path should align with the image's path

        sample = self.loader(path)
        condition_sample = self.loader(condition_path)
        
        # 确保条件图像和目标图像进行相同的预处理和裁剪
        if self.transform:
            # 训练时应用相同的随机裁剪
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(
                    sample, output_size=(256, 256)
                )
                sample = F.crop(sample, i, j, h, w)
                condition_sample = F.crop(condition_sample, i, j, h, w)
            # 验证时使用中心裁剪
            else:
                sample = self.transform(sample)
                condition_sample = self.transform(condition_sample)
        
        # 生成控制图像
        control_images = get_control_for_each_scale(condition_sample, self.pn)

        if self.target_transform is not None:
            target = self.target_transform(target)

        post_trans = transforms.Compose([transforms.ToTensor(), normalize_01_into_pm1])
        sample = post_trans(sample)
        control_tensors = []
        for ci in control_images:
            control_tensors.append(post_trans(ci))

        return sample, control_tensors, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".npy")


def pil_loader(path: str) -> Image.Image:
    """专为.npy医学图像设计的加载器"""
    if path.endswith('.npy'):
        # 加载.npy格式的医学图像
        img = np.load(path).astype(np.float32)
        
        # 标准化：减去均值除以标准差
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / (std + 1e-8)  # 防止除以0
        
        # 归一化到[0,1]范围保留完整动态范围
        img_min, img_max = np.min(img), np.max(img)
        img_range = img_max - img_min
        if img_range > 1e-8:  # 避免除0
            img = (img - img_min) / img_range
        else:
            img = np.zeros_like(img)  # 全零图像处理
        
        # 转换为3通道PIL图像
        img = (img * 255).astype(np.uint8)
        img = np.stack([img]*3, axis=-1)  # (H,W,3)
        return Image.fromarray(img)
    else:
        # 普通RGB图像处理
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(ConditionDatasetFolder):
    """A generic data loader where the images are arranged in a specific way."""

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            unet_cache_dir: str = '/data/tangyingxin/CAR/unet_features',  # UNet特征缓存目录
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            unet_cache_dir=unet_cache_dir,  # 传递UNet缓存目录参数
        )
        self.imgs = self.samples
