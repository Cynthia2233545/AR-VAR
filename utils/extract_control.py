import numpy as np
import cv2
import PIL.Image as PImage
import torch
from conditions.util import resize_image, HWC3, load_image, nms, ckpt_path
from PIL import Image
import matplotlib.pyplot as plt
def extract(type, image_path):
    # input_image = np.array(load_image(image_path).resize(size=(256, 256)))
    input_image = np.load(image_path).astype(np.float32)
    control = None
    if type == 'canny':
        low_threshold = 100
        high_threshold = 200
        canny_image = cv2.Canny(input_image, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = PImage.fromarray(canny_image)
        control = canny_image
    elif type == 'normal':
        from conditions.normal import NormalBaeDetector
        processor = NormalBaeDetector.from_pretrained(ckpt_path)
        control = PImage.fromarray(processor(input_image, output_type="np"))
    elif type == 'depth':
        from conditions.midas import MidasDetector
        apply_midas = MidasDetector()
        with torch.no_grad():
            input_image = HWC3(input_image)
            detected_map, _ = apply_midas(resize_image(input_image, 384), bg_th=0.4)
            detected_map = HWC3(detected_map)
            H, W, C = input_image.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            control = PImage.fromarray(detected_map)
    elif type == 'hed':
        from conditions.hed import HEDdetector
        apply_hed = HEDdetector()
        with torch.no_grad():
            input_image = HWC3(input_image)
            detected_map = apply_hed(resize_image(input_image, 512))
            detected_map = HWC3(detected_map)
            H, W, C = input_image.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            control = PImage.fromarray(detected_map)
    elif type == 'sketch':
        img = resize_image(input_image, 512)
        from conditions.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
        control = model_pidinet(img, device='cuda:0', ckpt_path=ckpt_path)
        control = nms(control, 127, 3.0)
        control = cv2.GaussianBlur(control, (0, 0), 3.0)
        control[control > 4] = 255
        control[control < 255] = 0
        control = control[:, :, None]
        control = np.concatenate([control, control, control], axis=2)
        control = PImage.fromarray(control).resize((256, 256))
    elif type == 'ct':
        input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image)) 
        img_cropped = np.expand_dims(input_image, axis=2)
        image_stacked = np.repeat(img_cropped, 3, axis=2)
        # print("-----------------------------------------------------------------",image_stacked.shape)
        image_stacked = (image_stacked * 255).astype(np.uint8)  # 转换为 uint8 类型
        # 使用 PIL 将 NumPy 数组转换为 Image 对象
        image_stacked = Image.fromarray(image_stacked)
        control = image_stacked
        # control.save('control.png')
    return control
