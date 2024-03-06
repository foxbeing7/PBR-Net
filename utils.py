import numpy as np
import math
import torch
import torchvision.models as models
import torchvision
from skimage.metrics import structural_similarity as ssim

import os
import shutil
# 运行可以生成目标文件夹结构

def compute_ssim(target_image, generated_image):


    target_image = np.clip(target_image, 0, 1)
    generated_image = np.clip(generated_image, 0, 1)

    ssim_value, _ = ssim(target_image, generated_image, multichannel=True)

    return ssim_value


def adjust_folder_structure(original_folder, target_folder):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    subfolders = ['image', 'albedo', 'normal', 'mask']

    for subfolder in subfolders:
        target_subfolder = os.path.join(target_folder, subfolder)
        os.makedirs(target_subfolder, exist_ok=True)

        for sub_id in os.listdir(original_folder):
            sub_id_path = os.path.join(original_folder, sub_id)
            if os.path.isdir(sub_id_path):
                source_subfolder = os.path.join(sub_id_path, subfolder)
                target_sub_id_folder = os.path.join(target_subfolder, sub_id)
                os.makedirs(target_sub_id_folder, exist_ok=True)
                for file in os.listdir(source_subfolder):
                    source_file_path = os.path.join(source_subfolder, file)
                    target_file_path = os.path.join(target_sub_id_folder, file)
                    shutil.copyfile(source_file_path, target_file_path)

# Adjust folder structure
original_folder = './dataset/blender_data'
target_folder = './dataset/adjusted_blender_data'
adjust_folder_structure(original_folder, target_folder)

