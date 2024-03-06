"""
 make sure dataset s structure like: ---dataset
                                      ----image
                                        ----...
                                      ----albedo
                                        ----...
                                     ----normal
                                        ----...
                                     ----mask
                                        ----...
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


source_dir = './dataset/half/image'
albedo_dir = './dataset/half/albedo'
normal_dir = './dataset/half/normal'
mask_dir = './dataset/half/mask'
slice_size = 1

class CustomDataset(Dataset):
    def __init__(self, source_dir, albedo_dir, normal_dir, mask_dir , transform=None):
        self.source_dir = source_dir
        self.albedo_dir = albedo_dir
        self.normal_dir = normal_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Get list of source image filenames with paths
        self.image_list = self._get_image_list()

    def _get_image_list(self):
        image_list = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.png'):
                    source_path = os.path.join(root, file)
                    image_list.append(source_path)
        random.shuffle(image_list)
        subset_size = int(slice_size * len(image_list))
        return image_list[:subset_size]
        # return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        source_path = self.image_list[idx]


        rel_path = os.path.relpath(source_path, self.source_dir)
        # todo
        # if albedo is end with .png , then remove .replace
        # albedo_path = os.path.join(self.albedo_dir, rel_path.replace('.png', '.jpg'))
        # normal_path = os.path.join(self.normal_dir, rel_path)
        # mask_path = os.path.join(self.mask_dir ,rel_path.replace('.png', '.jpg'))
        albedo_path = os.path.join(self.albedo_dir, rel_path)
        normal_path = os.path.join(self.normal_dir, rel_path)
        mask_path = os.path.join(self.mask_dir, rel_path)

        try:
            image = Image.open(source_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning!one or more files not found for {source_path}.Skipping")
        try:
            albedo = Image.open(albedo_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning!one or more files not found for {albedo_path}.Skipping")
        try:
            normal = Image.open(normal_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning!one or more files not found for {normal_path}.Skipping")
        try:
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            print(f"Warning!one or more files not found for {mask_path}.Skipping")
            return None

        mask_array = np.array(mask)
        binary_mask_array = (mask_array > 100).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask_array)

        image_masked = Image.fromarray(np.array(image) * np.expand_dims(np.array(mask_image) > 0, axis=2))
        albedo_masked = Image.fromarray(np.array(albedo) * np.expand_dims(np.array(mask_image) > 0, axis=2))
        normal_masked = Image.fromarray(np.array(normal) * np.expand_dims(np.array(mask_image) > 0, axis=2))

        if self.transform:
            image = self.transform(image_masked)
            albedo = self.transform(albedo_masked)
            normal = self.transform(normal_masked)
        return {'image': image, 'albedo': albedo, 'normal': normal}


data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


custom_dataset = CustomDataset(source_dir, albedo_dir, normal_dir, mask_dir, transform=data_transform)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(custom_dataset))
val_size = len(custom_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

