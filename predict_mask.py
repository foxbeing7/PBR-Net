import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from attmodel import normal_UNet, albedo_UNet, pbr_unet
import cv2

def predict(model, image_path, output_path, mask_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    size = (512,512)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_image = Image.open(image_path).convert('RGB')
    ori_size = input_image.size

    if mask_path is not None:
        mask_image = Image.open(mask_path).convert('L')

        mask_array = np.array(mask_image)
        binary_mask_array = (mask_array > 37.5).astype(np.uint8) * 255
        binary_mask_image = Image.fromarray(binary_mask_array)

        masked_image = Image.fromarray(np.array(input_image) * np.expand_dims(np.array(binary_mask_image) > 0, axis=2))
        input_tensor = transform(masked_image).unsqueeze(0).to(device)
    else:
        input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Pass mask_tensor as an additional argument if using it
        if input_tensor is not None:
            outputs_normal, outputs_albedo = model(input_tensor)
        else:
            print("error....input tensor is null")
            return

    os.makedirs(output_path, exist_ok=True)
    # masked_image = Image.fromarray(np.array(input_image) * np.expand_dims(np.array(mask_image) > 0, axis=2))
    masked_image.save(os.path.join(output_path, 'masked.png'))
    mask_image.save(os.path.join(output_path,'mask.png'))
    # Save input image
    input_image.save(os.path.join(output_path, 'image.png'))
    # Inverse Trans ...
    inv_normalize = transforms.Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
    )
    # Save predicted normal
    normal = inv_normalize(outputs_normal)
    normal_image = normal.squeeze(0).cpu().permute(1, 2, 0).numpy()
    inv_resized_normal = Image.fromarray((normal_image * 255).astype('uint8')).resize(ori_size)
    inv_masked_normal = Image.fromarray(np.array(inv_resized_normal) * np.expand_dims(np.array(binary_mask_image) > 0, axis=2))
    inv_masked_normal.save(os.path.join(output_path, 'normal.png'))

    # Save predicted albedo
    albedo = inv_normalize(outputs_albedo)
    albedo_image = albedo.squeeze(0).cpu().permute(1, 2, 0).numpy()
    inv_resized_albedo = Image.fromarray((albedo_image * 255).astype('uint8')).resize(ori_size)
    inv_masked_albedo = Image.fromarray(np.array(inv_resized_albedo) * np.expand_dims(np.array(binary_mask_image) > 0, axis=2))
    inv_masked_albedo.save(os.path.join(output_path, 'albedo.png'))

if __name__ == '__main__':
    model_path = './ckpts/latest_finetune_50.pth'
    output_base_path = './output/predict_with_mask/Lumos基础上微调50个epoch'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_unet_model = normal_UNet()
    albedo_unet_model = albedo_UNet()
    combined_unet_model = pbr_unet(normal_unet_model, albedo_unet_model)

    combined_unet_model.load_state_dict(torch.load(model_path))
    combined_unet_model.to(device)
    combined_unet_model.eval()

    input_image_folder = r'./dataset/Lumos/test/image'
    mask_image_folder = r'./dataset/Lumos/test/mask'
    for image_file in os.listdir(input_image_folder):
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            image_path = os.path.join(input_image_folder, image_file)
            output_path = os.path.join(output_base_path, image_file.split('.')[0])


            # mask_filename = f"{image_file.split('.')[0]}_mask.jpg"
            # mask_path = os.path.join(mask_image_folder, mask_filename)
            # Construct mask path based on the image file name
            mask_filename_png = f"{image_file.split('.')[0]}_mask.png"
            mask_filename_jpg = f"{image_file.split('.')[0]}_mask.jpg"

            mask_path_png = os.path.join(mask_image_folder, mask_filename_png)
            mask_path_jpg = os.path.join(mask_image_folder, mask_filename_jpg)

            # Check if the mask exists in PNG format, otherwise, try JPG format
            if os.path.exists(mask_path_png):
                mask_path = mask_path_png
            elif os.path.exists(mask_path_jpg):
                mask_path = mask_path_jpg
            else:
                print(f"Warning: Mask not found for image {image_file}")
                continue
            # predict, pass mask_path as an argument
            predict(combined_unet_model, image_path, output_path, mask_path)

    print('预测完成，图片的normal和albedo保存在:', output_base_path)
