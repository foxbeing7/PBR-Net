# -*- coding: utf-8 -*-
import math
import os
from PIL import Image
import numpy as np
import cv2
import colorsys

light_intensity = 1.2
num_directions = 360
input_image_path = "./test/Huge/image.png"
normal_image_path = "./test/Huge/normal.png"
albedo_image_path = "./test/Huge/albedo.png"
output_folder = "./test/Huge/relight_spectrum"
output_video_path = "./test/Huge/relight_spectrum/0_relighted_video.mp4"
frame_rate = 24
gen_image = True

def load_image(file_path):
    image = Image.open(file_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return np.array(image)

def calculate_color_factor(spectrum_factor):
    # 计算沿着赤橙黄绿青蓝紫的颜色序列
    hue = spectrum_factor / (2 * math.pi)
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    color_factor = np.array([r, g, b])
    return color_factor

def render_lighting(normal_image, albedo_image, light_direction, light_intensity, spectrum_factor):
    normal = normal_image / 255.0 * 2 - 1
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    albedo = albedo_image / 255.0

    light_direction = light_direction / np.linalg.norm(light_direction)

    lambertian_intensity = np.maximum(0, np.sum(normal * light_direction, axis=-1))

    # 计算沿着赤橙黄绿青蓝紫的颜色序列
    color_factor = calculate_color_factor(spectrum_factor)
    scaled_color = light_intensity * color_factor

    color = albedo * lambertian_intensity[..., np.newaxis] * scaled_color

    return color

def relight_image(input_image_path, normal_image_path, albedo_image_path, output_folder, output_video_path=None, frame_rate=24):
    input_image = load_image(input_image_path)
    normal_image = load_image(normal_image_path)
    albedo_image = load_image(albedo_image_path)

    images = []

    for i in range(num_directions):
        angle = 2 * math.pi * i / num_directions
        spectrum_factor = 2 * math.pi * i / num_directions  # 光谱变化因子
        light_direction = np.array([math.cos(angle), math.sin(angle), 0])
        result_color = render_lighting(normal_image, albedo_image, light_direction, light_intensity, spectrum_factor)

        relighted_image = np.minimum((input_image / 255.0) * (albedo_image / 255.0) + result_color, 1.0)

        output_image_path = f"{output_folder}/relighted_{i:03d}.png"
        Image.fromarray((relighted_image * 255).astype(np.uint8)).save(output_image_path)
        images.append((relighted_image * 255).astype(np.uint8))

        print("relighted图片保存在了:", output_image_path)

def create_video(image_folder, output_video_path, frame_rate=24):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

os.makedirs(output_folder, exist_ok=True)

if gen_image:
    relight_image(input_image_path, normal_image_path, albedo_image_path, output_folder, output_video_path, frame_rate)

if output_video_path is not None:
    create_video(output_folder, output_video_path)
    print("生成了relighted视频保存在：", output_video_path)
