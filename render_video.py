'''
如果output_video_path不为空，则会生成渲染视频.
'''
# -*- coding: utf-8 -*-
import math
import os
from PIL import Image
import numpy as np
import cv2

light_color = np.array([127, 127, 254])
light_intensity = 1.5
num_directions = 360
input_image_path = "./output/predict_with_mask/test0/boss/image.png"
normal_image_path = "./output/predict_with_mask/test0/boss/normal.png"
albedo_image_path = "./output/predict_with_mask/test0/boss/albedo.png"
output_folder = "./output/predict_with_mask/test0/boss/relight_white"
output_video_path = "./output/predict_with_mask/test0/boss/relight_white/0_relighted_video.mp4"
frame_rate = 24
gen_image = True
def load_image(file_path):
    image = Image.open(file_path)
    return np.array(image)

def render_lighting(normal_image, albedo_image, light_direction, light_color, light_intensity):
    normal = normal_image / 255.0 * 2 - 1
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    albedo = albedo_image / 255.0

    light_direction = light_direction / np.linalg.norm(light_direction)

    lambertian_intensity = np.maximum(0, np.sum(normal * light_direction, axis=-1))

    scaled_color = light_color / 255.0

    color = albedo * lambertian_intensity[..., np.newaxis] * scaled_color * light_intensity

    return color

def relight_image(input_image_path, normal_image_path, albedo_image_path, output_folder, output_video_path=None, frame_rate=24):
    input_image = load_image(input_image_path)
    normal_image = load_image(normal_image_path)
    albedo_image = load_image(albedo_image_path)

    images = []

    for i in range(num_directions):
        angle = 2 * math.pi * i / num_directions
        light_direction = np.array([math.cos(angle), math.sin(angle), 0])
        result_color = render_lighting(normal_image, albedo_image, light_direction, light_color, light_intensity)

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
    create_video(output_folder,output_video_path)
    print("生成了relighted视频保存在：",output_video_path)