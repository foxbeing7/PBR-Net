import numpy as np
import cv2
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def exr_to_png(exr_path, png_path):
    # 读取 EXR 格式的图像
    image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

    # 写入 PNG 格式的图像
    cv2.imwrite(png_path, image)

def batch_convert_exr_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 EXR 文件
    for exr_filename in os.listdir(input_folder):
        if exr_filename.endswith(".exr"):
            exr_path = os.path.join(input_folder, exr_filename)

            # 构建对应的 PNG 文件名
            png_filename = os.path.splitext(exr_filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)

            # 转换并保存图像
            exr_to_png(exr_path, png_path)

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"D:\workspace\project\image-relighting\output\bld\aya"
    output_folder = r"D:\workspace\project\image-relighting\output\bld\aya\png"

    # 批量转换 EXR 到 PNG
    batch_convert_exr_to_png(input_folder, output_folder)
