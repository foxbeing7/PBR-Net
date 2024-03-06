# this code convert an exr img folder to png img folder
# change the path config ...

import os
import numpy as np
from PIL import Image
import OpenEXR
import Imath


exr_folder = r'D:\workspace\data\light_stage\source\1223'
png_folder = r'D:\workspace\data\light_stage\image'
def convert_exr_to_png(exr_path, png_path):

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()


    width = int(header['dataWindow'].max.x - header['dataWindow'].min.x + 1)
    height = int(header['dataWindow'].max.y - header['dataWindow'].min.y + 1)


    redstr = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    greenstr = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
    bluestr = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))

    red = np.frombuffer(redstr, dtype=np.float32)
    green = np.frombuffer(greenstr, dtype=np.float32)
    blue = np.frombuffer(bluestr, dtype=np.float32)


    red = (red * 255).clip(0, 255).astype(np.uint8)
    green = (green * 255).clip(0, 255).astype(np.uint8)
    blue = (blue * 255).clip(0, 255).astype(np.uint8)


    img = Image.merge("RGB", (Image.fromarray(red.reshape(height, width)),
                              Image.fromarray(green.reshape(height, width)),
                              Image.fromarray(blue.reshape(height, width))))


    img.save(png_path)
    print(f"-------------{exr_path} 转换完成----------------")
def convert_folder(exr_folder, png_folder):

    os.makedirs(png_folder, exist_ok=True)

    for root, dirs, files in os.walk(exr_folder):
        for file in files:
            if file.endswith('.exr'):
                exr_path = os.path.join(root, file)

                png_path = os.path.join(png_folder, os.path.relpath(exr_path, exr_folder).replace('.exr', '.png'))

                os.makedirs(os.path.dirname(png_path), exist_ok=True)

                convert_exr_to_png(exr_path, png_path)


convert_folder(exr_folder, png_folder)



