# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: feature-data-enhancement
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/5
# @Time        : 下午5:33
# @Description :
import numpy as np
from PIL import Image
import os

all_image_array = None

def step1_images():
    global all_image_array
    # 设置目录和输出文件名
    directory = "./out/breast/Step1/best/US"
    output_file = "./out/Step1_us.png"
    real_image = "./out/breast/real_us.png"
    # 获取目录下所有图片文件
    image_files = [real_image] + [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]

    img_array = None

    # 将所有图片拼接合成新图片

    for i, v in enumerate(image_files[:10]):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)


    img.save(output_file)
    all_image_array = img

def step1_images_ceus():
    global all_image_array

    # 设置目录和输出文件名
    directory = "./out/breast/Step1/best/CEUS"
    output_file = "./out/Step1_ceus.png"
    real_image = "./out/breast/real_ceus.png"
    # 获取目录下所有图片文件
    image_files = [real_image] + [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]

    img_array = None

    # 将所有图片拼接合成新图片

    for i, v in enumerate(image_files[:10]):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)


    img.save(output_file)
    all_image_array = np.concatenate((all_image_array, img), axis=0)
def step2_images():
    global all_image_array

    # 设置目录和输出文件名
    directory = "./out/breast/Step2/US"
    output_file = "./out/Step2_us.png"
    real_image = "./out/breast/real_us.png"
    # 获取目录下所有图片文件
    image_files = [real_image] + [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]

    img_array = None

    # 将所有图片拼接合成新图片

    for i, v in enumerate(image_files[:10]):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)


    img.save(output_file)
    all_image_array = np.concatenate((all_image_array, img), axis=0)



def step2_images_ceus():
    global all_image_array

    # 设置目录和输出文件名
    directory = "./out/breast/Step2/CEUS"
    output_file = "./out/Step2_ceus.png"
    real_image = "./out/breast/real_ceus.png"
    # 获取目录下所有图片文件
    image_files = [real_image] + [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]

    img_array = None

    # 将所有图片拼接合成新图片

    for i, v in enumerate(image_files[:10]):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)


    img.save(output_file)
    all_image_array = np.concatenate((all_image_array, img), axis=0)



if __name__ == '__main__':
    step1_images()
    step2_images()
    step1_images_ceus()
    step2_images_ceus()

    img = Image.fromarray(all_image_array)
    img.save("./out/all.png")
