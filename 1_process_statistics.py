# coding=utf-8
import csv
import os
import random

import tifffile as tiff
import numpy as np
import cv2


img_w = 256
img_h = 256


"""
0_函数说明：
   只保留后缀为tif的图像
"""
def process_0(path):
    file_list = os.listdir(path) # 获取path目录下的所有文件
    for file in file_list:
        file_path = os.path.join(path, file)  # 获取文件路径
        if os.path.isdir(file_path):  # 如果当前是文件夹，递归
            process_0(file_path)
        else:
            if not file_path.endswith('.TIF'):
                os.remove(file_path)


"""
1_函数说明：
    统计图片的尺寸以及面积
    输入：图片路径
    输出：
"""
def process_1(path):
    result = []
    file_list = os.listdir(path) # 获取path目录下的所有文件
    for file in file_list:
        file_name = file.split(".")[0]  # 获取文件的名称（用来保存统计信息）
        file_path = os.path.join(path, file)  # 获取文件路径
        if os.path.isdir(file_path):  # 如果当前是文件夹，递归
            process_1(file_path)
        else:
            file = tiff.imread(file_path)  # 读取图片
            chicu = file.shape[:2]  # 获取图片的尺寸
            area = chicu[0] * chicu[1] # 获取图片的面积
            result.append((file_name, chicu, area))
    with open('统计_' + path.split('\\')[-2] + '.csv', 'a', newline='') as myfile:
        mywriter = csv.writer(myfile)
        mywriter.writerows(result)


"""
2_函数说明：假定对象的最小尺寸要求不小于64*64
    1. 删除高或宽尺寸小于64的图片
"""
def process_2(path):
    file_list = os.listdir(path)  # 获取path目录下的所有文件
    for file in file_list:
        file_path = os.path.join(path, file)  # 获取文件路径
        if os.path.isdir(file_path):  # 如果当前是文件夹，递归
            process_2(file_path)
        else:
            file = tiff.imread(file_path)  # 读取图片
            chicu = file.shape[:2]  # 获取图片的尺寸
            if chicu[0] < img_w or chicu[1] < img_h: # 1. 删除不符合要求的图片
                os.remove(file_path)


def creat_dataset(image_num=50000, mode='original', path='', aug_path=''):
    if not os.path.exists(aug_path):
        os.mkdir(aug_path)
    file_list = os.listdir(path)  # 获取path目录下的所有文件
    image_sets = [] # 统
    dir_names = []
    for file in file_list: # 该文件夹下的子文件夹（building,zhibei,water,other）
        dir_names.append(file)
        image_sets.append(os.listdir(os.path.join(path, file)))
    # print(image_sets)
    for i in range(len(image_sets)): # 每个文件夹下的文件
        if not os.path.exists(aug_path + os.sep + dir_names[i]):
            os.mkdir(aug_path + os.sep + dir_names[i])

        # print(len(image_sets[i]))
        image_each = image_num / len(image_sets[i])
        # print(image_each)
        g_count = 0
        for j in range(len(image_sets[i])):
                count = 0
                # print(image_sets[i][j])
                img_path = path + os.sep + dir_names[i] + os.sep + image_sets[i][j]
                print(img_path)
                src_img = tiff.imread(img_path)
                X_height, X_width, _ = src_img.shape
                while count < image_each:
                    random_width = random.randint(0, X_width - img_w - 1)
                    random_height = random.randint(0, X_height - img_h - 1)
                    src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
                    if mode == 'augment':
                        src_roi = data_augment(src_roi)
                    # print(aug_path + os.sep + dir_names[i])
                    cv2.imwrite((aug_path + os.sep + dir_names[i] + '/%d.tif' % g_count), src_roi)
                    count += 1
                    g_count += 1


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    return xb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb):
    if np.random.random() < 0.25:
        xb = rotate(xb, 90)
    if np.random.random() < 0.25:
        xb = rotate(xb, 180)
    if np.random.random() < 0.25:
        xb = rotate(xb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    # if np.random.random() < 0.2:
    #     xb = add_noise(xb)

    return xb


if __name__ == '__main__':

    """
    说明：
            1. 文件夹 2_process_2_64下：保存的是尺寸大于64的图像
            2. 文件夹 2_process_2_128下：保存的是尺寸大于128的图像
            3. 文件夹 2_process_aug_64：对应2_process_2
            3. 文件夹 2_process_aug_128：对应2_process_2_128
            
    """
    # 服务器地址
    # Sample_Library
    # path = r'F:\remote_sensing\2015_4_class\arggis\2015_2\2_process_1'
    # print(path.split('\\')[-2])
    # process_1(path)
    # path = r'F:\remote_sensing\2015_4_class\arggis\2015_2\2_process_2_256'
    # process_2(path)
    # aug_path = r'F:\remote_sensing\2015_4_class\arggis\2015_2\2_process_aug_256'
    # creat_dataset(image_num=10, mode='augment', path=path, aug_path=aug_path)

    # path = r'F:\remote_sensing\2015_4_class\arggis\2015_3\2_process'
    # process_0(path)
    # process_1(path)
    # path = r'F:\remote_sensing\2015_4_class\arggis\2015_3\2_process_128'
    # process_2(path)



    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_4\2_process"
    # process_0(path)
    # process_1(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_4\2_process_128"
    # process_2(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_5\2_process"
    # process_0(path)
    # process_1(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_5\2_process_128"
    # process_2(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_6\2_process"
    # process_0(path)
    # process_1(path)
    #
    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_6\2_process_128"
    # process_2(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_7\2_process"
    # process_0(path)
    # process_1(path)
    #
    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_7\2_process_128"
    # process_2(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_8\2_process"
    # process_0(path)
    # process_1(path)

    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_8\2_process_128"
    # process_2(path)
    #
    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_1\2_process"
    # process_0(path)
    # process_1(path)
    #
    # path = r"F:\remote_sensing\2015_4_class\arggis\2015_1\2_process_128"
    # process_2(path)


    # path = r'F:\remote_sensing\2015_4_class\arggis\2015_5\2_process_256'
    # qianzhui = '5_'
    # for file in os.listdir(path):
    #     for name in os.listdir(os.path.join(path, file)):
    #         newName = qianzhui + name.split("_")[1]
    #         print(newName)
    #         os.rename(os.path.join(path, file, name), os.path.join(path, file, newName))

    path = r'F:\remote_sensing\2015_4_class\summary\256'
    # process_2(path)
    aug_path = r'F:\remote_sensing\2015_4_class\summary\aug_256'
    creat_dataset(image_num=50, mode='augment', path=path, aug_path=aug_path)