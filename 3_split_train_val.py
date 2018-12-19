import os
import numpy as np
import shutil

np.random.seed(2016)

"""
UCMerced_LandUse 数据说明：
共21类，每类100张
其中每类的前91张作为训练集，后9张作为测试集
训练集：1911
测试集：189


2015_4_classes 数据说明：
共4类，building:48  other:56  water:46  zhibei:92
其中每类的前4张作为测试集
训练集：242
测试集：16

"""

# root_train = '/media/files/xdm/classification/data/UCMerced_LandUse/train_split'
# root_val = '/media/files/xdm/classification/data/UCMerced_LandUse/val_split'

root_train = '/media/files/xdm/classification/data/2015_4_classes/aug_256/train/train_split'
root_val = '/media/files/xdm/classification/data/2015_4_classes/aug_256/train/val_split'


root_total = '/media/files/xdm/classification/data/2015_4_classes/aug_256/train'


# root_train = r'E:\UCMerced_LandUse\data\train_split'
# root_val = r'E:\UCMerced_LandUse\data\val_split'
#
# root_total = r'E:\UCMerced_LandUse\data\train'

if not os.path.exists(root_train):
    os.mkdir(root_train)

if not os.path.exists(root_val):
    os.mkdir(root_val)

# # 共21类
# FishNames = ['agricultural', 'airplane', 'baseballdiamond', 'beach',
#              'buildings', 'chaparral', 'denseresidential', 'forest',
#              'freeway', 'golfcourse', 'harbor', 'intersection',
#              'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot',
#              'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
#              ]

# 共21类
FishNames = ['building', 'other', 'water', 'zhibei']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for fish in FishNames:
    # 如果该文件夹不存在，则创建
    if fish not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, fish))

    total_images = os.listdir(os.path.join(root_total, fish))

    nbr_train = int(len(total_images) * split_proportion)

    # 数据打乱顺序
    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_train, fish, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if fish not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, fish))

    for img in val_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_val, fish, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
# # training samples: 191, # val samples: 51