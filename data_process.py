import tifffile as tiff
import os

"""
2016年的原图是4通道的，而且蓝绿通道顺序反了

处理图像：将蓝绿通道交换顺序，并去掉红外波段

"""

def RGBA_RGB(path, save_path):
    fileList = os.listdir(path)
    for file in fileList:
        image = tiff.imread(os.path.join(path, file))
        # 交换通道
        image = image[:, :, (2, 1, 0)]
        # image = image.resize((1000, 1000))
        print(image.shape)
        tiff.imsave(save_path + file, image)


# 只保留后缀是.TIF的图像
def remove_file(path):
    fileList = os.listdir(path)
    for file in fileList:
        pathname = os.path.splitext(os.path.join(path, file))
        if pathname[1] != ".TIF":
            os.remove(os.path.join(path, file))


if __name__ == '__main__':
    path = r'F:\remote_sensing\16年原图\\'
    save_path = r'F:\remote_sensing\16年_RGB\\'

    # remove_file(path)

    RGBA_RGB(path, save_path)