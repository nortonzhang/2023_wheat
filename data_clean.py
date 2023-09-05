import shutil
import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


# 实际的图片保存和读取的过程中存在中文，所以这里通过这两种方式来应对中文读取的情况。
# handle chinese path
def cv_imread(file_path, type=-1):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)  #读取中文图片路径
    if type == 0:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img


def cv_imwrite(file_path, cv_img, is_gray=True):
    if len(cv_img.shape) == 3 and is_gray:       #灰度图且参数为长高和通道数
        cv_img = cv_img[:, :, 0]
    cv2.imencode(file_path[-4:], cv_img)[1].tofile(file_path)   #调整大小输入缓冲区


def data_clean(src_folder, english_name):
    clean_folder = src_folder + "_cleaned"
    if os.path.isdir(clean_folder):
        print("保存目录已存在")
        shutil.rmtree(clean_folder)    #删除文件夹及子文件
    os.mkdir(clean_folder)
    # 数据清洗的过程主要是通过oepncv来进行读取，读取之后没有问题就可以进行保存
    # 数据清洗的过程中，一是为了保证数据是可以读取的，二是需要将原先的中文修改为英文，方便后续的程序读取。
    image_names = os.listdir(src_folder)
    with tqdm(total=len(image_names)) as pabr:    #添加进度提示信息
        for i, image_name in enumerate(image_names):
            image_path = osp.join(src_folder, image_name)
            try:
                img = cv_imread(image_path)
                img_channel = img.shape[-1]
                if img_channel == 3:
                    save_image_name = english_name + "_" + str(i) + ".jpg"
                    save_path = osp.join(clean_folder, save_image_name)
                    cv_imwrite(file_path=save_path, cv_img=img, is_gray=False)    #对图像进行重命名
            except:
                print("{}是坏图".format(image_name))         #图像有问题就抛出异常，并不再采用该图像
            pabr.update(1)         #进度条叠加


if __name__ == '__main__':

    data_clean(src_folder="E:\\04-design\\03-data\\Crown and Root Rot",
               english_name="CrownAndRootRot")
    data_clean(src_folder="E:\\04-design\\03-data\\Healthy Wheat",
               english_name="HealthyWheat")
    data_clean(src_folder="E:\\04-design\\03-data\\Leaf Rust",
               english_name="LeafRust")
    data_clean(src_folder="E:\\04-design\\03-data\\Powdery Mildew",
               english_name="PowderyMildew")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat Aphids",
               english_name="WheatAphids")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat cyst nematode",
               english_name="WheatCystNematode")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat Loose Smut",
               english_name="WheatLooseSmut")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat Red Spider",
               english_name="WheatRedSpider")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat scab",
               english_name="WheatScab")
    data_clean(src_folder="E:\\04-design\\03-data\\wheat sharp eyespot",
               english_name="WheatSharpEyespot")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat stalk rot",
               english_name="WheatStalkRot")
    data_clean(src_folder="E:\\04-design\\03-data\\Wheat Take-all",
               english_name="WheatTake-all")

