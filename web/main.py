#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：hw_template
@File    ：main.py
@Author  ：Norton
@Date    ：2023/3/31 16:23
@Description：用来显示页面
'''

from flask import Flask, render_template
import torch
# from train_resnet import SelfNet
from train import SELFMODEL
import os
import io
import json
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
from torchutils import get_torch_transforms
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, render_template


# 实力化对象Flask
app = Flask(__name__)
class_json_path = "E:\\04-design\\2023_wheat\\web\\class_indices.json"
device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

model_path = "checkpoints/resnet50d_pretrained_224/resnet50d_9epochs_accuracy0.91595_weights.pth"  # todo  模型路径
classes_names = ['CrownAndRootRot', 'HealthyWheat', 'LeafRust', 'PowderyMildew', 'WheatLooseSmut', 'WheatAphids', 'WheatCystNematode', 'WheatRedSpider', 'WheatScab', 'WheatSharpEyespot', 'WheatStalkRot', 'WheatTake-all']  # todo 类名
# model_path = "E:/04-design/checkpoints/efficientnet_b3a_pretrained_224/efficientnet_b3a_8epochs_accuracy0.98925_weights.pth"  # todo  模型路径
# classes_names = ['CrownAndRootRot', 'HealthyWheat', 'LeafRust']  # todo 类名
img_size = 224  # todo 图片大小
model_name = "resnet50d"  # todo 模型名称
num_classes = len(classes_names)  # todo 类别数目


def predict_batch(model_path, target_dir, save_dir):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False).cuda()
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片
    image_names = os.listdir(target_dir)
    for i, image_name in enumerate(image_names):
        image_path = osp.join(target_dir, image_name)
        img = Image.open(image_path)
        img = valid_transforms(img)  #数据增强
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model(img)
        label_id = torch.argmax(output).item()
        predict_name = classes_names[label_id]
        save_path = osp.join(save_dir, predict_name)
        if not osp.isdir(save_path):
            os.makedirs(save_path)
        shutil.copy(image_path, save_path)
        print(f"{i + 1}: {image_name} result {predict_name}")


def predict_single(model_path, image_byte):
    data_transforms = get_torch_transforms(img_size=img_size)
    # train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片
    # img = Image.open(image_path)
    img = Image.open(io.BytesIO(image_byte))
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    label_id = torch.argmax(output).item()
    predict_name = classes_names[label_id]
    return predict_name



# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info



@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    # info = predict_single(model_path=model_path, image_byte=img_bytes)
    return jsonify(info)



@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")

if __name__ == '__main__':

    app.run()
    # # 批量预测函数
    # predict_batch(model_path=model_path,
    #               target_dir="E:/04-design/2023_pytorch110_classification_42-master/data/pr",
    #               save_dir="E:/04-design/2023_pytorch110_classification_42-master/data/pr/mini_result")
    # 单张图片预测函数
    # predict_single(model_path=model_path, image_path="images/test_imgs/506659320_6fac46551e.jpg")
#
#
#
#
#
#     # # 创建了网址/show/info和函数index的对应关系
#     # # 以后用户再浏览器访问/show/info时，网站自动执行函数index
#     # @app.route("/show/info")
#     # def index():
#     #     # flask内部会自动打开这个文件，读取内容，并将内容返回给用户
#     #     # 默认：去当前项目目录的templates文件夹中找。
#     #     return render_template("up.html")
#     #