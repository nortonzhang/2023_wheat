# # import torch
# # from torchvision import models
# #
# # # 有 GPU 就用 GPU，没有就用 CPU
# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # print('device', device)
# #
# # model = torch.load('save.pt')
# # model = model.eval().to(device)
# #
# # x = torch.randn(1, 3, 224, 224).to(device)
# #
# # output = model(x)
# #
# # output.shape
# #
# # x = torch.randn(1, 3, 224, 224).to(device)
# #
# #
# # with torch.no_grad():
# #     torch.onnx.export(
# #         model,                   # 要转换的模型
# #         x,                       # 模型的任意一组输入
# #         'resnet50d_wheat23.onnx', # 导出的 ONNX 文件名
# #         opset_version=11,        # ONNX 算子集版本
# #         input_names=['input'],   # 输入 Tensor 的名称（自己起名字）
# #         output_names=['output']  # 输出 Tensor 的名称（自己起名字）
# #     )
#
#
# import onnx
#
# # 读取 ONNX 模型
# onnx_model = onnx.load('test.onnx')
#
# # 检查模型格式是否正确
# onnx.checker.check_model(onnx_model)
#
# print('无报错，onnx模型载入成功')


import torch
import torchvision.models as models

model = models.vgg16(pretrained=False)
model.load_state_dict(torch.load("C:\\Users\\Norton\\.cache\\torch\\hub\\checkpoints\\vgg16-397923af.pth"))
print(model)