import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json
import torch.nn as nn
# 加载预训练的 VGG19 模型
# vgg19 = models.vgg19(pretrained=True)
# vgg19.eval()  # 设置模型为评估模式




class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  #224

            nn.MaxPool2d(kernel_size=2, stride=2), #112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), #56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), #28
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2), #14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2), #7
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


vgg19 = VGG19()
vgg19.eval()  # 设置模型为评估模式

# 加载预训练权重
vgg19_weights_path = "vgg19.pth"  # 预训练权重文件路径
vgg19.load_state_dict(torch.load(vgg19_weights_path))

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),  # 将图像调整为 256x256
    transforms.CenterCrop(224),  # 居中裁剪 224x224 的图像
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])

# 加载 ImageNet 类别标签
with open("imagenet_classes.json") as f:
    class_labels = json.load(f)

# 加载并预处理图像
image_path = "flash.jpeg"  # 你想要分类的图像路径
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加 batch 维度

# 将输入传递给模型并获取输出
with torch.no_grad():
    output = vgg19(input_batch)

# 使用 softmax 函数将输出转换为概率
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 打印出前5个最高概率的类别及其概率值
top5_prob, top5_catid = torch.topk(probabilities, 5) #概率值和id
for i in range(top5_prob.size(0)):
    print(f"类别：{class_labels[top5_catid[i]]}, 概率：{top5_prob[i].item()}")
