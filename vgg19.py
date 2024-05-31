import torch
import torchvision.models as models

# 下载预训练的 VGG19 模型
vgg19 = models.vgg19(weights=True)

# 输出 VGG19 模型的结构
print(vgg19)

# 保存模型的权重
torch.save(vgg19.state_dict(), 'vgg19.pth')
