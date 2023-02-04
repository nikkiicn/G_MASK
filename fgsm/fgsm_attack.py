import torch
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.hub import load_state_dict_from_url
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from mf_attack.fgsm.f_net import Facenet
from facenet_pytorch.utils.utils import show_config
from mf_attack.nets.facenet import Facenet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 三个输入，定义如下：
# eplisions:要用于运行的epsilon值列表。在列表中保持0很重要，因为它代表了原始测试集上的模型性能。
# 同样，从直觉上说， 我们认为ε越大，扰动越明显，但是从降低模型准确性的角度来看，攻击越有效，降低了模型的准确性。
# 由于数据的范围是 [0,1]，则epsilon值不得超过1
# pretrained_model：通向预先训练过的MNIST模型的路径，该模型是用 pytorch/examples/mnist 。为了简单起见，请在这里下载经过预先训练的模型。
# use_cuda：布尔标志，使用CUDA(如果需要和可用的话)
# 输入
"""epsilons：用于运行的epsilon值列表"""
epsilons = [0, .05, .1, .15, .2, .25, .3]
# 预训练模型
pretrained_model = "../model_data/facenet_mobilenet.pth"
backbone="mobilenet"
use_cuda = True

# 声明LFW测试数据集和数据加载
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ])),
#     batch_size=1, shuffle=True)
datasets = '../../data/FaceRecognition/securityAI/securityAI_round1_images'
dataloader = torch.utils.data.DataLoader(datasets, batch_size=64, shuffle=True)

# 定义我们正在使用的设备
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

# 深度可分离卷积块
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        # 深度可分离卷积
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),
        # 1X1普通卷积
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 224,224,3 -> 112,112,32
            conv_bn(3, 32, 2),
            # 112,112,32 -> 112,112,64
            conv_dw(32, 64, 1),

            # 112,112,64 -> 56,56,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 56,56,128 -> 28,28,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 28,28,256 -> 14,14,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 14,14,512 -> 7,7,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_store",
                                                progress=True)
            self.model.load_state_dict(state_dict)
            # state_dict = './mf_attack/model_store/backbone_weights_of_mobilenetv1.pth'
            # self.model.load_state_dict(torch.load(state_dict))

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_store",
                                                progress=True)
            # state_dict = '../mf_attack/model_store/backbone_weights_of_inception_resnetv1.pth'
            self.model.load_state_dict(torch.load(state_dict))

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train",
                 pretrained=False):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        if mode == 'predict':
            x = self.backbone(x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            # 训练时作用，防止模型过拟合
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x


class FaceNet(object):
    _defaults = {
        "model_path": "../model_data/facenet_mobilenet.pth",
        "backbone": "mobilenet",
        "cuda": True,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    def generate(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('model loaded : {}'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

model = FaceNet()

# 加载已经预训练的模型
# model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
# model.eval()
# 区别model.eval()与model.train()

# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0, 1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image

# 测试函数
def test(model, device, test_loader, epsilon):
    # 精度计数器
    correct = 0
    adv_examples = []
    # 循环遍历测试集中的所有示例
    for data, target, in test_loader:
        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)
        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True
        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue
        # 计算损失
        loss = F.nll_loss(output, target)
        # 模型参数的梯度置为0
        model.zero_grad()
        # 计算后向传递模型的梯度
        loss.backward()
        # 收集data_grad
        data_grad = data.grad.data
        # 唤醒FGSM进行攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # 重新分类受扰乱的图像
        output = model(perturbed_data)
        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # 稍后保存一些用于可视化的示例
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    # 计算这个epsilon的最终准确度
    final_acc = correct/float(len((test_loader)))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples

# 运行攻击
accuracies = []
examples = []
# 对每个epsilons运行测试
for eps in epsilons:
    acc, ex = test(model, device, dataloader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# 在每个epsilon上绘制几个对抗样本的例子
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()