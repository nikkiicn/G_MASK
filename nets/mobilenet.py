import torch.nn as nn



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

        self.avg = nn.AdaptiveAvgPool2d((1,1))
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
