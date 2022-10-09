import torch.nn as nn

class Conv_Bttl(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super(Conv_Bttl, self).__init__()

        expand_channels = in_channels * t
        self.identity = stride == 1 and in_channels == out_channels

        if t == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilenetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobilenetV2,self).__init__()

        conv_bottleneck =  Conv_Bttl
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            conv_bottleneck(32,16,1,1),

            conv_bottleneck(16,24,1,6),
            conv_bottleneck(24,24,2,6),

            conv_bottleneck(24,32,1,6),
            conv_bottleneck(32,32,1,6),
            conv_bottleneck(32,32,2,6),

            conv_bottleneck(32,64,1,6),
            conv_bottleneck(64,64,1,6),
            conv_bottleneck(64,64,1,6),
            conv_bottleneck(64,64,2,6),

            conv_bottleneck(64,96,1,6),
            conv_bottleneck(96,96,1,6),
            conv_bottleneck(96,96,1,6),

            conv_bottleneck(96,160,1,6),
            conv_bottleneck(160,160,1,6),
            conv_bottleneck(160,160,2,6),

            conv_bottleneck(160,320,1,6),
            nn.Conv2d(320,1280,1,1),
            nn.AvgPool2d(7)
        )

        self.fclayer = nn.Linear(1280,num_classes)
    
    def forward(self,x):
        x=self.model(x)
        x=x.view(-1,1280)
        return self.fclayer(x)


class MobilenetV2tiny(nn.Module):
    def __init__(self, num_classes):
        super(MobilenetV2tiny,self).__init__()

        conv_bottleneck =  Conv_Bttl
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding =1 ,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            conv_bottleneck(16,8,1,1),

            conv_bottleneck(8,12,1,5),
            conv_bottleneck(12,12,1,5),

            conv_bottleneck(12,16,1,5),
            conv_bottleneck(16,16,1,5),
            conv_bottleneck(16,16,2,5),

            conv_bottleneck(16,32,1,5),
            conv_bottleneck(32,32,1,5),
            conv_bottleneck(32,32,1,5),

            conv_bottleneck(32,64,1,5),
            conv_bottleneck(64,64,1,5),
            conv_bottleneck(64,64,2,5),

            conv_bottleneck(64,128,1,5),
            nn.Conv2d(128,512,1,1),
            nn.AvgPool2d(4)
        )

        self.fclayer = nn.Linear(512,num_classes)
    
    def forward(self,x):
        x=self.model(x)
        x=x.view(-1,512)
        return self.fclayer(x)