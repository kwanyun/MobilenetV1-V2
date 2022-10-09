import torch.nn as nn

class MobilenetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobilenetV1,self).__init__()

        def Conv_DW(in_channels, out_channels,stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels,in_channels,3,stride,groups=in_channels,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels,out_channels,1,stride=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            Conv_DW(32,64),
            Conv_DW(64,128,2),
            Conv_DW(128,128),
            Conv_DW(128,256,2),
            Conv_DW(256,256),
            Conv_DW(256,512,2),
            Conv_DW(512,512),
            Conv_DW(512,512),
            Conv_DW(512,512),
            Conv_DW(512,512),
            Conv_DW(512,512),
            Conv_DW(512,1024,2),
            Conv_DW(1024,1024,1),
            nn.AvgPool2d(7),
        )
        self.fclayer = nn.Linear(1024,num_classes)

    def forward(self,x):
        x=self.model(x)
        x=x.view(-1,1024)
        return self.fclayer(x)
        

class MobilenetV1tiny(nn.Module):
    def __init__(self, num_classes):
        super(MobilenetV1tiny,self).__init__()

        def Conv_DW(in_channels, out_channels,stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels,in_channels,3,stride,padding=1,groups=in_channels,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            Conv_DW(16,32),
            Conv_DW(32,64,2),
            Conv_DW(64,64),
            Conv_DW(64,128,2),
            Conv_DW(128,128),
            Conv_DW(128,128),
            Conv_DW(128,128),
            Conv_DW(128,256,1),
            nn.AvgPool2d(4),
        )
        self.fclayer = nn.Linear(256,num_classes)

    def forward(self,x):
        x=self.model(x)
        x=x.view(-1,256)
        return self.fclayer(x)
        
