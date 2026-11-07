"""Mobile-friendly depthwise separable and squeeze-excitation blocks."""
import torch
from torch import nn

class SqueezeExcitation(nn.Module):
    def __init__(self,channels,reduction=4):
        super().__init__(); hidden=max(4,channels//reduction)
        self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.ReLU6(inplace=True),nn.Conv2d(hidden,channels,1),nn.Hardsigmoid())
    def forward(self,x): return x*self.fc(self.pool(x))

class DSConv(nn.Module):
    def __init__(self,cin,cout,stride=1,attention=True):
        super().__init__()
        self.depthwise=nn.Conv2d(cin,cin,3,stride,1,groups=cin,bias=False)
        self.dw_bn=nn.BatchNorm2d(cin); self.pointwise=nn.Conv2d(cin,cout,1,bias=False)
        self.pw_bn=nn.BatchNorm2d(cout); self.act=nn.ReLU6(inplace=True)
        self.se=SqueezeExcitation(cout) if attention else nn.Identity()
        self.skip=nn.Identity() if stride==1 and cin==cout else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x):
        y=self.act(self.dw_bn(self.depthwise(x))); y=self.pw_bn(self.pointwise(y)); y=self.se(y)
        return self.act(y+self.skip(x))

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(DSConv(cin,cout),DSConv(cout,cout))
    def forward(self,x): return self.net(x)
