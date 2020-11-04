from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

# BLOCKS
class ConvBNRelu6(nn.Sequential):
    def __init__(self, in_, out, kernel, stride, padding=0, groups=1):
        super(ConvBNRelu6, self).__init__(
            nn.Conv2d(in_channels=in_, out_channels=out, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out),
            nn.ReLU6(inplace=True)
        )

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=128, c=1, t=1, s=1):
        super(BottleneckBlock, self).__init__()
        tk = int(round(in_channels*t))
        self.use_res = (s==1) and (in_channels==c)
        layers = []
        if t != 1:
            layers.append(ConvBNRelu6(in_=in_channels, out=tk, kernel=(1,1), stride=(1,1)))
        layers += [
            ConvBNRelu6(in_=tk, out=tk, kernel=(3,3), stride=(s,s), padding=(1,1), groups=tk), #depthwise
            nn.Conv2d(in_channels=tk, out_channels=c, kernel_size=(1,1)),
            nn.BatchNorm2d(c)
        ]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

# for packaging block construction data more neatly
class Sequence(object):
  def __init__(self, **kwargs):
      for k,v in kwargs.items():
          setattr(self,k,v)

# network
class Model(torch.nn.Module):
    def __init__(self, input_dim=256, img_depth=3, sequence_list=[], dropout=.2, fc_hidden_dims=500):
        super(Model, self).__init__()
        layers = []
        inp_size = img_depth
        for seq_dict in sequence_list:
          seq = Sequence(**seq_dict)
          for i in range(seq.n):
            if i > 0 and seq.s > 1:
                seq.s = 1
            if seq.op == 'conv2d':
                layers.append(ConvBNRelu6(in_=inp_size, out=seq.c, kernel=seq.kernel, stride=seq.s, padding=seq.kernel//2))
            elif seq.op == 'bottleneck':
                layers.append(BottleneckBlock(in_channels=inp_size, c=seq.c, t=seq.t, s=seq.s))
            elif seq.op == 'avgpool':
                layers.append(nn.AvgPool2d(input_dim))
            inp_size = seq.c
            input_dim //= (input_dim if seq.s==-1 else seq.s)
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Linear(inp_size, fc_hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dims, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def get_pretrained_model(filepath="mixedTraining2.pt"):
    model_data = torch.load(filepath)
    net = Model(input_dim=256, img_depth=1, sequence_list=model_data['sequence_list'], fc_hidden_dims=model_data['fc_hidden_dims'])
    net.load_state_dict(model_data['state_dict'])
    return net

def predict(image_filepath, use_cuda=False):
    device = 'cuda:0' if use_cuda else 'cpu'
    net = get_pretrained_model().eval().to(device)
    image = Image.open(image_filepath).convert("L")
    transform = ToTensor()
    inp = transform(image).unsqueeze(0).to(device)
    output = net(inp)
    return output.detach().item()