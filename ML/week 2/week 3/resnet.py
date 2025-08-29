import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(1, 3, 1920, 1080)  

conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(64)

conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)

conv_shortcut = nn.Conv2d(3, 64, kernel_size=1)
bn_shortcut = nn.BatchNorm2d(64)

out = conv1(x)
out = bn1(out)
out = F.relu(out)

out = conv2(out)
out = bn2(out)

shortcut = conv_shortcut(x)
shortcut = bn_shortcut(shortcut)

out += shortcut
out = F.relu(out)

print(out.shape)  
