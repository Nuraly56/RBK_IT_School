import torch
import torch.nn as nn
import torch.nn.functional as F

# входное изображение
x = torch.randn(1, 3, 1920, 1080)  # 1 картинка, 3 канала (RGB), размер 32x32

# сверточный слой 1
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(64)

# сверточный слой 2
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)

# shortcut путь, если каналы не совпадают
conv_shortcut = nn.Conv2d(3, 64, kernel_size=1)
bn_shortcut = nn.BatchNorm2d(64)

# основной путь (две свёртки)
out = conv1(x)
out = bn1(out)
out = F.relu(out)

out = conv2(out)
out = bn2(out)

# shortcut (обработка x, чтобы размер подходил)
shortcut = conv_shortcut(x)
shortcut = bn_shortcut(shortcut)

# сложение и ReLU
out += shortcut
out = F.relu(out)

print(out.shape)  # результат
