import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19
import math
import time
from datetime import datetime


def calculate_psnr(sr_tensor, hr_tensor):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    sr_tensor和hr_tensor应该是范围在[0,1]的张量
    """
    mse = nn.MSELoss()(sr_tensor, hr_tensor)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.criterion(sr_features, hr_features)


# ResidualBlock和ResSRCNN类保持不变
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResSRCNN(nn.Module):
    def __init__(self, num_channels=3, num_residual_blocks=6):
        super(ResSRCNN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.mid_layer = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU()
        )

        self.output_layer = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.mid_layer(out)
        out = self.output_layer(out)
        out = out + residual
        return torch.clamp(out, 0.0, 1.0)


class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=2):
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.image_files = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_img = Image.open(hr_path).convert('RGB')

        if self.patch_size:
            if hr_img.width < self.patch_size or hr_img.height < self.patch_size:
                new_size = max(self.patch_size, self.patch_size)
                hr_img = hr_img.resize((new_size, new_size), Image.BICUBIC)

            x = torch.randint(0, hr_img.width - self.patch_size + 1, (1,)).item()
            y = torch.randint(0, hr_img.height - self.patch_size + 1, (1,)).item()
            hr_img = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        lr_size = (hr_img.size[0] // self.scale, hr_img.size[1] // self.scale)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)

        return lr_tensor, hr_tensor



def test_ressrcnn(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.clamp(output.squeeze(0).cpu(), 0.0, 1.0)
        output_img = transforms.ToPILImage()(output)
    return output_img


if __name__ == '__main__':
    # 设置参数
    hr_dir = r'D:\论文\SRGAN-master\srgan\DIV2K\DIV2K_train_HR'
    patch_size = 96
    batch_size = 16
    num_epochs = 100
    save_dir = 'model_checkpoints'

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建数据集和数据加载器
    dataset = SRDataset(hr_dir, patch_size=patch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = ResSRCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型进行测试
    model = ResSRCNN()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 测试图像
    test_image_path = 'test_image.png'
    result = test_ressrcnn(model, test_image_path, device)
    result.save('super_resolution_result-1.2.png')