import torch.nn as nn
import torch

class D_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.make_2_conv(3, 64)        # 64, 56, 56
        self.conv2 = self.make_2_conv(64, 128)      # 128, 28, 28
        self.conv3 = self.make_3_conv(128, 256)     # 256, 14, 14
        self.conv4 = self.make_3_conv(256, 512)     # 512, 7, 7
        self.conv5 = self.make_3_conv(512, 512)     # 512, 3, 3
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=7) # 512, 7, 7
        self.fc = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def make_2_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def make_3_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(-1)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    fake_data = torch.rand(8, 3, 112, 112)
    model = D_Model()
    output = model(fake_data)
    print(output)