import torch
import torch.nn as nn

class G_Model(nn.Module):           # Customize of U-NET model
    def __init__(self):
        super().__init__()
        self.conv_1 = self.make_block(3, 64)            # 64, 112, 112
        self.pool_1 = nn.MaxPool2d(kernel_size=2)       # 64, 56, 56 

        self.conv_2 = self.make_block(64, 128)          # 128, 56, 56
        self.pool_2 = nn.MaxPool2d(kernel_size=2)       # 128, 28, 28

        self.conv_3 = self.make_block(128, 256)         # 256, 28, 28
        self.pool_3 = nn.MaxPool2d(kernel_size=2)       # 256, 14, 14

        self.conv_4 = self.make_block(256, 512)         # 512, 14, 14
        self.pool_4 = nn.MaxPool2d(kernel_size=2)       # 512, 7, 7

        self.conv_5 = self.make_block(512, 1024)         # 1024, 7, 7

        self.up_6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)      # 1024, 28, 28
        self.conv_6 = self.make_block(1024, 512)             # 512, 28, 28                            

        self.up_7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)      # 512, 28, 28
        self.conv_7 = self.make_block(512, 256)              # 256, 28, 28                                           

        self.up_8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)      # 256, 56, 56
        self.conv_8 = self.make_block(256, 128)              # 128, 56, 56                                       

        self.up_9 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)      # 128, 112, 112
        self.conv_9 = self.make_block(128, 64)                # 64, 112, 112

        self.conv_10 = self.make_block(64, 3)                    # 3, 112, 112   
        self.sigmoid = nn.Sigmoid()
    
    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        conv1 = self.conv_1(x)
        pool1 = self.pool_1(conv1)

        conv2 = self.conv_2(pool1)
        pool2 = self.pool_2(conv2)

        conv3 = self.conv_3(pool2)
        pool3 = self.pool_3(conv3)

        conv4 = self.conv_4(pool3)
        pool4 = self.pool_4(conv4)

        conv5 = self.conv_5(pool4)

        up6 = self.up_6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv_6(merge6)

        up7 = self.up_7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv_7(merge7)
        
        up8 = self.up_8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv_8(merge8)

        up9 = self.up_9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv_9(merge9)

        conv10 = self.conv_10(conv9)
        output = self.sigmoid(conv10)
        return output
    
if __name__ == "__main__":
    model = G_Model()
    fake_date = torch.rand(8, 3, 112, 112)
    output = model(fake_date)
    print(output.shape)