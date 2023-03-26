import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvTwoOut(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.DoubleConvTwoOut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.DoubleConvTwoOut(x), self.DoubleConvTwoOut(x)
    

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class DownSingle(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            SingleConv(in_channels, out_channels,
                       kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.encoder(x)


class DownDoubled(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            SingleConv(in_channels, out_channels,
                       kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.encoder(x), self.encoder(x)


class UpSingle(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True, kernel_size=3, stride=1, padding=1):
        super().__init__()

        if trilinear:
            self.upSingle = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.upSingle = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = SingleConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        x1 = self.upSingle(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY //
                   2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BeforeOut(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.mergeConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.mergeConv(x)


class ONet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConvTwoOut(in_channels, n_channels)
        
        self.enc1_bot = DownSingle(n_channels, 2 * n_channels, kernel_size=3, padding=1)
        self.enc2_bot = DownSingle(2 * n_channels, 4 * n_channels, kernel_size=3, padding=1)
        self.enc3_bot = DownSingle(4 * n_channels, 8 * n_channels, kernel_size=3, padding=1)
        self.enc4_bot = DownSingle(8 * n_channels, 8 * n_channels, kernel_size=3, padding=1)
        
        self.enc1_top = DownSingle(n_channels, 2 * n_channels, kernel_size=5, padding=2)
        self.enc2_top = DownSingle(2 * n_channels, 4 * n_channels, kernel_size=5, padding=2)
        self.enc3_top = DownSingle(4 * n_channels, 8 * n_channels, kernel_size=5, padding=2)
        self.enc4_top = DownSingle(8 * n_channels, 8 * n_channels, kernel_size=5, padding=2)

        self.dec1_bot = UpSingle(16 * n_channels, 4 * n_channels, kernel_size=3, padding=1)
        self.dec2_bot = UpSingle(8 * n_channels, 2 * n_channels, kernel_size=3, padding=1)
        self.dec3_bot = UpSingle(4 * n_channels, n_channels, kernel_size=3, padding=1)
        self.dec4_bot = UpSingle(2 * n_channels, n_channels, kernel_size=3, padding=1)
        
        self.dec1_top = UpSingle(16 * n_channels, 4 * n_channels, kernel_size=5, padding=2)
        self.dec2_top = UpSingle(8 * n_channels, 2 * n_channels, kernel_size=5, padding=2)
        self.dec3_top = UpSingle(4 * n_channels, n_channels, kernel_size=5, padding=2)
        self.dec4_top = UpSingle(2 * n_channels, n_channels, kernel_size=5, padding=2)
        
        self.before_out = BeforeOut(2*n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1_bot, x1_top = self.conv(x)
        
        x2_bot = self.enc1_bot(x1_bot)
        x3_bot = self.enc2_bot(x2_bot)
        x4_bot = self.enc3_bot(x3_bot)
        x5_bot = self.enc4_bot(x4_bot)
        
        x2_top = self.enc1_top(x1_top)
        x3_top = self.enc2_top(x2_top)
        x4_top = self.enc3_top(x3_top)
        x5_top = self.enc4_top(x4_top)

        mask_bot = self.dec1_bot(x5_bot, x4_bot)
        mask_bot = self.dec2_bot(mask_bot, x3_bot)
        mask_bot = self.dec3_bot(mask_bot, x2_bot)
        mask_bot = self.dec4_bot(mask_bot, x1_bot)
        
        mask_top = self.dec1_top(x5_top, x4_top)
        mask_top = self.dec2_top(mask_top, x3_top)
        mask_top = self.dec3_top(mask_top, x2_top)
        mask_top = self.dec4_top(mask_top, x1_top)
        
        mask = self.before_out(mask_bot, mask_top)
        mask = self.out(mask)
        return mask
