import torch
import torch.nn as nn
from collections import OrderedDict

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    A standard U-Net architecture for image segmentation.
    The encoder part is explicitly defined to be easily extracted.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Encoder Path ---
        # We use an OrderedDict to name the layers for easy extraction
        self.encoder = nn.Sequential(OrderedDict([
            ('enc_block1', DoubleConv(in_channels, 64)),
            ('pool1', nn.MaxPool2d(2)),
            ('enc_block2', DoubleConv(64, 128)),
            ('pool2', nn.MaxPool2d(2)),
            ('enc_block3', DoubleConv(128, 256)),
            ('pool3', nn.MaxPool2d(2)),
            ('enc_block4', DoubleConv(256, 512)),
        ]))

        # --- Bottleneck ---
        self.bottleneck_pool = nn.MaxPool2d(2)
        self.bottleneck_conv = DoubleConv(512, 1024)

        # --- Decoder Path ---
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_block4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_block3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_block2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_block1 = DoubleConv(128, 64)

        # --- Output Layer ---
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.encoder.enc_block1(x)
        x2 = self.encoder.enc_block2(self.encoder.pool1(x1))
        x3 = self.encoder.enc_block3(self.encoder.pool2(x2))
        x4 = self.encoder.enc_block4(self.encoder.pool3(x3))

        # --- Bottleneck ---
        bottleneck = self.bottleneck_conv(self.bottleneck_pool(x4))

        # --- Decoder ---
        d4 = self.upconv4(bottleneck)
        # Concatenate with skip connection
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.dec_block4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec_block3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec_block2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec_block1(d1)

        # --- Final Output ---
        return self.out_conv(d1)


def load_unet_encoder_model(in_channels=1, out_channels=2):
    """
    Instantiates the UNet model and returns only its encoder part.
    This encoder can then be used as a frozen feature extractor.

    Args:
        in_channels (int): Number of input channels for the UNet (e.g., 1 for grayscale).
        out_channels (int): Number of output classes for the UNet segmentation task.

    Returns:
        torch.nn.Sequential: The encoder part of the U-Net model.
    """
    # Instantiate the full U-Net. The number of out_channels doesn't affect
    # the encoder structure, so we can use a default value.
    full_unet_model = UNet(in_channels=in_channels, out_channels=out_channels)

    # The encoder was explicitly defined as a Sequential module, so we can
    # simply return it.
    print("Successfully extracted U-Net encoder.")
    return full_unet_model.encoder


# --- Example Usage (for demonstration) ---
if __name__ == '__main__':
    # 1. Define the model for a segmentation task (e.g., background + disk + square = 3 classes)
    print("--- Testing Full U-Net ---")
    segmentation_unet = UNet(in_channels=1, out_channels=3)
    test_image = torch.randn(1, 1, 256, 256) # (B, C, H, W)
    output_mask = segmentation_unet(test_image)
    print(f"Input shape: {test_image.shape}")
    print(f"Output mask shape: {output_mask.shape}") # Should be (1, 3, 256, 256)
    print("-" * 30)

    # 2. Create the feature extractor using the loader function
    print("--- Testing Encoder Extraction ---")
    feature_extractor = load_unet_encoder_model(in_channels=1, out_channels=3)
    # The feature_extractor is now ready. You would typically load pre-trained weights into it.
    # For example: feature_extractor.load_state_dict(torch.load('path/to/trained_weights.pth'))

    features = feature_extractor(test_image)
    print(f"Input shape: {test_image.shape}")
    print(f"Output feature map shape: {features.shape}") # Shape after the last encoder block
    print("-" * 30)
