import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# The CSDN_Tem block remains the same as it's an efficient conv block
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=3,
            stride=1, padding=1, groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=1, padding=0, groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class AdaIN(nn.Module):
    """Adaptive Instance Normalization Layer."""
    def __init__(self):
        super().__init__()

    def forward(self, content_features, style_features):
        """
        Args:
            content_features (torch.Tensor): The image features, shape [B, C, H, W].
            style_features (torch.Tensor): The style vector from the histogram, shape [B, 2*C].
        """
        # Instance normalization removes the original style
        content_mean = content_features.mean(dim=[2, 3], keepdim=True)
        content_std = content_features.std(dim=[2, 3], keepdim=True) + 1e-8
        normalized_features = (content_features - content_mean) / content_std

        # Split the style vector into scale (gamma) and bias (beta)
        # Reshape to [B, C, 1, 1] for broadcasting
        style_scale, style_bias = style_features.chunk(2, dim=1)
        style_scale = style_scale.unsqueeze(-1).unsqueeze(-1)
        style_bias = style_bias.unsqueeze(-1).unsqueeze(-1)

        # Apply the new style
        return normalized_features * style_scale + style_bias

class GuidanceEncoder(nn.Module):
    """
    Encodes a 2D guidance map into a 1D style vector using convolutions.
    """
    def __init__(self, input_channels=1, num_features=128, output_size=768):
        super(GuidanceEncoder, self).__init__()
        # A simple CNN to downsample the map and extract features
        self.encoder = nn.Sequential(
            CSDN_Tem(input_channels, num_features // 2), # Ex: [B, 1, H, W] -> [B, 64, H, W]
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2), # Downsample
            CSDN_Tem(num_features // 2, num_features),   # Ex: [B, 64, H/2, W/2] -> [B, 128, H/2, W/2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2), # Downsample
            CSDN_Tem(num_features, num_features),        # Ex: [B, 128, H/4, W/4] -> [B, 128, H/4, W/4]
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1) # Flatten spatial dims -> [B, 128, 1, 1]
        )
        # Final linear layer to get the correct style vector size
        self.fc = nn.Linear(num_features, output_size)

    def forward(self, guidance_map):
        features = self.encoder(guidance_map)
        features_flat = features.view(features.size(0), -1) # Flatten to [B, 128]
        style_vector = self.fc(features_flat)
        return style_vector
    
    
class MRI_Synthesis_Net(nn.Module):
    """
    A residual-based network for MRI contrast synthesis, initialized
    as an identity function.
    """
    def __init__(self, scale_factor=1, num_hist_bins=256):
        super(MRI_Synthesis_Net, self).__init__()
        self.scale_factor = scale_factor
        if self.scale_factor > 1:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)


        intermediate_feature_size = 512  
        self.number_f = 128 # Number of features
        style_output_size = 3 * (2 * self.number_f)
        self.guidance_encoder = GuidanceEncoder(
            input_channels=1,
            num_features=self.number_f,
            output_size=intermediate_feature_size # This should be 512
        )
        
        
        self.histogram_processor = nn.Sequential(
            nn.Linear(num_hist_bins, intermediate_feature_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(intermediate_feature_size, intermediate_feature_size)
        )

        style_output_size = 3 * (2 * self.number_f) # 3 AdaIN layers * (scale + bias) * channels
        self.fusion_head = nn.Linear(intermediate_feature_size * 2, style_output_size)


        # --- AdaIN Layers ---
        self.adain1 = AdaIN()
        self.adain2 = AdaIN()
        self.adain3 = AdaIN()

        # --- Image Processing Path (U-Net like) ---
        self.e_conv0 = CSDN_Tem(1, self.number_f)
        self.e_conv1 = CSDN_Tem(self.number_f, self.number_f)
        self.e_conv2 = CSDN_Tem(self.number_f, self.number_f)
        self.e_conv3 = CSDN_Tem(self.number_f, self.number_f)
        self.e_conv4 = CSDN_Tem(self.number_f, self.number_f)
        self.e_conv5 = CSDN_Tem(self.number_f*2, self.number_f)
        self.e_conv6 = CSDN_Tem(self.number_f*2, self.number_f)
        self.e_conv7 = CSDN_Tem(self.number_f*2, 1)
        
        self._init_weights()
        self._init_identity()

    def _init_weights(self):
        """Initializes network weights using Kaiming Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Apply Kaiming initialization for convolutional layers
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                # Initialize bias to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _init_identity(self):
        """Initializes the weights and biases of the final layer to zero."""
        nn.init.normal_(self.e_conv7.depth_conv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.e_conv7.depth_conv.bias, mean=0.0, std=0.02)
        nn.init.normal_(self.e_conv7.point_conv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.e_conv7.point_conv.bias, mean=0.0, std=0.02)

    def forward(self, x, target_hist, guidance_map):
        x_original = x
        x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False) if self.scale_factor > 1 else x

        # --- Get Style Features from Histogram ---
        hist_features = self.histogram_processor(target_hist)
        guidance_features = self.guidance_encoder(guidance_map)
        
        combined_features = torch.cat([hist_features, guidance_features], dim=1)
        style_features = self.fusion_head(combined_features)


        # Split the style vector for each AdaIN layer
        style1, style2, style3 = style_features.split(2 * self.number_f, dim=1)
        
        # --- U-Net Path with AdaIN Injection ---
        f0 = F.leaky_relu(self.e_conv0(x_down), negative_slope=0.2)
        
        # Apply first style injection
        f1_conv = self.e_conv1(f0)
        f1_styled = self.adain1(f1_conv, style1)
        f1 = F.leaky_relu(f1_styled, negative_slope=0.2)
        
        # Apply second style injection
        f2_conv = self.e_conv2(f1)
        f2_styled = self.adain2(f2_conv, style2)
        f2 = F.leaky_relu(f2_styled, negative_slope=0.2)
        
        f3_conv = self.e_conv3(f2)
        f3_styled = self.adain3(f3_conv, style3)
        f3 = F.leaky_relu(f3_styled, negative_slope=0.2)
        
        # Continue the rest of the path as normal
        f4 = F.leaky_relu(self.e_conv4(f3), negative_slope=0.2)
        f5 = F.leaky_relu(self.e_conv5(torch.cat([f3, f4], 1)), negative_slope=0.2)
        f6 = F.leaky_relu(self.e_conv6(torch.cat([f2, f5], 1)), negative_slope=0.2)
        
        residual = self.e_conv7(torch.cat([f1, f6], 1))

        if self.scale_factor > 1:
            residual = self.upsample(residual)

        # residual = 2*torch.tanh(residual)

        output_image = x_original + residual #+ (torch.sigmoid(residual) * 4 - 2)
        # output_image = x_original + (torch.tanh(residual) * 2)
        # print("RES :", torch.min(residual), torch.max(residual), torch.min(torch.tanh(residual) * 2), torch.max(torch.tanh(residual) * 2))
        # return torch.clamp(output_image, -1, 1), torch.clamp(torch.sigmoid(residual) * 2 - 1, -1, 1)
        return torch.tanh(output_image), residual/2