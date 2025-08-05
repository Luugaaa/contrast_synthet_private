import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MRI_Synthesis_Net(nn.Module):
    """
    A residual-based network for MRI contrast synthesis, initialized
    as an identity function.
    """
    def __init__(self, scale_factor=1):
        super(MRI_Synthesis_Net, self).__init__()
        self.scale_factor = scale_factor
        if self.scale_factor > 1:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        number_f = 128 # Number of features
        
        # U-Net like encoder-decoder structure
        self.e_conv0 = CSDN_Tem(1, number_f) # Takes 1-channel MRI
        self.e_conv1 = CSDN_Tem(number_f, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f*2, number_f)
        self.e_conv6 = CSDN_Tem(number_f*2, number_f)
        
        # Final layer outputs a 1-channel residual image
        self.e_conv7 = CSDN_Tem(number_f*2, 1)

        # Initialize the final layer to output zeros
        
        self._init_weights()
        # self._init_identity()

    def _init_weights(self):
        """Initializes network weights using Kaiming Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Apply Kaiming initialization for convolutional layers
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                # Initialize bias to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _init_identity(self):
        """Initializes the weights and biases of the final layer to zero."""
        nn.init.normal_(self.e_conv7.depth_conv.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.e_conv7.depth_conv.bias, mean=0.0, std=0.15)
        nn.init.normal_(self.e_conv7.point_conv.weight, mean=0.0, std=0.15)
        nn.init.normal_(self.e_conv7.point_conv.bias, mean=0.0, std=0.15)

    def forward(self, x):
        """
        x: The input MRI (e.g., T1-weighted)
        """
        x_original = x # Keep the original full-resolution input
        
        # --- Process at lower resolution for efficiency ---
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)

        # --- Feature Extraction (U-Net style) ---
        f0 = F.leaky_relu(self.e_conv0(x_down), negative_slope=0.2)
        f1 = F.leaky_relu(self.e_conv1(f0), negative_slope=0.2)
        f2 = F.leaky_relu(self.e_conv2(f1), negative_slope=0.2)
        f3 = F.leaky_relu(self.e_conv3(f2), negative_slope=0.2)
        f4 = F.leaky_relu(self.e_conv4(f3), negative_slope=0.2)
        f5 = F.leaky_relu(self.e_conv5(torch.cat([f3, f4], 1)), negative_slope=0.2)
        f6 = F.leaky_relu(self.e_conv6(torch.cat([f2, f5], 1)), negative_slope=0.2)
        
        # --- Predict the Residual ---
        # The output of the final layer is the residual (delta) image
        residual = self.e_conv7(torch.cat([f1, f6], 1))

        # --- Upsample residual and add to original image ---
        if self.scale_factor > 1:
            residual = self.upsample(residual)

        # The final synthesized image is the original + the learned residual
        output_image = x_original + (torch.sigmoid(residual)*4-2)
        # output_image = residual
        
        # It's good practice to clamp the output to the expected image range (e.g., 0 to 1)
        return torch.clamp(output_image, -1, 1), torch.clamp(torch.sigmoid(residual)*2-1, -1, 1)