import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. EDGE STRUCTURE (COSINE SIMILARITY) LOSS
# ==========================================================
class EdgeLoss(nn.Module):
    """
    Calculates a loss based on the difference in gradient direction between two images.
    This loss is designed to preserve edge structure while being insensitive
    to the magnitude (intensity) of the edges. This version is corrected to
    properly handle flat (zero-gradient) areas.
    """
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        self.device = device
        
        # Sobel kernels for gradient calculation
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
                               ).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
                               ).unsqueeze(0).unsqueeze(0).to(device)
        
        # Define convolution layers with fixed, non-trainable Sobel kernels
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

    def get_gradients(self, img):
        """Helper function to compute and return gradients."""
        gx = self.conv_x(img)
        gy = self.conv_y(img)
        return gx, gy

    def forward(self, pred_img, true_img):
        """Calculates the final loss using the corrected method."""
        # Epsilon for numerical stability
        eps = 1e-6

        # Calculate gradients for the predicted image
        gx_pred, gy_pred = self.get_gradients(pred_img)
        
        # Calculate gradients for the true image
        gx_true, gy_true = self.get_gradients(true_img)

        # Calculate magnitudes (L2 norm of the gradient vectors)
        mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + eps)
        mag_true = torch.sqrt(gx_true**2 + gy_true**2 + eps)

        # --- CORRECTED LOGIC ---
        # Normalize the gradient vectors to get direction only.
        # In flat areas (where magnitude is near zero), the normalized gradients will also be zero.
        # This correctly results in a zero loss contribution from non-edge regions.
        norm_gx_pred = gx_pred / mag_pred
        norm_gy_pred = gy_pred / mag_pred
        
        norm_gx_true = gx_true / mag_true
        norm_gy_true = gy_true / mag_true
        
        # The loss is the L1 distance between the normalized gradient vectors.
        # This measures the difference in direction and is zero for flat areas.
        loss = torch.abs(norm_gx_pred - norm_gx_true) + torch.abs(norm_gy_pred - norm_gy_true)
        
        # We return the mean over all pixels, which is now correct because
        # non-edge pixels contribute zero to the sum.
        return loss.mean()

# ==========================================================
# 2. IMAGE GENERATION & SETUP
# ==========================================================
def create_circle(size, radius, intensity=1.0):
    """Creates a numpy array with a circle of a given intensity."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    mask = dist_from_center <= radius
    img[mask] = intensity
    return img

def create_square(size, width, intensity=1.0):
    """Creates a numpy array with a square of a given intensity."""
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half_width = width // 2
    img[center-half_width:center+half_width, center-half_width:center+half_width] = intensity
    return img

def to_tensor(np_array, device):
    """Converts a numpy array to a PyTorch tensor."""
    return torch.from_numpy(np_array).unsqueeze(0).unsqueeze(0).to(device)

def main():
    # Setup device and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    edge_loss_fn = EdgeLoss(device)

    # Image parameters
    IMG_SIZE = 128
    
    # --- Create the three test images ---
    # Image A: High-contrast circle
    circle_high_contrast_np = create_circle(IMG_SIZE, radius=40, intensity=1.0)
    
    # Image B: Low-contrast circle (same structure, different intensity)
    circle_low_contrast_np = create_circle(IMG_SIZE, radius=40, intensity=0.3)
    
    # Image C: High-contrast square (different structure)
    square_high_contrast_np = create_square(IMG_SIZE, width=80, intensity=1.0)

    # Convert numpy arrays to PyTorch tensors
    img_a = to_tensor(circle_high_contrast_np, device)
    img_b = to_tensor(circle_low_contrast_np, device)
    img_c = to_tensor(square_high_contrast_np, device)

    # ==========================================================
    # 3. CALCULATE LOSSES
    # ==========================================================
    with torch.no_grad():
        # Case 1: Compare images with SAME structure but DIFFERENT intensity
        loss_intensity_diff = edge_loss_fn(img_a, img_b)
        
        # Case 2: Compare images with DIFFERENT structure
        loss_structure_diff = edge_loss_fn(img_a, img_c)

    print("\n--- RESULTS (Corrected) ---")
    print(f"Loss (Same Structure, Different Intensity):  {loss_intensity_diff.item():.6f}")
    print(f"Loss (Different Structure):                {loss_structure_diff.item():.6f}")
    print("\nConclusion: The loss is now correctly near-zero for intensity changes and high for structural changes.")

    # ==========================================================
    # 4. VISUALIZATION
    # ==========================================================
    with torch.no_grad():
        # Get gradients for visualization
        gx_a, gy_a = edge_loss_fn.get_gradients(img_a)
        mag_a = torch.sqrt(gx_a**2 + gy_a**2).cpu().squeeze()

        gx_b, gy_b = edge_loss_fn.get_gradients(img_b)
        mag_b = torch.sqrt(gx_b**2 + gy_b**2).cpu().squeeze()

        gx_c, gy_c = edge_loss_fn.get_gradients(img_c)
        mag_c = torch.sqrt(gx_c**2 + gy_c**2).cpu().squeeze()

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Corrected Edge Loss: Intensity vs. Structure Invariance', fontsize=20, y=0.98)
    
    # --- Row 1: Original Images ---
    axes[0, 0].imshow(circle_high_contrast_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('A: High-Contrast Circle\n(Intensity = 1.0)', fontsize=14)
    
    axes[0, 1].imshow(circle_low_contrast_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('B: Low-Contrast Circle\n(Intensity = 0.3)', fontsize=14)
    
    axes[0, 2].imshow(square_high_contrast_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('C: High-Contrast Square\n(Different Structure)', fontsize=14)

    # --- Row 2: Gradient Magnitudes ---
    axes[1, 0].imshow(mag_a, cmap='hot')
    axes[1, 0].set_title('Gradient of A', fontsize=14)
    
    axes[1, 1].imshow(mag_b, cmap='hot')
    axes[1, 1].set_title('Gradient of B\n(Note: Fainter, but same shape)', fontsize=14)
    
    axes[1, 2].imshow(mag_c, cmap='hot')
    axes[1, 2].set_title('Gradient of C\n(Note: Different shape)', fontsize=14)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # Add text annotations for the loss values
    fig.text(0.3, 0.48, f'Loss(A, B) = {loss_intensity_diff.item():.6f}', 
             fontsize=16, ha='center', color='lime', weight='bold',
             bbox=dict(facecolor='black', alpha=0.7))
             
    fig.text(0.7, 0.48, f'Loss(A, C) = {loss_structure_diff.item():.6f}', 
             fontsize=16, ha='center', color='red', weight='bold',
             bbox=dict(facecolor='black', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()
