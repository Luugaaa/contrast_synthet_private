import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import glob
import wandb
import random
import numpy as np
import math 


from unet import UNet
from prototype_4_mri_model import MRI_Synthesis_Net
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import io

def plot_histograms_to_image(input_hist, target_hist, gen_hist, num_bins):
    """
    Generates an image of overlapping histograms using matplotlib.

    Args:
        input_hist (torch.Tensor): Histogram data for the input image.
        target_hist (torch.Tensor): Histogram data for the target distribution.
        gen_hist (torch.Tensor): Histogram data for the generated image.
        num_bins (int): The number of bins in the histograms.

    Returns:
        PIL.Image: An image of the plot.
    """
    # Move tensors to CPU and convert to numpy
    input_hist = input_hist.cpu().numpy()
    target_hist = target_hist.cpu().numpy()
    gen_hist = gen_hist.cpu().numpy()

    # Normalize histograms to represent probability distributions for better comparison
    input_hist = input_hist / (input_hist.sum() + 1e-8)
    target_hist = target_hist / (target_hist.sum() + 1e-8)
    gen_hist = gen_hist / (gen_hist.sum() + 1e-8)

    # Define bin centers for the x-axis
    bin_centers = np.linspace(0.0, 1.0, num_bins)

    # Create plot
    plt.ioff() # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(5, 4)) # Create a figure and axes

    # Plot each histogram as a line with some transparency
    ax.plot(bin_centers, input_hist, label='Input', color='blue', alpha=0.7)
    ax.plot(bin_centers, target_hist, label='Target', color='red', alpha=0.7, linestyle='--')
    ax.plot(bin_centers, gen_hist, label='Generated', color='green', alpha=0.7)
    
    # Style the plot
    ax.set_title('Histogram Comparison')
    ax.set_xlabel('Pixel Intensity (Normalized)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Create a PIL Image from the buffer
    img = Image.open(buf)
    
    # Clean up
    plt.close(fig)
    
    return img

class ComplexContrastDataset(Dataset):
    """
    Loads an input image from set1 and a reference image from set2.
    The datasets are unpaired.
    """
    def __init__(self, root_dir, transform=None, set_to_load=None):
        self.transform = transform
        self.set_to_load = set_to_load
        self.set1_paths = sorted(glob.glob(os.path.join(root_dir, "set1", "*.png")))
        self.set2_paths = sorted(glob.glob(os.path.join(root_dir, "set2", "*.png")))
        # Shuffle the reference set to ensure random pairing
        random.shuffle(self.set2_paths)
        assert len(self.set1_paths) > 0, "No images found in data_complex/set1"
        assert len(self.set2_paths) > 0, "No images found in data_complex/set2"

    def __len__(self):
        # The length is determined by the smaller of the two sets
        return min(len(self.set1_paths), len(self.set2_paths))

    def __getitem__(self, idx):
        # Use modulo to wrap around the shorter list if sets are different sizes
        set2_idx = idx % len(self.set2_paths)
        
        input_img_pil = Image.open(self.set1_paths[idx]).convert("L")
        ref_img_pil = Image.open(self.set2_paths[set2_idx]).convert("L")

        if self.transform:
            input_tensor = self.transform(input_img_pil)
            ref_tensor = self.transform(ref_img_pil)
            
        if self.set_to_load == "set1" :
            return input_tensor
        elif self.set_to_load == "set2" :
            return ref_tensor
        
        return input_tensor, ref_tensor
    
    
class DifferentiableHistogram(nn.Module):
    """
    Calculates a differentiable histogram for a batch of images using a
    triangular kernel for soft binning (a simplified form of KDE).
    This allows gradients to flow back to the image generator.
    """
    def __init__(self, num_bins=256, min_val=0.0, max_val=1.0):
        super(DifferentiableHistogram, self).__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Create bin centers that are not trainable parameters
        # The shape will be (1, 1, 1, num_bins) for broadcasting
        bin_centers = torch.linspace(min_val, max_val, num_bins).view(1, 1, -1)
        self.register_buffer("bin_centers", bin_centers)

        # Calculate bin width
        self.bin_width = (max_val - min_val) / (num_bins - 1)

    def forward(self, images_batch):
        """
        Args:
            images_batch (torch.Tensor): A batch of images of shape.
                                         Expected to be in the range [min_val, max_val].
        Returns:
            torch.Tensor: A tensor of shape containing the
                          differentiable histogram for each image.
        """
        B, C, H, W = images_batch.shape
        # Reshape for broadcasting:
        images_flat = images_batch.contiguous().view(B, -1, 1)

        # Calculate distance of each pixel to each bin center
        # Broadcasting: images_flat - bin_centers [1, 1, num_bins]
        # -> dist
        dist = torch.abs(images_flat - self.bin_centers)

        # Apply triangular kernel
        # This is a "soft" assignment. A pixel will have non-zero weight
        # for bins it's close to (within one bin_width).
        # The relu ensures weights are non-negative.
        weights = torch.relu(1 - dist / self.bin_width)

        # Sum the weights over all pixels for each image in the batch
        # This aggregates the contributions to each bin.
        # The result is a histogram of shape
        hist = torch.sum(weights, dim=1)

        return hist


class DifferentiableWassersteinLoss(nn.Module):
    def __init__(self):
        super(DifferentiableWassersteinLoss, self).__init__()
        self.l1_loss_fn = nn.L1Loss()

    def forward(self, gen_hist_batch, target_hist_batch):
        # Normalize both histograms to get probability distributions
        gen_hist_p = gen_hist_batch / (gen_hist_batch.sum(dim=1, keepdim=True) + 1e-8)
        target_hist_p = target_hist_batch / (target_hist_batch.sum(dim=1, keepdim=True) + 1e-8)

        # Calculate the Cumulative Distribution Functions (CDFs)
        gen_cdf = torch.cumsum(gen_hist_p, dim=1)
        target_cdf = torch.cumsum(target_hist_p, dim=1)
        
        l1_loss = self.l1_loss_fn(gen_hist_p, target_hist_p)

        # The 1D Wasserstein distance is the L1 distance between the CDFs
        return torch.mean(torch.abs(gen_cdf - target_cdf)) + l1_loss


class WassersteinHistogramLoss(nn.Module):
    """
    Calculates the 1D Wasserstein Distance (or Earth Mover's Distance) between
    the intensity distributions of a batch of generated images and a batch of
    target histograms.

    This loss is particularly well-suited for sparse, "spiky" histograms as it
    considers the distance between bins, not just the difference in their counts.
    """
    def __init__(self, num_bins=256):
        super(WassersteinHistogramLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, generated_img_batch, target_hist_batch):
        # 1. Calculate the histogram for the generated image batch
        gen_hist_batch = calculate_batch_histograms(generated_img_batch, self.num_bins)

        # 2. Normalize both histograms to get probability distributions (sum to 1)
        gen_hist_p = gen_hist_batch / (gen_hist_batch.sum(dim=1, keepdim=True) + 1e-8)
        target_hist_p = target_hist_batch / (target_hist_batch.sum(dim=1, keepdim=True) + 1e-8)

        # 3. Calculate the Cumulative Distribution Functions (CDFs)
        # The CDF at bin 'i' is the sum of all probabilities up to and including 'i'.
        gen_cdf = torch.cumsum(gen_hist_p, dim=1)
        target_cdf = torch.cumsum(target_hist_p, dim=1)

        # 4. The 1D Wasserstein distance is the L1 distance between the CDFs.
        # This measures the "area" between the two cumulative distribution curves.
        return torch.mean(torch.abs(gen_cdf - target_cdf))
    
    
# ==========================================================
# EDGE STRUCTURE (COSINE SIMILARITY) LOSS
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
        def edge_direction_loss(a, b):
            gx_a, gy_a = self.get_gradients(a)
            gx_b, gy_b = self.get_gradients(b)
            
            mag_a = torch.sqrt(gx_a**2 + gy_a**2 + 1e-6)
            mag_b = torch.sqrt(gx_b**2 + gy_b**2 + 1e-6)

            norm_gx_a = gx_a / mag_a
            norm_gy_a = gy_a / mag_a
            norm_gx_b = gx_b / mag_b
            norm_gy_b = gy_b / mag_b

            dot = norm_gx_a * norm_gx_b + norm_gy_a * norm_gy_b
            angle_loss = 1.0 - dot.clamp(-1.0, 1.0)
            return angle_loss.mean()

        # Compute loss on normal and inverted predictions
        normal_loss = edge_direction_loss(pred_img, true_img)
        inverted_loss = edge_direction_loss(1.0 - pred_img, true_img)

        # Choose the lower of the two
        return (normal_loss + inverted_loss)/2
    

# ==========================================================
# RANGE LOSS
# ==========================================================
class RangeLoss(nn.Module):
    """
    Penalizes pixel values that fall outside the desired [-1, 1] range.
    """
    def __init__(self, target_min=-1.0, target_max=1.0):
        super(RangeLoss, self).__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, image_batch):
        # Penalize values that are GREATER than the max target
        # relu(x - 1) is 0 if x <= 1, and positive otherwise.
        loss_above = torch.nn.functional.relu(image_batch - self.target_max)

        # Penalize values that are LESS than the min target
        # relu(-1 - x) is 0 if x >= -1, and positive otherwise.
        loss_below = torch.nn.functional.relu(self.target_min - image_batch)

        # The total loss is the average penalty across all pixels
        return torch.mean(loss_above + loss_below)
    
    

class TargetHistogramLoss(nn.Module):
    """
    Calculates L1 loss between a generated image's histogram and a target histogram.
    """
    def __init__(self, num_bins=256):
        super(TargetHistogramLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, generated_img_batch, target_hist_batch):
        # 1. Calculate the full histogram for the generated batch
        gen_hist_batch = calculate_batch_histograms(generated_img_batch, self.num_bins)
        
        # 2. Normalize both sets of histograms to get probability distributions
        gen_hist_p = gen_hist_batch / (gen_hist_batch.sum(dim=1, keepdim=True) + 1e-8)
        target_hist_p = target_hist_batch / (target_hist_batch.sum(dim=1, keepdim=True) + 1e-8)
        
        # 3. Calculate L1 loss for the entire batch at once
        return torch.nn.functional.l1_loss(gen_hist_p, target_hist_p)

# ==========================================================
# HISTOGRAM HELPER FUNCTIONS (REFINED)
# ==========================================================

def calculate_batch_histograms(images_batch, num_bins, mask=None):
    """
    Calculates histograms for a whole batch of images in a vectorized manner.
    If a mask is provided, only pixels where the mask is True are counted.

    Args:
        images_batch (torch.Tensor): A batch of images of shape [B, C, H, W].
        num_bins (int): The number of bins for the histogram.
        mask (torch.Tensor, optional): A boolean tensor of shape [B, C, H, W].

    Returns:
        torch.Tensor: A tensor of shape [B, num_bins] containing the histogram for each image.
    """
    B, C, H, W = images_batch.shape
    device = images_batch.device

    # Denormalize once at the beginning
    images_denorm = images_batch * 0.5 + 0.5
    
    # If no mask is provided, create a mask that includes all pixels
    if mask is None:
        mask = torch.ones_like(images_batch, dtype=torch.bool)
    
    # Vectorized Binning
    bin_indices = (images_denorm * (num_bins - 1)).long()
    batch_offsets = torch.arange(B, device=device) * num_bins
    offset_indices = bin_indices + batch_offsets.view(B, 1, 1, 1)

    # Use scatter_add_ on the flattened vector
    flat_hist = torch.zeros(B * num_bins, device=device)
    flat_indices_to_scatter = offset_indices[mask]
    flat_hist.scatter_add_(0, flat_indices_to_scatter, torch.ones_like(flat_indices_to_scatter, dtype=flat_hist.dtype))

    return flat_hist.view(B, num_bins)

def generate_shuffled_histogram(input_images, num_bins=256, dark_threshold=0.05, chunk_threshold_ratio=0.1):
    """
    Generates a target histogram by segmenting the distribution into chunks and
    shuffling them, preserving the internal structure of peaks.

    This method identifies "valleys" (bins with low counts) and uses them as
    cut-points. The resulting segments (chunks) are then shuffled.

    Args:
        input_images (torch.Tensor): The batch of input images, normalized to [-1, 1].
        num_bins (int): The number of bins for the histogram.
        dark_threshold (float): Pixel values below this (in [0,1] space) have their
                                histogram counts fixed and are not shuffled.
        chunk_threshold_ratio (float): A ratio of the mean non-zero bin count. Bins
                                       with counts below this threshold are
                                       considered "valleys" or split-points.
                                       Lower values lead to fewer, larger chunks.
    """
    B = input_images.shape[0]
    device = input_images.device
    images_denorm = input_images * 0.5 + 0.5

    # 1. Calculate histograms for the two parts of the image (same as before)
    hist_below_thresh = calculate_batch_histograms(input_images, num_bins, mask=(images_denorm < dark_threshold))
    hist_above_thresh = calculate_batch_histograms(input_images, num_bins, mask=(images_denorm >= dark_threshold))

    # 2. Segment and Shuffle the 'Above Threshold' Part
    shuffled_hist_above = torch.zeros_like(hist_above_thresh)
    for i in range(B):
        original_hist = hist_above_thresh[i]

        # Find non-zero counts to determine if there's anything to shuffle
        non_zero_counts = original_hist[original_hist > 0]
        if non_zero_counts.nelement() == 0:
            shuffled_hist_above[i] = original_hist
            continue

        # --- NEW LOGIC: Segment into chunks ---
        
        # 2a. Define the valley threshold dynamically based on the histogram's content.
        # This makes it robust to images with different contrast levels.
        valley_threshold = non_zero_counts.mean() * chunk_threshold_ratio

        # 2b. Find indices of all bins that are valleys (our split points).
        split_indices = torch.where(original_hist <= valley_threshold)[0]
        
        # 2c. Create a list of chunk boundaries. Start with 0, end with num_bins,
        # and include all the split points in between.
        boundaries = torch.cat([torch.tensor([0], device=device), split_indices, torch.tensor([num_bins], device=device)])
        boundaries = torch.unique(boundaries) # Remove duplicates

        # 2d. Extract the chunks based on the boundaries.
        chunks = []
        for j in range(len(boundaries) - 1):
            start_idx = boundaries[j]
            end_idx = boundaries[j+1]
            if start_idx < end_idx: # Ensure chunk is not empty
                chunks.append(original_hist[start_idx:end_idx])

        # 2e. Shuffle the list of chunks.
        random.shuffle(chunks)

        # 2f. Reconstruct the histogram by concatenating the shuffled chunks.
        if chunks:
            new_hist = torch.cat(chunks)
            shuffled_hist_above[i] = new_hist
        else:
            # Fallback in case no valid chunks were created
            shuffled_hist_above[i] = original_hist


    # 3. Combine the fixed and shuffled parts (same as before)
    target_histograms = hist_below_thresh + shuffled_hist_above

    return target_histograms

def rebin_histogram(
    source_hist: torch.Tensor,
    min_vals_hist: torch.Tensor,
    image_ranges_hist: torch.Tensor,
    source_range=(-1.0, 1.0),
    target_range=(0.0, 1.0)
) -> torch.Tensor:
    """
    Rescales a histogram by rebinning its counts to match image normalization.

    Args:
        source_hist (torch.Tensor): The histogram to rescale, shape (B, num_bins).
        min_vals_hist (torch.Tensor): The minimum values for normalization, shape (B,).
        image_ranges_hist (torch.Tensor): The ranges (max-min) for normalization, shape (B,).
        source_range (tuple): The original pixel value range, e.g., (-1.0, 1.0).
        target_range (tuple): The target pixel value range for the new histogram, e.g., (0.0, 1.0).

    Returns:
        torch.Tensor: The rescaled histogram with uniformly spaced bins, shape (B, num_bins).
    """
    B, num_bins = source_hist.shape
    device = source_hist.device

    # 1. Define the bin centers of the original (source) histogram
    source_bin_centers = torch.linspace(source_range[0], source_range[1], num_bins, device=device)

    # Prepare tensors for broadcasting
    min_vals = min_vals_hist.view(B, 1)
    ranges = image_ranges_hist.view(B, 1)
    
    # Avoid division by zero if range is 0
    ranges[ranges == 0] = 1e-6

    # 2. Transform the bin centers using the image normalization formula
    transformed_centers = (source_bin_centers.unsqueeze(0) - min_vals) / ranges

    # 3. Calculate the corresponding indices in the new target histogram
    target_bin_width = (target_range[1] - target_range[0]) / num_bins
    
    # Calculate target indices and clamp them to the valid range [0, num_bins-1]
    target_indices = torch.floor(
        (transformed_centers - target_range[0]) / target_bin_width
    ).long()
    target_indices = torch.clamp(target_indices, 0, num_bins - 1)

    # 4. Use scatter_add_ to efficiently place counts in the new histogram
    rescaled_hist_list = []
    for i in range(B):
        new_hist = torch.zeros_like(source_hist[i])
        
        # Adds the values from source_hist[i] into new_hist at the computed indices.
        # If multiple source bins map to the same target index, their counts are summed.
        new_hist.scatter_add_(dim=0, index=target_indices[i], src=source_hist[i])
        rescaled_hist_list.append(new_hist)

    return torch.stack(rescaled_hist_list)


def get_histogram_data(image_tensor, num_bins=256):
    """
    Calculates and returns the full histogram counts for a single image tensor.
    """
    # Convert to a batch of 1 to use the vectorized function
    image_batch = image_tensor.unsqueeze(0)
    # Calculate full histogram (mask=None) and return the first (and only) result
    return calculate_batch_histograms(image_batch, num_bins, mask=None)[0]


def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    """Creates a 2D Gaussian kernel."""
    # Create a 1D Gaussian distribution
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    
    # Calculate the 2D Gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
                      )
    # Make sure the kernel sums to 1
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to use with nn.Conv2d
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel

class DiceEdgeLoss(nn.Module):
    """
    Computes a differentiable Dice loss between edge masks extracted from predicted and target images.
    The edge masks are obtained via Sobel gradients and softly binarized to preserve gradient flow.
    """

    def __init__(self, device, threshold=0.5, alpha=10.0, blur_kernel_size=5, blur_sigma=2.0):
        super(DiceEdgeLoss, self).__init__()
        self.device = device
        self.threshold = threshold
        self.alpha = alpha  # sharpness of soft thresholding
        
        # self.blur_kernel_size = blur_kernel_size
        # self.blur_sigma = blur_sigma
        # gaussian_kernel = get_gaussian_kernel(blur_kernel_size, blur_sigma).to(device)
        # self.blur_conv = nn.Conv2d(1, 1, kernel_size=blur_kernel_size, padding='same', bias=False)
        # self.blur_conv.weight = nn.Parameter(gaussian_kernel, requires_grad=False)

        # Sobel filters
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
                               ).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
                               ).unsqueeze(0).unsqueeze(0).to(device)

        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

    def get_edge_mask(self, img):
        # img_blurred = self.blur_conv(img)
        
        gx = self.conv_x(img)
        gy = self.conv_y(img)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

        # Soft binarization: sigmoid step function
        # Produces values in [0, 1] that approximate a binary edge mask
        soft_mask = torch.sigmoid(self.alpha * (grad_mag - self.threshold))
        return soft_mask

    def forward(self, pred_img, target_img):
        """
        pred_img, target_img: (B, 1, H, W) tensors
        """

        pred_edges = self.get_edge_mask(pred_img)
        target_edges = self.get_edge_mask(target_img)

        # Dice loss: 1 - Dice coefficient
        intersection = (pred_edges * target_edges).sum(dim=(1, 2, 3))
        union = pred_edges.sum(dim=(1, 2, 3)) + target_edges.sum(dim=(1, 2, 3)) + 1e-6
        dice = (2 * intersection) / union

        return 1.0 - dice.mean(), pred_edges, target_edges
  
class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a tensor.
    
    Args:
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be corrupted.
        
        Returns:
            Tensor: Corrupted tensor image.
        """
        # torch.randn_like(tensor) creates a tensor of the same size as the input
        # with values from a standard normal distribution (mean=0, std=1).
        # We then scale it by `self.std` and add `self.mean`.
        noise = torch.randn_like(tensor) * self.std + self.mean
        
        # Add the noise to the tensor
        noisy_tensor = tensor + noise
        
        # Your data is normalized to [-1, 1]. It's good practice to clamp the
        # values back into this range after adding noise.
        noisy_tensor = torch.clamp(noisy_tensor, -1.0, 1.0)
        
        return noisy_tensor

    def __repr__(self):
        # This makes it print nicely, e.g., when you print the transform pipeline
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class TotalVariationLoss(nn.Module):
    """
    Calculates the Total Variation (TV) loss for a batch of images.
    This loss encourages spatial smoothness in the generated images,
    penalizing high-frequency noise.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image_batch):
        """
        Args:
            image_batch (torch.Tensor): A batch of images of shape [B, C, H, W].
        
        Returns:
            torch.Tensor: The mean TV loss for the batch.
        """
        # Calculate the differences between adjacent pixels horizontally
        # The slice [:, :, :, :-1] gets all pixels except the last column
        # The slice [:, :, :, 1:] gets all pixels except the first column
        h_variation = torch.sum(torch.abs(image_batch[:, :, :, :-1] - image_batch[:, :, :, 1:]))
        
        # Calculate the differences between adjacent pixels vertically
        # The slice [:, :, :-1, :] gets all pixels except the last row
        # The slice [:, :, 1:, :] gets all pixels except the first row
        v_variation = torch.sum(torch.abs(image_batch[:, :, :-1, :] - image_batch[:, :, 1:, :]))
        
        # The total variation loss is the sum of these variations,
        # usually normalized by the number of pixels.
        b, c, h, w = image_batch.size()
        tv_loss = (h_variation + v_variation) / (b * c * h * w)
        
        return tv_loss

def generate_shuffled_tile_histogram(image_tile, num_bins, dark_threshold=0.05):
    """
    Applies the shuffling logic to a single image tile.
    Separates dark pixels, keeps their counts fixed, and shuffles the rest.
    This is the core logic applied at the finest level of the hierarchy.
    """
    # Note: For this target generation, we don't need differentiability,
    # so we can use a simpler, faster histogram function.
    # We will reuse your `calculate_batch_histograms` for this.
    
    # Ensure the input is a batch of 1 for the helper function
    if image_tile.dim() == 3:
        image_tile = image_tile.unsqueeze(0)

    images_denorm = image_tile * 0.5 + 0.5 # Denormalize from [-1, 1] to [0, 1]
    
    # 1. Calculate histograms for the two parts of the tile
    hist_below_thresh = calculate_batch_histograms(image_tile, num_bins, mask=(images_denorm < dark_threshold))
    hist_above_thresh = calculate_batch_histograms(image_tile, num_bins, mask=(images_denorm >= dark_threshold))
    
    # 2. Shuffle the 'Above Threshold' Part
    original_hist = hist_above_thresh[0] # We are working with a single tile
    non_zero_indices = torch.where(original_hist > 0)[0]
    
    if non_zero_indices.nelement() > 0:
        counts_to_shuffle = original_hist[non_zero_indices]
        shuffled_counts = counts_to_shuffle[torch.randperm(len(counts_to_shuffle))]
        
        shuffled_hist_above = torch.zeros_like(original_hist)
        shuffled_hist_above[non_zero_indices] = shuffled_counts
    else:
        shuffled_hist_above = original_hist

    # 3. Combine and return the final target histogram for the tile
    # The result needs to be unsqueezed to have a "batch" dim of 1 for consistency
    return (hist_below_thresh[0] + shuffled_hist_above).unsqueeze(0)


def generate_hierarchical_target_histograms(input_batch, max_scale, num_bins, dark_threshold=0.05):
    """
    Generates a hierarchical, consistent set of target histograms.
    
    Args:
        input_batch (torch.Tensor): The batch of input images.
        max_scale (int): The finest scale level (e.g., 2 for a 4x4 grid).
    
    Returns:
        list: A list where each element corresponds to a sample in the batch.
              Each element is a dictionary: `targets[scale][row][col] = histogram_tensor`.
    """
    batch_size, _, H, W = input_batch.shape
    batch_targets = []

    for b in range(batch_size):
        input_image = input_batch[b]
        hierarchical_hists = {}
        
        # --- Step 1: Generate histograms at the finest scale ---
        finest_grid_size = 2**max_scale
        patch_h, patch_w = H // finest_grid_size, W // finest_grid_size
        
        finest_hists = [([0] * finest_grid_size) for _ in range(finest_grid_size)]
        
        for r in range(finest_grid_size):
            for c in range(finest_grid_size):
                h_start, w_start = r * patch_h, c * patch_w
                h_end, w_end = h_start + patch_h, w_start + patch_w
                tile = input_image[:, h_start:h_end, w_start:w_end]
                
                # Generate the shuffled target for this specific tile
                finest_hists[r][c] = generate_shuffled_tile_histogram(tile, num_bins, dark_threshold)
        
        hierarchical_hists[max_scale] = finest_hists

        # --- Step 2: Sum up histograms for coarser scales (bottom-up) ---
        for s in range(max_scale - 1, -1, -1):
            grid_size_s = 2**s
            coarse_hists = [([0] * grid_size_s) for _ in range(grid_size_s)]
            
            for r in range(grid_size_s):
                for c in range(grid_size_s):
                    # Get the four children from the finer scale (s+1)
                    child1 = hierarchical_hists[s+1][2*r][2*c]
                    child2 = hierarchical_hists[s+1][2*r][2*c+1]
                    child3 = hierarchical_hists[s+1][2*r+1][2*c]
                    child4 = hierarchical_hists[s+1][2*r+1][2*c+1]
                    
                    # The parent's histogram is the sum of its children's histograms
                    coarse_hists[r][c] = child1 + child2 + child3 + child4
            
            hierarchical_hists[s] = coarse_hists
            
        batch_targets.append(hierarchical_hists)
        
    return batch_targets

class HierarchicalHistogramLoss(nn.Module):
    def __init__(self, max_scale, num_bins=1024, scale_weights=None):
        super(HierarchicalHistogramLoss, self).__init__()
        self.max_scale = max_scale
        self.differentiable_hist = DifferentiableHistogram(num_bins=num_bins, min_val=-1.0, max_val=1.0)
        self.wasserstein_loss = DifferentiableWassersteinLoss()
        
        if scale_weights is None:
            # By default, give equal weight to each scale
            self.scale_weights = [1.0] * (max_scale + 1)
        else:
            assert len(scale_weights) == (max_scale + 1), "Must provide a weight for each scale"
            self.scale_weights = scale_weights

    def forward(self, generated_batch, batch_target_hists):
        total_loss = 0.0
        batch_size, _, H, W = generated_batch.shape
        
        for b in range(batch_size):
            generated_image = generated_batch[b]
            target_hists = batch_target_hists[b]
            
            for s in range(self.max_scale + 1):
                grid_size = 2**s
                patch_h, patch_w = H // grid_size, W // grid_size
                
                for r in range(grid_size):
                    for c in range(grid_size):
                        # Extract patch from the generated image
                        h_start, w_start = r * patch_h, c * patch_w
                        h_end, w_end = h_start + patch_h, w_start + patch_w
                        gen_patch = generated_image[:, h_start:h_end, w_start:w_end]
                        
                        # Calculate its histogram
                        gen_hist = self.differentiable_hist(gen_patch.unsqueeze(0))
                        
                        # Get the corresponding pre-computed target histogram
                        target_hist = target_hists[s][r][c]
                        
                        # Calculate loss and apply scale weight
                        patch_loss = self.wasserstein_loss(gen_hist, target_hist)
                        total_loss += patch_loss * self.scale_weights[s]
        
        # Normalize by batch size and total number of tiles
        num_tiles = sum([4**s for s in range(self.max_scale + 1)])
        return total_loss / (batch_size * num_tiles)
        
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.001
    BATCH_SIZE = 10
    NUM_EPOCHS = 300
    NUMBER_OF_BINS = 32
    
    LAMBDA_FEAT = 1.0
    LAMBDA_EDGE_OUTPUT = 3.0
    LAMBDA_HISTOGRAM = 6.0
    LAMBDA_RANGE = 500.0 
    LAMBDA_TV = 1.7
    LAMBDA_DISIM = 0.2
    
    LAMBDA_HISTOGRAM_HIERARCHICAL = 10.0
    HIST_MAX_SCALE = 1
    HIST_SCALE_WEIGHTS = [1.0, 1.5]
        
    DATA_DIR = "data_prototype_3"
    FEATURE_EXTRACTOR_PATH = "unet_prototype_3.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    wandb.init(
        project="mri-synthesis-prototype-4-cleaned",
        config={
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lambda_feat": LAMBDA_FEAT, "loss_type": "CosineSimilarity"
        }
    )

    # ==========================================================
    # 4. LOAD MODELS
    # ==========================================================
    generator = MRI_Synthesis_Net(scale_factor=1, num_hist_bins=NUMBER_OF_BINS).to(device)

    generator.to(device)
    generator.train()

    # Load the UNet trained on the complex data
    full_unet = UNet(in_channels=1, out_channels=2) # 2 classes: bg and triangle
    full_unet.load_state_dict(torch.load(FEATURE_EXTRACTOR_PATH, map_location=device))
    feature_extractor = full_unet.to(device) # Use the whole U-Net as a feature extractor
    print("Successfully loaded U-Net for feature extraction.")
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # ==========================================================
    # 5. DATA LOADING
    # ==========================================================
    NOISE_STD = 0.05
    to_tensor_transform = transforms.ToTensor()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        AddGaussianNoise(mean=0.0, std=NOISE_STD)
    ])
    dataset = ComplexContrastDataset(root_dir=DATA_DIR, transform=transform)
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {len(dataset)} image pairs for training.")

    # ==========================================================
    # 6. LOSS & OPTIMIZER
    # ==========================================================
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    edge_loss_fn = DiceEdgeLoss(device=device) #DiceLoss() #EdgeLoss(device=device)
    # histogram_loss_fn = WassersteinHistogramLoss(num_bins=NUMBER_OF_BINS).to(device)
    histogram_loss_fn = DifferentiableWassersteinLoss().to(device)
    
    range_loss_fn = RangeLoss().to(device)
    
    tv_loss_fn = TotalVariationLoss().to(device)
    
    hierarchical_hist_loss_fn = HierarchicalHistogramLoss(
        max_scale=HIST_MAX_SCALE,
        scale_weights=HIST_SCALE_WEIGHTS,
        num_bins=NUMBER_OF_BINS
    ).to(device)
    
    similarity_loss_fn = nn.L1Loss()
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.00001)
    differentiable_hist = DifferentiableHistogram(num_bins=NUMBER_OF_BINS, min_val=-1.0, max_val=1.0).to(device)
    
    print("Starting synthesizer prototype 2 training...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (input_images, ref_images) in enumerate(train_loader):
            input_images = input_images.to(device)
            # ref_images = ref_images.to(device)
            
            blur_kernel_size=5
            blur_sigma=2.0
            
            gaussian_kernel = get_gaussian_kernel(blur_kernel_size, blur_sigma).to(device)
            blur_conv = nn.Conv2d(1, 1, kernel_size=blur_kernel_size, padding='same', bias=False)
            blur_conv.weight = nn.Parameter(gaussian_kernel, requires_grad=False)
            
            input_blurred = blur_conv(input_images)


            # target_hist_scale_0 = generate_shuffled_histogram(input_blurred, num_bins=NUMBER_OF_BINS)
            target_hist_scale_0 = generate_shuffled_histogram(input_images, num_bins=NUMBER_OF_BINS)

            # print("SHOULD BE :", target_hist.size())
            # target_hists = generate_hierarchical_target_histograms(
            #     input_images, 
            #     max_scale=HIST_MAX_SCALE, 
            #     num_bins=NUMBER_OF_BINS
            # )
            # list_of_scale_0_hists = [th[0][0][0] for th in target_hists]
            # target_hist_scale_0 = torch.cat(list_of_scale_0_hists, dim=0)
            # print("\n\n BUT IS :", target_hist_scale_0.size())
            

            generated_output, residual = generator(input_images, target_hist_scale_0)     
            # blurred_gen = blur_conv(generated_output) 
            # blurred_res = blur_conv(residual) 
            
            gen_hist = differentiable_hist(generated_output)
            # gen_hist = differentiable_hist(torch.tanh(residual*2+input_blurred))
            
            L_edge_preservation_output, edges_generated, edges_input = edge_loss_fn(generated_output, input_images)
            L_edge_preservation_output += edge_loss_fn(residual, input_images)[0]
            
            # L_range = range_loss_fn(generated_output)
            L_range = range_loss_fn(residual)
            
            # L_histogram = histogram_loss_fn(generated_output, target_hist)
            L_histogram = histogram_loss_fn(gen_hist, target_hist_scale_0)
            # L_histogram = hierarchical_hist_loss_fn(generated_output, target_hists)
            
            threshold = -0.9
            mask = input_images > threshold

            masked_gen_output = torch.masked_select(generated_output, mask)
            masked_input_images = torch.masked_select(input_images, mask)

            # 3. Calculate the L1 loss on the selected pixels
            # Add a check to prevent errors if no pixels are selected
            if masked_input_images.nelement() > 0:
                L_im_sim_input = similarity_loss_fn(masked_gen_output, masked_input_images)
            else:
                # If no pixels are above the threshold, the loss is zero for this component
                L_im_sim_input = torch.tensor(0.0, device=generated_output.device)
                
            L_im_disim_input = 1.0 -  L_im_sim_input
            
            L_tv = tv_loss_fn(residual)
            
                        
            used_losses = {
                        "L_edge_preservation_output": [L_edge_preservation_output, LAMBDA_EDGE_OUTPUT],
                        "L_histogram": [L_histogram, LAMBDA_HISTOGRAM],
                        "L_range": [L_range, LAMBDA_RANGE],
                        "L_tv": [L_tv, LAMBDA_TV],
                        "L_im_disim_input": [L_im_disim_input, LAMBDA_DISIM],
                    }
            
            
                
            total_loss = 0.0
            weighted_losses_log = {}
            for l in used_losses : 
                loss, weight = used_losses[l][0], used_losses[l][1]
                total_loss += loss * weight
                weighted_losses_log[f"weighted_losses/{l}"] = loss.item() * weight
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            # num_bins_for_log = NUMBER_OF_BINS
            # input_hist_data = get_histogram_data(input_images[0], num_bins=num_bins_for_log)
            # target_hist_data = target_hist_scale_0[0] # This is already computed
            # gen_hist_data = get_histogram_data(generated_output[0], num_bins=num_bins_for_log)

            # # 2. Prepare data for the plot
            # # Create x-axis labels (bin centers)
            # bin_centers = [i / num_bins_for_log for i in range(num_bins_for_log)]
            
            # # Create the plot object
            # histogram_plot = wandb.plot.line_series(
            #     xs=bin_centers,
            #     ys=[
            #         input_hist_data.cpu().numpy(), 
            #         target_hist_data.cpu().numpy(), 
            #         gen_hist_data.cpu().numpy()
            #     ],
            #     keys=["Input", "Target", "Generated"],
            #     title="Histogram Comparison (Input vs. Target vs. Generated)",
            #     xname="Pixel Intensity"
            # )

            # --- W&B Logging ---
            wandb.log({
                "miscellaneous/epoch": epoch,
                "main_losses/total_loss": total_loss.item(),
                "main_losses/L_edge_preservation_output" : L_edge_preservation_output.item(),
                "metric_losses/L_histogram" : L_histogram.item(),
                # "histogram/comparison_plot": histogram_plot,
                **weighted_losses_log,
            })
            if epoch%5==0 or epoch == 1 or epoch == NUM_EPOCHS-1: 
                resizer = transforms.Resize((393, 458))
                
                histogram_images = []
                for i in range(4):
                    # 1. Get histogram data for the i-th image in the batch
                    # Using get_histogram_data which denormalizes internally from [-1, 1]
                    input_hist_data = get_histogram_data(input_images[i], num_bins=NUMBER_OF_BINS)
                    target_hist_data = target_hist_scale_0[i]
                    gen_hist_data = get_histogram_data(generated_output[i], num_bins=NUMBER_OF_BINS)
                    
                    # 2. Create the plot image
                    hist_plot_img = plot_histograms_to_image(
                        input_hist=input_hist_data,
                        target_hist=target_hist_data,
                        gen_hist=gen_hist_data,
                        num_bins=NUMBER_OF_BINS
                    )
                    
                    # 3. Convert PIL Image to Tensor and add to list
                    histogram_images.append(to_tensor_transform(resizer(hist_plot_img)))

                # 4. Create a grid from the list of histogram images
                # if histogram_images:
                hist_grid = make_grid(histogram_images, nrow=4)
                wandb.log({"images/comparison_plot": wandb.Image(hist_grid, caption=f"Epoch {epoch+1}: Histogram Comparison")})
                    
                
                
                with torch.no_grad():
                    img_grid = make_grid(
                        # torch.cat((input_images[:4], edges_input[:4], edges_generated[:4], residual[:4], generated_output[:4])),
                        torch.cat((input_images[:4], edges_input[:4], input_blurred[:4], edges_generated[:4], residual[:4], generated_output[:4])),
                        nrow=4, normalize=True
                    )
                    wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Residual | Output")})
            
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
