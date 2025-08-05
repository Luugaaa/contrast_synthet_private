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

from unet import UNet
from prototype_4_mri_model import MRI_Synthesis_Net
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================================
# 2. DATASET CLASS FOR COMPLEX DATA
# ==========================================================
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

class SobelFilter(nn.Module):
    """
    A Sobel filter that returns a differentiable, binary-like ("soft") edge map.
    """
    def __init__(self, device, threshold=0.1, steepness=50):
        super(SobelFilter, self).__init__()
        self.threshold = threshold
        self.steepness = steepness  # Controls how sharp the "threshold" is
        
        # Define the Gx and Gy Sobel kernels (unchanged)
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.to(device)

    def forward(self, x):
        # Apply the Gx and Gy filters
        grad_x = self.conv_x(x.detach()) # Detach to treat sobel as a post-processing step for loss calculation
        grad_y = self.conv_y(x.detach())
        
        # Calculate the magnitude of the gradient
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # --- Create a "soft" binary map using sigmoid ---
        # This is a differentiable approximation of a hard threshold.
        # It creates a smooth transition from 0 to 1 around the threshold value.
        soft_binary_map = torch.sigmoid((magnitude - self.threshold) * self.steepness)
        
        # Re-attach the gradient by multiplying with the original input,
        # but in a way that the gradient is only non-zero where the mask is active.
        # This is a common trick to make non-differentiable operations "differentiable".
        # We effectively say "the gradient of the mask is the gradient of the input image".
        return soft_binary_map * x
    
    
# ==========================================================
# PIXEL COUNT PRESERVATION LOSS
# ==========================================================
class PixelCountLoss(nn.Module):
    """
    Penalizes the model for reducing the overall 'energy' or 'activeness' of pixels,
    using a smooth, differentiable sigmoid function instead of a hard threshold.
    """
    def __init__(self, intensity_threshold=0.05, steepness=10.0):
        super(PixelCountLoss, self).__init__()
        # Threshold on the [0, 1] scale
        self.threshold = intensity_threshold
        # Steepness of the sigmoid transition
        self.steepness = steepness

    def forward(self, output_img, input_img):
        # Denormalize from [-1, 1] to [0, 1] for thresholding
        output_denorm = output_img * 0.5 + 0.5
        input_denorm = input_img * 0.5 + 0.5

        # Create soft, differentiable masks using a sigmoid function.
        # This measures the "activeness" of each pixel on a scale from 0 to 1.
        soft_output_mask = torch.sigmoid((output_denorm - self.threshold) * self.steepness)
        soft_input_mask = torch.sigmoid((input_denorm - self.threshold) * self.steepness)

        # Sum the "activeness" over the whole image for both input and output
        total_energy_output = torch.sum(soft_output_mask)
        total_energy_input = torch.sum(soft_input_mask)

        # Calculate the loss based on the reduction of total energy.
        # We use relu to only penalize if energy is lost.
        loss = torch.nn.functional.relu(total_energy_input - total_energy_output)
        
        # Normalize by the input energy to make it a fractional loss, which is more stable
        # across different images. Add a small epsilon to prevent division by zero.
        loss = loss / (total_energy_input + 1e-6)
        
        return loss

# ==========================================================
# NEW INTENSITY CHANGE LOSS
# ==========================================================
class IntensityChangeLoss(nn.Module):
    """
    Encourages the model to change pixel intensities, but only for pixels
    that are not part of the black background in the input image.
    """
    def __init__(self, background_threshold=0.05):
        super(IntensityChangeLoss, self).__init__()
        # Threshold on the [0, 1] scale to define the background
        self.threshold = background_threshold

    def forward(self, output_img, input_img):
        # Denormalize from [-1, 1] to [0, 1] for thresholding
        input_denorm = input_img * 0.5 + 0.5

        # Create a mask of the non-background pixels from the input
        # We only care about changes in these regions
        content_mask = (input_denorm > self.threshold).float()

        # Calculate the absolute difference between the output and input
        abs_difference = torch.abs(output_img - input_img)

        # Apply the mask to focus only on the content regions
        masked_difference = abs_difference * content_mask

        # We want to MAXIMIZE this difference to encourage change.
        # This is equivalent to MINIMIZING its negative.
        # We take the mean over the masked region to get the average change.
        # Add epsilon to prevent division by zero if mask is all black.
        loss = -torch.sum(masked_difference) / (torch.sum(content_mask) + 1e-6)
        
        return loss

# ==========================================================
# DIFFERENTIABLE HISTOGRAM LOSS
# ==========================================================
class HistogramLoss(nn.Module):
    """
    Calculates the 1D Sliced Wasserstein Distance between the intensity
    distributions of the non-background pixels of two images.
    """
    def __init__(self, background_threshold=0.05):
        super(HistogramLoss, self).__init__()
        self.threshold = background_threshold

    def forward(self, generated_img, target_img):
        # Denormalize from [-1, 1] to [0, 1] to work with the threshold
        gen_denorm = generated_img * 0.5 + 0.5
        target_denorm = target_img * 0.5 + 0.5

        # Create masks to select only the non-background pixels
        gen_mask = gen_denorm > self.threshold
        target_mask = target_denorm > self.threshold

        # Extract the pixel intensity values from the content areas
        gen_pixels = torch.masked_select(gen_denorm, gen_mask)
        target_pixels = torch.masked_select(target_denorm, target_mask)

        # If either image has no content, the loss is zero
        if gen_pixels.nelement() == 0 or target_pixels.nelement() == 0:
            return torch.tensor(0.0, device=generated_img.device)

        # Sort the pixel values to approximate the cumulative distribution function (CDF)
        gen_sorted, _ = torch.sort(gen_pixels)
        target_sorted, _ = torch.sort(target_pixels)

        # To compare distributions, they must have the same number of samples.
        # We resample the shorter tensor to match the length of the longer one.
        len_gen = gen_sorted.shape[0]
        len_target = target_sorted.shape[0]

        if len_gen > len_target:
            # Reshape to (N, C, L) for interpolation
            target_sorted = torch.nn.functional.interpolate(target_sorted.unsqueeze(0).unsqueeze(0),
                                                            size=len_gen, mode='linear',
                                                            align_corners=False).squeeze(0).squeeze(0)
        elif len_target > len_gen:
            gen_sorted = torch.nn.functional.interpolate(gen_sorted.unsqueeze(0).unsqueeze(0),
                                                         size=len_target, mode='linear',
                                                         align_corners=False).squeeze(0).squeeze(0)

        # The Wasserstein-1 distance is the L1 distance between the sorted distributions (inverse CDFs)
        return torch.mean(torch.abs(gen_sorted - target_sorted))

# ==========================================================
# BLACK PIXEL PRESERVATION LOSS
# ==========================================================
class BlackPixelPreservationLoss(nn.Module):
    """
    Penalizes the model if the number of black pixels in the output is more
    than the maximum number of black pixels in either the input or reference.
    """
    def __init__(self, background_threshold=0.05):
        super(BlackPixelPreservationLoss, self).__init__()
        # Threshold on the [0, 1] scale to define the background
        self.threshold = background_threshold

    def forward(self, output_img, input_img):
        # Denormalize from [-1, 1] to [0, 1] for thresholding
        output_denorm = output_img * 0.5 + 0.5
        input_denorm = input_img * 0.5 + 0.5

        # Create boolean masks of pixels below the intensity threshold (i.e., "black" pixels)
        output_mask = output_denorm < self.threshold
        input_mask = input_denorm < self.threshold

        # Count the number of black pixels in each image in the batch
        count_output = torch.sum(output_mask, dim=[1, 2, 3]).float()
        count_input = torch.sum(input_mask, dim=[1, 2, 3]).float()

        # The target is the maximum number of black pixels from either source
        target_count = count_input

        # Penalize only if the output has fewer black pixels than the target
        # This is a direct measure of how much background was incorrectly filled in
        loss = torch.nn.functional.relu(count_output*0.9 - target_count).mean()
        
        return loss

class MeanIntensityLoss(nn.Module):
    """
    Penalizes the image if its mean intensity is too close to the extremes (0 or 1),
    acting as a guardrail against generating all-black or all-white images.
    """
    def __init__(self, min_target=0.2, max_target=0.8):
        super(MeanIntensityLoss, self).__init__()
        self.min_target = min_target
        self.max_target = max_target

    def forward(self, x):
        # Denormalize from [-1, 1] to [0, 1]
        x_denorm = x * 0.5 + 0.5
        mean_intensity = torch.mean(x_denorm)
        
        # Penalize if the mean is below the min target or above the max target
        loss = torch.nn.functional.relu(self.min_target - mean_intensity) + \
               torch.nn.functional.relu(mean_intensity - self.max_target)
               
        return loss
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten both tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # The loss is 1 - Dice
        return 1 - dice
    

class WassersteinHistogramLoss(nn.Module):
    """
    Calculates the 1D Wasserstein Distance (or Earth Mover's Distance) between
    the intensity distributions of a batch of generated images and a batch of
    target histograms.

    This loss is particularly well-suited for sparse, "spiky" histograms as it
    considers the distance between bins, not just the difference in their counts.
    """
    def __init__(self, num_bins=1024):
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

    # def forward(self, pred_img, true_img):
    #     """Calculates the final loss using the corrected method."""
    #     # Epsilon for numerical stability
    #     eps = 1e-6
    #     # Calculate gradients for the predicted image
    #     gx_pred, gy_pred = self.get_gradients(pred_img)
        
    #     # Calculate gradients for the true image
    #     gx_true, gy_true = self.get_gradients(true_img)

    #     # Calculate magnitudes (L2 norm of the gradient vectors)
    #     mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + eps)
    #     mag_true = torch.sqrt(gx_true**2 + gy_true**2 + eps)

    #     # Normalize the gradient vectors to get direction only.
    #     # In flat areas (where magnitude is near zero), the normalized gradients will also be zero.
    #     # This correctly results in a zero loss contribution from non-edge regions.
    #     norm_gx_pred = gx_pred / mag_pred
    #     norm_gy_pred = gy_pred / mag_pred
        
    #     norm_gx_true = gx_true / mag_true
    #     norm_gy_true = gy_true / mag_true
        
    #     # The loss is the L1 distance between the normalized gradient vectors.
    #     # This measures the difference in direction and is zero for flat areas.
    #     # loss = torch.abs(norm_gx_pred - norm_gx_true) + torch.abs(norm_gy_pred - norm_gy_true)
        
    #     # # We return the mean over all pixels, which is now correct because
    #     # # non-edge pixels contribute zero to the sum.
    #     # return loss.mean()
        
    #     # Inner product of normalized gradient vectors
    #     dot_product = norm_gx_pred * norm_gx_true + norm_gy_pred * norm_gy_true
    #     # Clamp for numerical stability
    #     angle_diff = 1.0 - dot_product.clamp(-1.0, 1.0)
    #     return angle_diff.mean()
    
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
    def __init__(self, num_bins=1024):
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

def generate_shuffled_histogram(input_images, num_bins=1024, dark_threshold=0.05):
    """
    Generates a target histogram by shuffling the counts of pixels ABOVE a
    threshold, while keeping the counts of pixels BELOW the threshold fixed.
    """
    B = input_images.shape[0]
    images_denorm = input_images * 0.5 + 0.5

    # 1. Calculate histograms for the two parts of the image
    hist_below_thresh = calculate_batch_histograms(input_images, num_bins, mask=(images_denorm < dark_threshold))
    hist_above_thresh = calculate_batch_histograms(input_images, num_bins, mask=(images_denorm >= dark_threshold))

    # 2. Shuffle the 'Above Threshold' Part
    shuffled_hist_above = torch.zeros_like(hist_above_thresh)
    for i in range(B):
        original_hist = hist_above_thresh[i]
        non_zero_indices = torch.where(original_hist > 0)[0]

        if non_zero_indices.nelement() == 0:
            shuffled_hist_above[i] = original_hist
            continue

        counts_to_shuffle = original_hist[non_zero_indices]
        shuffled_counts = counts_to_shuffle[torch.randperm(len(counts_to_shuffle))]

        new_hist = torch.zeros_like(original_hist)
        new_hist[non_zero_indices] = shuffled_counts
        shuffled_hist_above[i] = new_hist

    # 3. Combine the fixed and shuffled parts
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


def get_histogram_data(image_tensor, num_bins=1024):
    """
    Calculates and returns the full histogram counts for a single image tensor.
    """
    # Convert to a batch of 1 to use the vectorized function
    image_batch = image_tensor.unsqueeze(0)
    # Calculate full histogram (mask=None) and return the first (and only) result
    return calculate_batch_histograms(image_batch, num_bins, mask=None)[0]


class DiceEdgeLoss(nn.Module):
    """
    Computes a differentiable Dice loss between edge masks extracted from predicted and target images.
    The edge masks are obtained via Sobel gradients and softly binarized to preserve gradient flow.
    """

    def __init__(self, device, threshold=0.1, alpha=10.0):
        super(DiceEdgeLoss, self).__init__()
        self.device = device
        self.threshold = threshold
        self.alpha = alpha  # sharpness of soft thresholding

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
    
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.0001 
    BATCH_SIZE = 16 # Adjusted for larger images
    NUM_EPOCHS = 300
    LAMBDA_FEAT = 1.0
    LAMBDA_PIXEL = 1.5
    LAMBDA_DISIM = 0.2
    LAMBDA_TRANSFORM = 1.0
    LAMBDA_REF_FEAT = 1.0
    LAMBDA_EDGE_OUTPUT = 1.0
    LAMBDA_EDGE_RESIDUAL = 0.5
    LAMBDA_INTENSITY = 0.3
    LAMBDA_HISTOGRAM = 300.0
    LAMBDA_BLACK_PIXEL = 0.2
    LAMBDA_RES_VAR = 0.06
    LAMBDA_MEAN_INTENSITY = 10.0
    LAMBDA_RANGE = 250.0 
    
    LAMBDA_SIM_INPUT = 0.2
    LAMBDA_SIM_TARGET = 0.2
    
    DATA_DIR = "data_prototype_3"
    FEATURE_EXTRACTOR_PATH = "unet_prototype_3.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    wandb.init(
        project="mri-synthesis-prototype-4",
        config={
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lambda_feat": LAMBDA_FEAT, "loss_type": "CosineSimilarity"
        }
    )

    # ==========================================================
    # 4. LOAD MODELS
    # ==========================================================
    generator = MRI_Synthesis_Net(scale_factor=1, num_hist_bins=1024).to(device)
    # dummy_image_input = torch.randn(1, 1, 64, 64)
    # dummy_hist_input = torch.randn(1, 1024)

    # # Pass the dummy data directly using the `input_data` argument
    # print("Model Summary:")
    # summary(generator.to('cpu'), input_data=[dummy_image_input, dummy_hist_input])

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = ComplexContrastDataset(root_dir=DATA_DIR, transform=transform)
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {len(dataset)} image pairs for training.")

    # ==========================================================
    # 6. LOSS & OPTIMIZER
    # ==========================================================
    cosine_similarity_loss = nn.CosineSimilarity(dim=1)
    pixel_loss_fn = PixelCountLoss(intensity_threshold=0.02)
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    similarity_loss_fn = nn.L1Loss()
    # edge_similarity_loss_fn = nn.MSELoss()
    edge_loss_fn = DiceEdgeLoss(device=device) #DiceLoss() #EdgeLoss(device=device)

    diff_feature_loss = nn.L1Loss()
    sobel_filter = SobelFilter(device=device)
    intensity_change_loss_fn = IntensityChangeLoss(background_threshold=0.05)
    # histogram_loss_fn = HistogramLoss(background_threshold=0.05)
    black_pixel_loss_fn = BlackPixelPreservationLoss(background_threshold=0.05)
    mean_intensity_loss_fn = MeanIntensityLoss(min_target=0.2, max_target=0.8)
    histogram_loss_fn = WassersteinHistogramLoss(num_bins=1024).to(device)
    
    range_loss_fn = RangeLoss().to(device)
    
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.00001)
    # ==========================================================
    # 7. TRAINING LOOP
    # ==========================================================
    print("Pre-computing the average feature delta...")

    # Create separate datasets and loaders for each set
    set1_dataset = ComplexContrastDataset(root_dir=DATA_DIR, set_to_load='set1', transform=transform)
    set2_dataset = ComplexContrastDataset(root_dir=DATA_DIR, set_to_load='set2', transform=transform)
    loader1 = DataLoader(set1_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loader2 = DataLoader(set2_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 1: Calculate the mean feature vector for set 1 ---
    mean_features_set1 = None
    with torch.no_grad():
        for input_images in loader1:
            features = feature_extractor(input_images.to(device))
            if mean_features_set1 is None:
                mean_features_set1 = features.sum(dim=0)
            else:
                mean_features_set1 += features.sum(dim=0)
    mean_features_set1 /= len(set1_dataset)

    # --- Step 2: Calculate the mean feature vector for set 2 ---
    sum_ref_images = None
    sum_features_set2 = None

    with torch.no_grad():
        for ref_images_batch in loader2: # ref_images_batch has shape [B, C, H, W]
            # Move batch to the correct device
            ref_images_batch = ref_images_batch.to(device)

            # --- Correct Image Averaging ---
            # Sum the images in the current batch along the batch dimension (dim=0)
            if sum_ref_images is None:
                sum_ref_images = ref_images_batch.sum(dim=0)
            else:
                sum_ref_images += ref_images_batch.sum(dim=0)

            # --- Feature Averaging (Your original logic was correct) ---
            features = feature_extractor(ref_images_batch)
            if sum_features_set2 is None:
                sum_features_set2 = features.sum(dim=0)
            else:
                sum_features_set2 += features.sum(dim=0)
    
    avg_ref_images = sum_ref_images / len(set2_dataset)
    mean_features_set2 = sum_features_set2 / len(set2_dataset)

    # --- Step 3: The average delta is the difference between the two means ---
    avg_delta_features = mean_features_set2 - mean_features_set1
    
    print("Pre-computing the global average feature vector for inversion invariance...")
    sum_overall_features = None
    
    with torch.no_grad():
        # Process set 1 (original and inverted)
        for images_batch in loader1:
            images_batch = images_batch.to(device)
            feat_orig = feature_extractor(images_batch)
            feat_inv = feature_extractor(-images_batch) # Features of inverted image
            
            batch_sum = feat_orig.sum(dim=0) + feat_inv.sum(dim=0)
            
            if sum_overall_features is None:
                sum_overall_features = batch_sum
            else:
                sum_overall_features += batch_sum

        # Process set 2 (original and inverted)
        for images_batch in loader2:
            images_batch = images_batch.to(device)
            feat_orig = feature_extractor(images_batch)
            feat_inv = feature_extractor(-images_batch)
            
            sum_overall_features += feat_orig.sum(dim=0) + feat_inv.sum(dim=0)

    # Calculate the final global average from all four sources
    total_feature_vectors = 2 * (len(set1_dataset) + len(set2_dataset))
    avg_overall_features = sum_overall_features / total_feature_vectors
    print("Global average feature vector computed.")
    
    print("Starting synthesizer prototype 2 training...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (input_images, ref_images) in enumerate(train_loader):
            input_images = input_images.to(device)
            ref_images = ref_images.to(device)

            target_hist = generate_shuffled_histogram(input_images, num_bins=1024)
            generated_output, residual = generator(input_images, target_hist)
            
            # --- EXTRACT FEATURES ---
            
            feat_gen_orig = feature_extractor(generated_output)
            feat_gen_inv = feature_extractor(-generated_output)
            features_generated = (feat_gen_orig + feat_gen_inv) / 2.0
            
            
            # features_generated = feature_extractor(generated_output)
            features_input = feature_extractor(input_images)
            features_ref = feature_extractor(ref_images)
            
            # edges_generated = sobel_filter(generated_output)
            # # edges_residual = sobel_filter(generated_residual)
            # edges_input = sobel_filter(input_images)
            
            # B, C, H, W = edges_generated.shape
            # edges_generated_flat = edges_generated.view(B, -1)
            # edges_input_flat = edges_input.view(B, -1)
            
            # similarity = edge_similarity_loss_fn(edges_generated_flat, edges_input_flat)
            # L_edge_preservation_output = 1 - similarity.mean()
            
            # mask_generated = torch.sigmoid(edges_generated * steepness)
            # mask_input = torch.sigmoid(edges_input * steepness)
            
            # L_edge_preservation_output = edge_similarity_loss_fn(mask_generated, mask_input)
            
            # L_edge_preservation_output = edge_similarity_loss_fn(edges_generated, edges_input)
            
            
            L_edge_preservation_output, edges_generated, edges_input = edge_loss_fn(generated_output, input_images)
            # L_edge_preservation_output += edge_loss_fn(residual, input_images)[0]
            
            
            # L_edge_preservation_output += edge_loss_fn(-generated_output, input_images)
            
            # L_edge_preservation_residual = torch.nn.functional.relu(edges_residual - edges_input).mean()
            # L_edge_preservation_residual = edge_similarity_loss_fn(edges_residual, edges_input) \
            #                             + torch.nn.functional.relu(edges_residual - edges_input).mean()

            delta_actual_features = features_generated - features_input
            expanded_avg_delta_features = avg_delta_features.expand_as(delta_actual_features)
            L_transform = diff_feature_loss(delta_actual_features, expanded_avg_delta_features)
            
            min_vals = torch.amin(generated_output, dim=(1, 2, 3), keepdim=True)
            max_vals = torch.amax(generated_output, dim=(1, 2, 3), keepdim=True)

            # 2. Calculate the range, adding a small epsilon to prevent division by zero
            #    for flat (all-black or all-white) images.
            image_ranges = max_vals - min_vals + 1e-8

            # 3. Normalize the generated output to [-1, 1]
            #    This operation is fully differentiable.
            rescaled_generated_output = 2 * (generated_output - min_vals) / image_ranges - 1

            # 4. Now, calculate the similarity loss using the normalized output.
            #    The result will be cleanly bounded between 0 and 2.
            threshold = -0.9

            # 1. Create a boolean mask based on the input image's pixel values
            mask = input_images > threshold

            # 2. Select only the pixels from both images where the mask is True
            masked_gen_output = torch.masked_select(rescaled_generated_output, mask)
            masked_input_images = torch.masked_select(input_images, mask)

            # 3. Calculate the L1 loss on the selected pixels
            # Add a check to prevent errors if no pixels are selected
            if masked_input_images.nelement() > 0:
                L_im_sim_input = similarity_loss_fn(masked_gen_output, masked_input_images)
            else:
                # If no pixels are above the threshold, the loss is zero for this component
                L_im_sim_input = torch.tensor(0.0, device=generated_output.device)
                

            # L_im_sim_input = similarity_loss_fn(generated_output, input_images)
            L_im_sim_target = similarity_loss_fn(generated_output, ref_images)
            L_im_disim_input = 1.0 -  L_im_sim_input

            L_intensity_change = intensity_change_loss_fn(generated_output, input_images)

            expanded_avg_overall_features = avg_overall_features.expand_as(features_generated)
            # expanded_mean_features_set2 = mean_features_set2.expand_as(features_generated)
            L_sim_ref_feat = 1.0 - cosine_similarity_loss(features_generated, expanded_avg_overall_features).mean()

            L_range = range_loss_fn(generated_output)
            
            # Histogram Loss
            # dist_to_input_hist = histogram_loss_fn(generated_output, input_images)
            # expanded_avg_ref_images = avg_ref_images.expand_as(generated_output)
            # dist_to_avg_ref_hist = histogram_loss_fn(generated_output, expanded_avg_ref_images)
            # L_histogram = 1.0 - dist_to_input_hist #torch.minimum(dist_to_input_hist, dist_to_ref_hist)
            gen_min = torch.amin(generated_output, dim=(1, 2, 3), keepdim=True)
            gen_max = torch.amax(generated_output, dim=(1, 2, 3), keepdim=True)
            neg_one = torch.full_like(gen_min, -1.0)
            pos_one = torch.full_like(gen_max, 1.0)

            # 3. Use torch.minimum/maximum to get the final range for normalization
            # This creates a normalization range that is the union of the generated image's
            # range and the [-1, 1] range.
            min_vals_hist = torch.minimum(gen_min, neg_one)
            max_vals_hist = torch.maximum(gen_max, pos_one)
            
            image_ranges_hist = max_vals_hist - min_vals_hist + 1e-8
            normalized_output_image_for_hist = 2*(generated_output - min_vals_hist) / image_ranges_hist -1 
            
            rescaled_target_hist = rebin_histogram(
                source_hist=target_hist,
                min_vals_hist=min_vals_hist,
                image_ranges_hist=image_ranges_hist,
                source_range=(-1.0, 1.0) 
            )
            
            rescaled_input = 2*(input_images - min_vals_hist) / image_ranges_hist -1 
            
            # print("HERE :", torch.min(normalized_output_image_for_hist), torch.max(normalized_output_image_for_hist), torch.min(rescaled_target_hist), torch.max(rescaled_target_hist), target_hist[0], rescaled_target_hist[0])
            # L_histogram = histogram_loss_fn(normalized_output_image_for_hist, rescaled_target_hist)
            L_histogram = histogram_loss_fn(generated_output, target_hist)
            
            L_mean_intensity = mean_intensity_loss_fn(generated_output)
            
            L_black_pixel = black_pixel_loss_fn(generated_output, input_images)
            
            # res_mean = generated_residual.mean()
            # res_var = torch.mean(generated_residual**2) - res_mean**2
            # L_residual_variance = (1/(res_var + 0.99))**10
            
            used_losses = {
                        # "L_sim_ref_feat": [L_sim_ref_feat, LAMBDA_REF_FEAT],
                        "L_edge_preservation_output": [L_edge_preservation_output, LAMBDA_EDGE_OUTPUT],
                        # "L_edge_preservation_residual": [L_edge_preservation_residual, LAMBDA_EDGE_RESIDUAL],
                        "L_histogram": [L_histogram, LAMBDA_HISTOGRAM],
                        # "L_mean_intensity": [L_mean_intensity, LAMBDA_MEAN_INTENSITY],
                        # "L_im_disim_input": [L_im_disim_input, LAMBDA_DISIM],
                        "L_range": [L_range, LAMBDA_RANGE]
                    }
            
            
                
            total_loss = 0.0
            weighted_losses_log = {}
            for l in used_losses : 
                loss, weight = used_losses[l][0], used_losses[l][1]
                total_loss += loss * weight
                weighted_losses_log[f"weighted_losses/{l}"] = loss.item() * weight
            
            # total_loss = LAMBDA_TRANSFORM * L_transform \
            # total_loss = LAMBDA_REF_FEAT * L_sim_ref_feat \
            #     + LAMBDA_EDGE * L_edge_preservation \
            #     + LAMBDA_HISTOGRAM * L_histogram \
            #     + LAMBDA_MEAN_INTENSITY * L_mean_intensity
            #     # + LAMBDA_BLACK_PIXEL * L_black_pixel
            #     # + LAMBDA_RES_VAR * L_residual_variance \
                
            #     # + LAMBDA_HISTOGRAM * L_histogram \
            #     # + LAMBDA_INTENSITY * L_intensity_change \
            #     # + LAMBDA_DISIM * L_im_disim_input 
            #     # + LAMBDA_BLACK_PIXEL * L_black_pixel

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            num_bins_for_log = 1024
            # 1. Get the histogram data for all three distributions
            # input_hist_data = get_histogram_data(rescaled_input[0], num_bins=num_bins_for_log)
            # target_hist_data = rescaled_target_hist[0] # This is already computed
            # gen_hist_data = get_histogram_data(rescaled_generated_output[0], num_bins=num_bins_for_log)
            input_hist_data = get_histogram_data(input_images[0], num_bins=num_bins_for_log)
            target_hist_data = target_hist[0] # This is already computed
            gen_hist_data = get_histogram_data(generated_output[0], num_bins=num_bins_for_log)

            # 2. Prepare data for the plot
            # Create x-axis labels (bin centers)
            bin_centers = [i / num_bins_for_log for i in range(num_bins_for_log)]
            
            # Create the plot object
            histogram_plot = wandb.plot.line_series(
                xs=bin_centers,
                ys=[
                    input_hist_data.cpu().numpy(), 
                    target_hist_data.cpu().numpy(), 
                    gen_hist_data.cpu().numpy()
                ],
                keys=["Input", "Target", "Generated"],
                title="Histogram Comparison (Input vs. Target vs. Generated)",
                xname="Pixel Intensity"
            )

        # --- W&B Logging ---
        wandb.log({
            "miscellaneous/epoch": epoch,
            "main_losses/total_loss": total_loss.item(),
            # "feature_similarity_loss": L_feat.item(),
            "metric_losses/transform_loss": L_transform.item(),
            # "pixel_count_loss": L_pixel.item(),
            "metric_losses/sim_input_loss": L_im_sim_input.item(),
            "metric_losses/sim_target_loss": L_im_sim_target.item(), 
            "main_losses/L_edge_preservation_output" : L_edge_preservation_output.item(),
            # "main_losses/L_edge_preservation_residual" : L_edge_preservation_residual.item(),
            "main_losses/L_intensity_change" : L_intensity_change.item(), 
            "metric_losses/L_histogram" : L_histogram.item(),
            "metric_losses/black_pixel_loss": L_black_pixel.item(),
            "main_losses/L_sim_ref_feat": L_sim_ref_feat.item(),
            # "metric_losses/L_residual_variance": L_residual_variance.item(),
            "metric_losses/L_mean_intensity": L_mean_intensity.item(),
            "histogram/comparison_plot": histogram_plot,
            **weighted_losses_log,
        })
            
        scheduler.step()
        
        with torch.no_grad():
            img_grid = make_grid(
                torch.cat((input_images[:8], edges_input[:8], edges_generated[:8], residual[:8], generated_output[:8])),
                nrow=8, normalize=True
            )
            wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Residual | Output")})

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
