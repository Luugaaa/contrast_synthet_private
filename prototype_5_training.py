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
import nibabel as nib

from unet import UNet
from prototype_5_mri_model import MRI_Synthesis_Net
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import io
import tqdm
import time

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

class PreprocessedMriDataset(Dataset):
    """
    A fast and simple dataset for loading preprocessed 2D images (e.g., PNGs).
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the preprocessed image files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in directory: {image_dir}")
        
        print(f"Successfully found {len(self.image_paths)} preprocessed images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Open the image using PIL and convert to grayscale
        image = Image.open(image_path).convert('L')
        # Apply transforms (e.g., ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
        
        return image
    

class BidsMriDataset(Dataset):
    """
    A PyTorch Dataset for loading 2D slices from 3D NIfTI files
    stored in a BIDS-like directory structure.

    This dataset treats each 2D slice from each 3D volume as an
    individual sample, allowing a 2D model to be trained on 3D data.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the BIDS subject directory (e.g., '.../sub-01/').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.slices = []

        # Find all magnitude T2*w NIfTI files in the anat subfolder
        search_pattern = os.path.join(root_dir, 'anat', '*_part-mag_*T2starw.nii.gz')
        nifti_files = sorted(glob.glob(search_pattern))

        if not nifti_files:
            raise FileNotFoundError(f"No NIfTI files found for pattern: {search_pattern}")

        print(f"Found {len(nifti_files)} NIfTI volume(s). Pre-calculating slice indices...")

        # For each 3D volume, create a list of (file_path, slice_index) tuples.
        # This avoids loading all data into memory at once.
        for file_path in nifti_files:
            try:
                # We only need the header to get the shape, which is fast.
                img_header = nib.load(file_path).header
                num_slices = img_header.get_data_shape()[2] # Assuming slices are on the 3rd axis
                
                for slice_idx in range(num_slices):
                    self.slices.append((file_path, slice_idx))
            except Exception as e:
                print(f"Warning: Could not read header for {file_path}. Skipping. Error: {e}")

        if not self.slices:
            raise ValueError("Could not extract any slices from the found NIfTI files.")
            
        print(f"Successfully created dataset with {len(self.slices)} total 2D slices.")

    def __len__(self):
        """Returns the total number of 2D slices across all volumes."""
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Fetches a single 2D slice and its corresponding label.
        """
        # Get the file path and slice index for the requested sample
        file_path, slice_idx = self.slices[idx]

        # Load the full 3D NIfTI data volume
        # get_fdata() returns a NumPy array with a standard orientation
        nifti_volume = nib.load(file_path).get_fdata()

        # Extract the specific 2D slice
        # We assume the slicing is along the third axis (Axial view)
        mri_slice = nifti_volume[:, :, slice_idx]

        # --- Preprocessing ---

        # 1. Convert to float32
        mri_slice = mri_slice.astype(np.float32)

        # 2. Normalize to [0, 1] range. This is crucial for MRI data
        # where intensity values are not standardized.
        min_val, max_val = mri_slice.min(), mri_slice.max()
        if max_val > min_val:
            mri_slice = (mri_slice - min_val) / (max_val - min_val)
        
        # 3. Add a channel dimension to make it [1, H, W] for grayscale
        mri_slice_tensor = torch.from_numpy(mri_slice).unsqueeze(0)

        # 4. Apply any additional user-defined transforms
        #    (e.g., normalization to [-1, 1], noise, etc.)
        if self.transform:
            mri_slice_tensor = self.transform(mri_slice_tensor)

        return mri_slice_tensor
    
    
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

    def __init__(self, device, threshold=0.6, alpha=2.0, blur_kernel_size=5, blur_sigma=2.0):
        super(DiceEdgeLoss, self).__init__()
        self.device = device
        self.threshold = threshold
        self.alpha = alpha  # sharpness of soft thresholding
        self.l1_loss_fn = nn.L1Loss()
        
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
        
        l1_loss = self.l1_loss_fn(pred_edges, target_edges)

        # Dice loss: 1 - Dice coefficient
        intersection = (pred_edges * target_edges).sum(dim=(1, 2, 3))
        union = pred_edges.sum(dim=(1, 2, 3)) + target_edges.sum(dim=(1, 2, 3)) + 1e-6
        dice = (2 * intersection) / union

        return 1.0 - dice.mean() + l1_loss, pred_edges, target_edges
 
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


def generate_unified_targets_with_means(input_images, num_bins, num_chunks, dark_threshold):
    """
    Generates a synchronized target histogram and a corresponding tensor of
    target mean values for each chunk.
    """
    B, _, H, W = input_images.shape
    device = input_images.device
    chunk_size = num_bins // num_chunks
    assert num_bins % num_chunks == 0, "num_bins must be divisible by num_chunks"

    images_denorm = input_images * 0.5 + 0.5
    background_mask = (images_denorm < dark_threshold)
    
    hist_fixed = calculate_batch_histograms(input_images, num_bins, mask=background_mask)
    hist_shufflable = calculate_batch_histograms(input_images, num_bins, mask=~background_mask)

    original_shufflable_chunks = hist_shufflable.view(B, num_chunks, chunk_size)
    perms = torch.rand(B, num_chunks, device=device).argsort(dim=1)
    
    perms_expanded = perms.unsqueeze(-1).expand(-1, -1, chunk_size)
    shuffled_part_chunks = torch.gather(original_shufflable_chunks, dim=1, index=perms_expanded)
    
    shuffled_part = shuffled_part_chunks.view(B, num_bins)
    target_hist = hist_fixed + shuffled_part

    bin_values = torch.linspace(-1.0, 1.0, num_bins, device=device)
    bin_values_chunked = bin_values.view(1, num_chunks, chunk_size)
    
    original_chunk_means = (original_shufflable_chunks * bin_values_chunked).sum(dim=2) / (original_shufflable_chunks.sum(dim=2) + 1e-6)
    target_chunk_means = torch.gather(original_chunk_means, dim=1, index=perms)
    
    # Return the means directly instead of the full map
    return target_hist, target_chunk_means

class RegionWiseGuidanceLoss(nn.Module):
    """
    Compares the average intensity of the generated output within predefined regions
    to the target mean intensity for those regions. This version is fully batched.
    """
    def __init__(self, num_chunks, dark_threshold):
        super(RegionWiseGuidanceLoss, self).__init__()
        self.num_chunks = num_chunks
        self.dark_threshold = dark_threshold
        self.l1_loss = nn.L1Loss()

    def forward(self, generated_output, input_image, target_chunk_means):
        B, _, H, W = input_image.shape
        device = input_image.device

        images_denorm = input_image * 0.5 + 0.5
        background_mask = (images_denorm < self.dark_threshold) # Shape: [B, 1, H, W]

        # --- 1. Create a stack of all quantile masks ---
        # Create bounds for all chunks at once. Shape: [num_chunks]
        lower_bounds = torch.arange(0, self.num_chunks, device=device) / self.num_chunks
        upper_bounds = (torch.arange(1, self.num_chunks + 1, device=device)) / self.num_chunks

        # Reshape bounds for broadcasting with images. Shape: [1, num_chunks, 1, 1]
        lb_reshaped = lower_bounds.view(1, self.num_chunks, 1, 1)
        ub_reshaped = upper_bounds.view(1, self.num_chunks, 1, 1)
        
        # Compare image with all bounds. Resultant shape: [B, num_chunks, H, W]
        all_masks = (images_denorm >= lb_reshaped) & (images_denorm < ub_reshaped)
        
        # Exclude background from masks. Shape is preserved: [B, num_chunks, H, W]
        all_final_masks = all_masks & ~background_mask

        # --- 2. Calculate means for all regions at once ---
        # `generated_output` is [B, 1, H, W]. `all_final_masks` is [B, num_chunks, H, W].
        # Broadcasting works, resulting in shape: [B, num_chunks, H, W]
        masked_outputs = generated_output * all_final_masks.float()

        # Sum over spatial dimensions (H, W). Result shape: [B, num_chunks]
        sums_of_pixels = masked_outputs.sum(dim=[2, 3])
        nums_pixels_in_masks = all_final_masks.float().sum(dim=[2, 3])

        # Calculate means. Result shape is correctly [B, num_chunks]
        all_generated_means = sums_of_pixels / (nums_pixels_in_masks + 1e-6)

        # --- 3. Loss Calculation ---
        # Now we are correctly comparing two tensors of shape [B, num_chunks]
        loss = self.l1_loss(all_generated_means, target_chunk_means)
        
        # --- 4. Visualization Map Generation (also corrected) ---
        # Reshape means for broadcasting. Shape: [B, num_chunks, 1, 1]
        target_means_reshaped = target_chunk_means.view(B, self.num_chunks, 1, 1)
        generated_means_reshaped = all_generated_means.view(B, self.num_chunks, 1, 1)

        # Build the blocky maps
        input_guidance_map_fg = (all_final_masks.float() * target_means_reshaped).sum(dim=1, keepdim=True)
        input_guidance_map = torch.where(background_mask, -1.0, input_guidance_map_fg)

        generated_guidance_map_fg = (all_final_masks.float() * generated_means_reshaped).sum(dim=1, keepdim=True)
        generated_guidance_map = torch.where(background_mask, -1.0, generated_guidance_map_fg)

        return loss, input_guidance_map, generated_guidance_map
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.001
    BIDS_ROOT_PATH = "datasets/processed_BIDS_full/sub-01/"
    PROCESSED_DATA_DIR = "datasets/processed_png_raw/"
    MODEL_SAVE_PATH = "mri_contrast_generator.pth"
    BATCH_SIZE = 1
    DARK_PIXEL_THRESHOLD = 0.15
    
    NUM_EPOCHS = 300
    NUMBER_OF_BINS = 288
    HISTOGRAM_CHUNKS = 8
    
    LAMBDA_FEAT = 1.0
    LAMBDA_EDGE_OUTPUT = 3.0
    LAMBDA_HISTOGRAM = 3.0
    LAMBDA_RANGE = 500.0 
    LAMBDA_TV = 1.5
    LAMBDA_DISIM = 0.9
    LAMBDA_GUIDANCE = 15.0

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
        project="mri-synthesis-prototype-5",
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
    # full_unet = UNet(in_channels=1, out_channels=2) # 2 classes: bg and triangle
    # full_unet.load_state_dict(torch.load(FEATURE_EXTRACTOR_PATH, map_location=device))
    # feature_extractor = full_unet.to(device) # Use the whole U-Net as a feature extractor
    # print("Successfully loaded U-Net for feature extraction.")
    # feature_extractor.eval()
    # for param in feature_extractor.parameters():
    #     param.requires_grad = False

    # ==========================================================
    # 5. DATA LOADING
    # ==========================================================
    NOISE_STD = 0.05
    to_tensor_transform = transforms.ToTensor()
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    #     AddGaussianNoise(mean=0.0, std=NOISE_STD)
    # ])
    # dataset = ComplexContrastDataset(root_dir=DATA_DIR, transform=transform)
    mri_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # transforms.Resize((256, 256)),
        # transforms.CenterCrop(78),
    ])
    num_workers = 0 if device.type == 'mps' else 2
    # mri_dataset = BidsMriDataset(root_dir=BIDS_ROOT_PATH, transform=mri_transform)
    mri_dataset = PreprocessedMriDataset(image_dir=PROCESSED_DATA_DIR, transform=mri_transform)
            
    train_loader = DataLoader(mri_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
            
    # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {len(mri_dataset)} image pairs for training.")

    # ==========================================================
    # 6. LOSS & OPTIMIZER
    # ==========================================================
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    edge_loss_fn = DiceEdgeLoss(device=device) #DiceLoss() #EdgeLoss(device=device)
    # histogram_loss_fn = WassersteinHistogramLoss(num_bins=NUMBER_OF_BINS).to(device)
    histogram_loss_fn = DifferentiableWassersteinLoss().to(device)
    
    range_loss_fn = RangeLoss().to(device)
    
    tv_loss_fn = TotalVariationLoss().to(device)
    
    guidance_loss_fn = RegionWiseGuidanceLoss(
            num_chunks=HISTOGRAM_CHUNKS,
            dark_threshold=DARK_PIXEL_THRESHOLD
        ).to(device)
    
    # hierarchical_hist_loss_fn = HierarchicalHistogramLoss(
    #     max_scale=HIST_MAX_SCALE,
    #     scale_weights=HIST_SCALE_WEIGHTS,
    #     num_bins=NUMBER_OF_BINS
    # ).to(device)
    
    similarity_loss_fn = nn.L1Loss()
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.00001)
    differentiable_hist = DifferentiableHistogram(num_bins=NUMBER_OF_BINS, min_val=-1.0, max_val=1.0).to(device)
    len_dataset = len(train_loader)
    
    blur_kernel_size=5
    blur_sigma=2.0
    
    gaussian_kernel = get_gaussian_kernel(blur_kernel_size, blur_sigma).to(device)
    blur_conv = nn.Conv2d(1, 1, kernel_size=blur_kernel_size, padding='same', bias=False)
    blur_conv.weight = nn.Parameter(gaussian_kernel, requires_grad=False)
                
                
    print("Starting synthesizer prototype 2 training...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, input_images in tqdm.tqdm(enumerate(train_loader), total=len_dataset):
            t0 = time.time()
            input_images = input_images.to(device)
            # ref_images = ref_images.to(device)

            input_blurred = blur_conv(input_images)
            t1 = time.time()
            #print('blur thing : ', t1-t0)

            # target_hist_scale_0 = generate_shuffled_histogram(input_blurred, num_bins=NUMBER_OF_BINS)
            # target_hist_scale_0 = generate_shuffled_histogram(input_images, num_bins=NUMBER_OF_BINS)
            # target_hist_scale_0 = generate_batched_shuffled_histogram(
            #     input_images, 
            #     num_bins=NUMBER_OF_BINS, 
            #     num_chunks=HISTOGRAM_CHUNKS, 
            #     dark_threshold=0.15
            # )
            
            target_hist_scale_0, target_means = generate_unified_targets_with_means(
                input_images,
                num_bins=NUMBER_OF_BINS,
                num_chunks=HISTOGRAM_CHUNKS,
                dark_threshold=DARK_PIXEL_THRESHOLD
            )
            
            t2 = time.time()
            #print('Gen shuffle hist : ', t2-t1)

            # print("SHOULD BE :", target_hist.size())
            # target_hists = generate_hierarchical_target_histograms(
            #     input_images, 
            #     max_scale=HIST_MAX_SCALE, 
            #     num_bins=NUMBER_OF_BINS
            # )
            # list_of_scale_0_hists = [th[0][0][0] for th in target_hists]
            # target_hist_scale_0 = torch.cat(list_of_scale_0_hists, dim=0)
            # print("\n\n BUT IS :", target_hist_scale_0.size())
            
            # if epoch >=20 and random.randint(0,1): 
            #     generated_output, residual = generator(-input_images, target_hist_scale_0)  
            # else :   
            #     generated_output, residual = generator(input_images, target_hist_scale_0)  
            generated_output, residual = generator(input_images, target_hist_scale_0)  
            t3 = time.time()
            #print('Generator forward :', t3-t2)   
            # blurred_gen = blur_conv(generated_output) 
            # blurred_res = blur_conv(residual) 
            
            # gen_hist = differentiable_hist(generated_output)
            t4 = time.time()
            #print('Produce gen hist : ', t4-t3)
            gen_hist = differentiable_hist(torch.tanh(residual*2+input_blurred))
            
            L_edge_preservation_output, edges_generated, edges_input = edge_loss_fn(generated_output, input_images)
            L_edge_preservation_output += edge_loss_fn(residual, input_images)[0]
            t5 = time.time()
            #print('Edge loss : ', t5-t4)
            # L_range = range_loss_fn(generated_output)
            L_range = range_loss_fn(residual)
            t6  = time.time()
            #print('Range loss :', t6-t5)
            # L_histogram = histogram_loss_fn(generated_output, target_hist)
            L_histogram = histogram_loss_fn(gen_hist, target_hist_scale_0)
            t7 = time.time()
            #print('Hist loss : ', t7-t6)
            # L_histogram = hierarchical_hist_loss_fn(generated_output, target_hists)
            
            threshold = -0.85
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
            t8 = time.time()
            #print('Dissim loss :', t8-t7)
            
            L_tv = tv_loss_fn(residual)
            
            L_guidance, input_guidance_map, gen_guidance_map = guidance_loss_fn(generated_output, input_images, target_means)
                        
            used_losses = {
                        "L_edge_preservation_output": [L_edge_preservation_output, LAMBDA_EDGE_OUTPUT],
                        "L_histogram": [L_histogram, LAMBDA_HISTOGRAM],
                        "L_range": [L_range, LAMBDA_RANGE],
                        "L_tv": [L_tv, LAMBDA_TV],
                        # "L_im_disim_input": [L_im_disim_input, LAMBDA_DISIM],
                        "L_guidance": [L_guidance, LAMBDA_GUIDANCE],
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
        
            if batch_idx == 0 :
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
                    
                    NB_IMAGE_LOGGED = min(BATCH_SIZE, 6)
                    if epoch%5==0 or epoch == 1 or epoch == NUM_EPOCHS-1: 
                        resizer = transforms.Resize((393, 458))
                        
                        histogram_images = []
                        for i in range(NB_IMAGE_LOGGED):
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
                        hist_grid = make_grid(histogram_images, nrow=min(NB_IMAGE_LOGGED, 3))
                        wandb.log({"images/comparison_plot": wandb.Image(hist_grid, caption=f"Epoch {epoch+1}: Histogram Comparison")})
                            
                        
                        
                        with torch.no_grad():
                            img_grid = make_grid(
                                # torch.cat((input_images[NB_IMAGE_LOGGED], edges_input[NB_IMAGE_LOGGED], edges_generated[NB_IMAGE_LOGGED], residual[NB_IMAGE_LOGGED], generated_output[NB_IMAGE_LOGGED])),
                                torch.cat((input_images[:NB_IMAGE_LOGGED], input_guidance_map[:NB_IMAGE_LOGGED], gen_guidance_map[:NB_IMAGE_LOGGED], edges_input[:NB_IMAGE_LOGGED], edges_generated[:NB_IMAGE_LOGGED], residual[:NB_IMAGE_LOGGED], generated_output[:NB_IMAGE_LOGGED])),
                                nrow=NB_IMAGE_LOGGED, normalize=True
                            )
                            wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Residual | Output")})
            #print('Other :',time.time() - t8)        
            
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or (epoch + 1) == NUM_EPOCHS or epoch==0:
            torch.save(generator.state_dict(), MODEL_SAVE_PATH)
            print(f"Epoch {epoch + 1}: Model saved to {MODEL_SAVE_PATH}")
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
