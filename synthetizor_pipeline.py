import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import glob
import numpy as np
import warnings
import random 

warnings.filterwarnings("ignore", message="The given buffer is not writable, and PyTorch does not support non-writable tensors.")

try:
    from prototype_7_mri_model import MRI_Synthesis_Net
except ImportError:
    print("Error: Could not import 'MRI_Synthesis_Net'.")
    print("Please ensure the file 'prototype_7_mri_model.py' containing the model definition is in the same directory.")
    exit()

# =====================================================================================
# ## 헬퍼 함수 및 클래스 (Helper Functions & Classes from Training Code)
# These are the necessary components copied from your training script for preprocessing.
# =====================================================================================

class PreprocessedMriDataset(Dataset):
    """
    A fast and simple dataset for loading preprocessed 2D images (e.g., PNGs).
    """
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in directory: {image_dir}")
        print(f"Successfully found {len(self.image_paths)} preprocessed images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

def calculate_batch_histograms(images_batch, num_bins, mask=None):
    """
    Calculates histograms for a whole batch of images in a vectorized manner.
    """
    B, C, H, W = images_batch.shape
    device = images_batch.device
    images_denorm = images_batch * 0.5 + 0.5
    if mask is None:
        mask = torch.ones_like(images_batch, dtype=torch.bool)
    
    bin_indices = (images_denorm * (num_bins - 1)).long()
    batch_offsets = torch.arange(B, device=device) * num_bins
    offset_indices = bin_indices + batch_offsets.view(B, 1, 1, 1)

    flat_hist = torch.zeros(B * num_bins, device=device)
    flat_indices_to_scatter = offset_indices[mask]
    flat_hist.scatter_add_(0, flat_indices_to_scatter, torch.ones_like(flat_indices_to_scatter, dtype=flat_hist.dtype))

    return flat_hist.view(B, num_bins)

def generate_unified_targets(input_images, num_bins, num_chunks, dark_threshold):
    """
    Generates a synchronized target histogram and the permutation map.
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
    
    return target_hist, perms

def create_range_translation_guidance_map(input_image, perms, num_chunks, dark_threshold):
    """
    Generates a guidance map by performing a range-to-range intensity translation.
    """
    B, _, H, W = input_image.shape
    device = input_image.device

    images_denorm = input_image * 0.5 + 0.5
    background_mask = (images_denorm < dark_threshold)

    fg_pixels_01 = (images_denorm - dark_threshold) / (1.0 - dark_threshold)
    fg_pixels_01 = torch.clamp(fg_pixels_01, 0.0, 1.0)

    original_chunk_idx = (fg_pixels_01 * num_chunks).long()
    original_chunk_idx = torch.clamp(original_chunk_idx, 0, num_chunks - 1)

    chunk_width = 1.0 / num_chunks
    chunk_lower_bound = original_chunk_idx.float() * chunk_width
    relative_pos = (fg_pixels_01 - chunk_lower_bound) / chunk_width

    batch_indices = torch.arange(B, device=device).view(B, 1, 1)
    target_chunk_idx = perms[batch_indices, original_chunk_idx.squeeze(1)].unsqueeze(1)

    target_chunk_lower_bound = target_chunk_idx.float() * chunk_width
    new_fg_value_01 = target_chunk_lower_bound + relative_pos * chunk_width

    new_fg_value_denorm = new_fg_value_01 * (1.0 - dark_threshold) + dark_threshold
    final_map_01 = torch.where(background_mask, images_denorm, new_fg_value_denorm)
    guidance_map = final_map_01 * 2.0 - 1.0
    
    return guidance_map

    
def apply_permutation(hist_shufflable, perms, num_chunks):
    """
    Applies a predefined permutation to the shufflable part of a histogram.
    """
    B, _ = hist_shufflable.shape
    chunk_size = hist_shufflable.shape[1] // num_chunks
    
    original_chunks = hist_shufflable.view(B, num_chunks, chunk_size)
    
    # Expand perms to match chunk dimensions for torch.gather
    perms_expanded = perms.unsqueeze(-1).expand(-1, -1, chunk_size)
    
    shuffled_chunks = torch.gather(original_chunks, dim=1, index=perms_expanded)
    
    return shuffled_chunks.view(B, -1) # Return flattened shuffled histogram part

def run_inference():
    """
    Main function to run the inference process.
    - Selects images randomly.
    - Applies the same contrast permutations across all images.
    """
    # --- 1. Parameters ---
    MODEL_PATH = "mri_contrast_generator_prototype_7.pth"
    DATA_DIR = "datasets/processed_png_raw/"
    OUTPUT_DIR = "inference_results_random"
    
    NUM_IMAGES_TO_PROCESS = 10
    NUM_CONTRASTS_PER_IMAGE = 5
    
    NUMBER_OF_BINS = 288
    HISTOGRAM_CHUNKS = 8
    DARK_PIXEL_THRESHOLD = 0.15

    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {device}")

    # --- 3. Load Model ---
    model = MRI_Synthesis_Net(scale_factor=1, num_hist_bins=NUMBER_OF_BINS).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at '{MODEL_PATH}'")
        return
        
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")

    # --- 4. Load Data and Select Random Indices ---
    mri_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    try:
        dataset = PreprocessedMriDataset(image_dir=DATA_DIR, transform=mri_transform)
    except FileNotFoundError as e:
        print(e)
        return
        
    num_images = min(NUM_IMAGES_TO_PROCESS, len(dataset))
    
    # **MODIFICATION**: Create a list of all possible indices and shuffle it
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    # **MODIFICATION**: Select the first `num_images` from the shuffled list
    selected_indices = all_indices[:num_images]
    print(f"Randomly selected {num_images} images to process.")

    # **MODIFICATION**: Pre-generate a fixed set of permutations for the contrasts
    print(f"Pre-generating {NUM_CONTRASTS_PER_IMAGE} fixed contrast permutations...")
    contrast_permutations = [
        torch.rand(1, HISTOGRAM_CHUNKS, device=device).argsort(dim=1)
        for _ in range(NUM_CONTRASTS_PER_IMAGE)
    ]

    # --- 5. Inference Loop and Saving ---
    images_for_grid = []

    with torch.no_grad():
        # **MODIFICATION**: Loop through the randomly selected indices
        for i, image_idx in enumerate(selected_indices):
            image_output_dir = os.path.join(OUTPUT_DIR, f"image_{i:03d}_idx{image_idx}")
            os.makedirs(image_output_dir, exist_ok=True)

            original_image = dataset[image_idx].to(device)
            original_image_batch = original_image.unsqueeze(0)

            save_image(original_image, os.path.join(image_output_dir, "original.png"), normalize=True)
            
            current_image_row = [original_image.cpu()]

            print(f"  > Generating contrasts for image {i+1}/{num_images} (Dataset index: {image_idx})...")
            
            # Pre-calculate the fixed and shufflable parts of the histogram once per image
            images_denorm = original_image_batch * 0.5 + 0.5
            background_mask = (images_denorm < DARK_PIXEL_THRESHOLD)
            hist_fixed = calculate_batch_histograms(original_image_batch, NUMBER_OF_BINS, mask=background_mask)
            hist_shufflable = calculate_batch_histograms(original_image_batch, NUMBER_OF_BINS, mask=~background_mask)
            
            # **MODIFICATION**: Loop through the pre-generated permutations
            for j, perm in enumerate(contrast_permutations):
                # a. Create target histogram using the predefined permutation
                shuffled_part = apply_permutation(hist_shufflable, perm, HISTOGRAM_CHUNKS)
                target_hist = hist_fixed + shuffled_part

                # b. Create guidance map using the same predefined permutation
                guidance_map = create_range_translation_guidance_map(
                    original_image_batch, perm, HISTOGRAM_CHUNKS, DARK_PIXEL_THRESHOLD
                )
                
                # c. Run model inference
                generated_output, _ = model(original_image_batch, target_hist, guidance_map)
                
                # d. Post-process and save
                generated_contrast = generated_output.squeeze(0)
                save_image(
                    generated_contrast, 
                    os.path.join(image_output_dir, f"contrast_{j+1:02d}.png"), 
                    normalize=True
                )
                
                current_image_row.append(generated_contrast.cpu())

            images_for_grid.append(current_image_row)

    # --- 6. Create and Save Grid Image ---
    if images_for_grid:
        print("Assembling final grid image...")
        flat_image_list = [img for row in images_for_grid for img in row]
        
        grid = make_grid(
            flat_image_list, 
            nrow=NUM_CONTRASTS_PER_IMAGE + 1,
            normalize=True,
            pad_value=0.5,
            padding=4
        )
        
        grid_path = os.path.join(OUTPUT_DIR, "contrast_synthesis_grid.png")
        save_image(grid, grid_path)
        print(f"✅ Grid image saved successfully to {grid_path}")

    print("Inference process complete.")

if __name__ == '__main__':
    run_inference()