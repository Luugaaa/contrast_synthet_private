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
from mri_model import MRI_Synthesis_Net
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
    A non-trainable Sobel filter to detect edges in an image.
    Applies both Gx and Gy kernels and returns the gradient magnitude.
    """
    def __init__(self, device):
        super(SobelFilter, self).__init__()
        
        # Define the Gx and Gy Sobel kernels
        # Shape: (out_channels, in_channels, height, width)
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        
        # Create non-trainable convolutional layers
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Set the weights of the convolutions to the Sobel kernels
        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

        # Move the layers to the correct device
        self.to(device)

    def forward(self, x):
        # Apply the Gx and Gy filters
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        
        # Calculate the magnitude of the gradient
        # This represents the "edge map" of the image
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return magnitude

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
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.001 
    BATCH_SIZE = 16 # Adjusted for larger images
    NUM_EPOCHS = 300
    LAMBDA_FEAT = 1.0
    LAMBDA_PIXEL = 1.5
    LAMBDA_DISIM = 0.02
    LAMBDA_TRANSFORM = 1.0
    LAMBDA_REF_FEAT = 0.5
    LAMBDA_EDGE_OUTPUT = 0.2
    LAMBDA_EDGE_RESIDUAL = 0.2
    LAMBDA_INTENSITY = 0.3
    LAMBDA_HISTOGRAM = 1.1
    LAMBDA_BLACK_PIXEL = 0.2
    LAMBDA_RES_VAR = 0.06
    LAMBDA_MEAN_INTENSITY = 2.5
    
    LAMBDA_SIM_INPUT = 0.2
    LAMBDA_SIM_TARGET = 0.2
    
    DATA_DIR = "data_prototype_3"
    FEATURE_EXTRACTOR_PATH = "unet_prototype_3.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    wandb.init(
        project="mri-synthesis-prototype-3",
        config={
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lambda_feat": LAMBDA_FEAT, "loss_type": "CosineSimilarity"
        }
    )

    # ==========================================================
    # 4. LOAD MODELS
    # ==========================================================
    generator = MRI_Synthesis_Net(scale_factor=1).to(device)
    summary(generator.to('cpu'), input_size=(1, 64,64))
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
    edge_similarity_loss_fn = nn.MSELoss()

    diff_feature_loss = nn.L1Loss()
    sobel_filter = SobelFilter(device=device)
    intensity_change_loss_fn = IntensityChangeLoss(background_threshold=0.05)
    histogram_loss_fn = HistogramLoss(background_threshold=0.05)
    black_pixel_loss_fn = BlackPixelPreservationLoss(background_threshold=0.05)
    mean_intensity_loss_fn = MeanIntensityLoss(min_target=0.2, max_target=0.8)
    
    
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

            generated_output, generated_residual = generator(input_images)
            
            # --- EXTRACT FEATURES ---
            
            feat_gen_orig = feature_extractor(generated_output)
            feat_gen_inv = feature_extractor(-generated_output)
            features_generated = (feat_gen_orig + feat_gen_inv) / 2.0
            
            
            # features_generated = feature_extractor(generated_output)
            features_input = feature_extractor(input_images)
            features_ref = feature_extractor(ref_images)
            
            
            edges_generated = sobel_filter(generated_output)
            edges_residual = sobel_filter(generated_residual)
            edges_input = sobel_filter(input_images)
            L_edge_preservation_output = edge_similarity_loss_fn(edges_generated, edges_input)
            L_edge_preservation_residual = torch.nn.functional.relu(edges_residual - edges_input).mean()

            delta_actual_features = features_generated - features_input
            expanded_avg_delta_features = avg_delta_features.expand_as(delta_actual_features)
            L_transform = diff_feature_loss(delta_actual_features, expanded_avg_delta_features)
            
            L_im_sim_input = similarity_loss_fn(generated_output, input_images)
            L_im_sim_target = similarity_loss_fn(generated_output, ref_images)
            L_im_disim_input = 1.0 -  L_im_sim_input

            L_intensity_change = intensity_change_loss_fn(generated_output, input_images)

            expanded_avg_overall_features = avg_overall_features.expand_as(features_generated)
            # expanded_mean_features_set2 = mean_features_set2.expand_as(features_generated)
            L_sim_ref_feat = 1.0 - cosine_similarity_loss(features_generated, expanded_avg_overall_features).mean()

            # Histogram Loss
            dist_to_input_hist = histogram_loss_fn(generated_output, input_images)
            expanded_avg_ref_images = avg_ref_images.expand_as(generated_output)
            dist_to_avg_ref_hist = histogram_loss_fn(generated_output, expanded_avg_ref_images)
            L_histogram = 1.0 - dist_to_input_hist #torch.minimum(dist_to_input_hist, dist_to_ref_hist)
            
            L_mean_intensity = mean_intensity_loss_fn(generated_output)
            
            L_black_pixel = black_pixel_loss_fn(generated_output, input_images)
            
            res_mean = generated_residual.mean()
            res_var = torch.mean(generated_residual**2) - res_mean**2
            L_residual_variance = (1/(res_var + 0.99))**10
            
            used_losses = {
                        "L_sim_ref_feat": [L_sim_ref_feat, LAMBDA_REF_FEAT],
                        "L_edge_preservation_output": [L_edge_preservation_output, LAMBDA_EDGE_OUTPUT],
                        "L_edge_preservation_residual": [L_edge_preservation_residual, LAMBDA_EDGE_RESIDUAL],
                        "L_histogram": [L_histogram, LAMBDA_HISTOGRAM],
                        "L_mean_intensity": [L_mean_intensity, LAMBDA_MEAN_INTENSITY],
                        "L_im_disim_input": [L_im_disim_input, LAMBDA_DISIM]
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
            "main_losses/L_edge_preservation_residual" : L_edge_preservation_residual.item(),
            "main_losses/L_intensity_change" : L_intensity_change.item(), 
            "metric_losses/L_histogram" : L_histogram.item(),
            "metric_losses/black_pixel_loss": L_black_pixel.item(),
            "main_losses/L_sim_ref_feat": L_sim_ref_feat.item(),
            "metric_losses/L_residual_variance": L_residual_variance.item(),
            "metric_losses/L_mean_intensity": L_mean_intensity.item(),
            **weighted_losses_log
        })
            
        scheduler.step()


        
        with torch.no_grad():
            img_grid = make_grid(
                torch.cat((input_images[:8], generated_residual[:8], generated_output[:8])),
                nrow=8, normalize=True
            )
            wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Residual | Output")})

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
