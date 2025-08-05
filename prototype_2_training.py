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

from unet import UNet
from mri_model import MRI_Synthesis_Net
from torchsummary import summary

# ==========================================================
# 2. DATASET CLASS FOR COMPLEX DATA
# ==========================================================
class ComplexContrastDataset(Dataset):
    """
    Loads an input image from set1 and a reference image from set2.
    The datasets are unpaired.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
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
            
        return input_tensor, ref_tensor

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
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16 # Adjusted for larger images
    NUM_EPOCHS = 100
    LAMBDA_FEAT = 1.0
    LAMBDA_PIXEL = 1.5
    LAMBDA_DISIM = 0.3
    LAMBDA_TRANSFORM = 1.0
    
    DATA_DIR = "data_prototype_2"
    FEATURE_EXTRACTOR_PATH = "unet_prototype_2.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    wandb.init(
        project="mri-synthesis-prototype-2",
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
    dissimilarity_loss_fn = nn.L1Loss()
    diff_feature_loss = nn.L1Loss()
    # ==========================================================
    # 7. TRAINING LOOP
    # ==========================================================
    print("Starting synthesizer prototype 2 training...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (input_images, ref_images) in enumerate(train_loader):
            input_images = input_images.to(device)
            ref_images = ref_images.to(device)

            generated_output = generator(input_images)
            
            # --- EXTRACT FEATURES ---
            features_generated = feature_extractor(generated_output)
            features_input = feature_extractor(input_images)
            features_ref = feature_extractor(ref_images)

            # --- CALCULATE LOSSES WITH COSINE SIMILARITY ---
            sim_to_content = cosine_similarity_loss(features_generated, features_input)
            sim_to_contrast = cosine_similarity_loss(features_generated, features_ref)
            
            # The total feature loss encourages the output to be similar to both.
            L_feat = -(2* sim_to_content.mean()) #sim_to_contrast.mean() +
            
            delta_target_features = features_ref - features_input
            delta_actual_features = features_generated - features_input
            L_transform = diff_feature_loss(delta_actual_features, delta_target_features)
            # L_dissim = -1 * dissimilarity_loss_fn(generated_output, input_images)
            L_pixel = pixel_loss_fn(generated_output, input_images)
            
            total_loss = LAMBDA_FEAT * L_feat# + LAMBDA_PIXEL * L_pixel #+ LAMBDA_FEAT * L_feat + LAMBDA_PIXEL * L_pixel# + LAMBDA_DISIM * L_dissim LAMBDA_TRANSFORM * L_transform

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        # --- W&B Logging ---
        wandb.log({
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "feature_similarity_loss": L_feat.item(),
            "pixel_count_loss": L_pixel.item(),
        })
        
        with torch.no_grad():
            img_grid = make_grid(
                torch.cat((input_images[:8], ref_images[:8], generated_output[:8])),
                nrow=8, normalize=True
            )
            wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Residual | Output")})

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
