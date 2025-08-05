# ==========================================================
# 1. HYPERPARAMETERS & SETUP
# ==========================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import glob
import wandb

from mri_model import MRI_Synthesis_Net
from unet import UNet

# ==========================================================
# 2. DATASET CLASS (Reverted to simpler version)
# ==========================================================
class SquareDiskDataset(Dataset):
    """
    Loads a pair of (square_image, disk_image).
    The disk is used as a "content reference" for its features.
    """
    def __init__(self, square_dir, disk_dir, transform=None):
        self.transform = transform
        self.square_paths = sorted(glob.glob(os.path.join(square_dir, "*.png")))
        self.disk_paths = sorted(glob.glob(os.path.join(disk_dir, "*.png")))
        assert len(self.square_paths) == len(self.disk_paths), "Mismatch in number of images."
        assert len(self.square_paths) > 0, f"No images found in data directories."

    def __len__(self):
        return len(self.square_paths)

    def __getitem__(self, idx):
        square_img_pil = Image.open(self.square_paths[idx]).convert("L")
        disk_img_pil = Image.open(self.disk_paths[idx]).convert("L")

        if self.transform:
            input_square_tensor = self.transform(square_img_pil)
            reference_disk_tensor = self.transform(disk_img_pil)
            
        return input_square_tensor, reference_disk_tensor

class TotalVariationLoss(nn.Module):
    """Computes the total variation loss of an image."""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        # Calculate the differences between adjacent pixels
        h_variation = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        v_variation = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return 1/(h_variation + v_variation + 1)
    
    
    
def main():
    # ==========================================================
    # 3. HYPERPARAMETERS & SETUP
    # ==========================================================
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    # Adjusted weights for the new loss structure
    LAMBDA_FEAT = 1.0     # Feature similarity is now the main driver
    LAMBDA_DISSIM = 0.1   # A small push to encourage change
    LAMBDA_TV = 1.0

    SQUARE_DATA_PATH = "data/square"
    DISK_DATA_PATH = "data/disk"
    FEATURE_EXTRACTOR_PATH = "unet_segmentation_weights.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    wandb.init(
        project="mri-contrast-synthesis-prototype",
        config={
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lambda_feat": LAMBDA_FEAT, "lambda_dissim": LAMBDA_DISSIM, "loss_type": "CosineSimilarity"
        }
    )

    # ==========================================================
    # 4. LOAD MODELS
    # ==========================================================
    generator = MRI_Synthesis_Net(scale_factor=1).to(device)
    generator.train()

    full_unet = UNet(in_channels=1, out_channels=3)
    full_unet.load_state_dict(torch.load(FEATURE_EXTRACTOR_PATH, map_location=device))
    feature_extractor = full_unet.encoder.to(device)
    print("Successfully loaded and extracted U-Net encoder for feature loss.")
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
    dataset = SquareDiskDataset(square_dir=SQUARE_DATA_PATH, disk_dir=DISK_DATA_PATH, transform=transform)
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {len(dataset)} image pairs for training.")

    # ==========================================================
    # 6. LOSS & OPTIMIZER (NEW LOGIC)
    # ==========================================================
    cosine_similarity_loss = nn.CosineSimilarity(dim=1)
    dissimilarity_loss_fn = nn.L1Loss()
    tv_loss_fn = TotalVariationLoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # ==========================================================
    # 7. TRAINING LOOP (NEW LOGIC)
    # ==========================================================
    print("Starting synthesizer training with feature similarity objective...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (input_squares, reference_disks) in enumerate(train_loader):
            input_squares = input_squares.to(device)
            reference_disks = reference_disks.to(device)

            generated_output = generator(input_squares)

            # --- EXTRACT FEATURES ---
            features_generated = feature_extractor(generated_output)
            features_square = feature_extractor(input_squares)
            features_disk = feature_extractor(reference_disks)

            # --- CALCULATE LOSSES WITH COSINE SIMILARITY ---
            # 1. Feature Similarity Loss: The output must contain features of a square AND a disk.
            sim_to_square = cosine_similarity_loss(features_generated, features_square)
            sim_to_disk = cosine_similarity_loss(features_generated, features_disk)
            
            # We want to MAXIMIZE similarity, which is equivalent to MINIMIZING its negative.
            # The total feature loss encourages the output to be similar to both.
            L_feat = -(sim_to_square.mean() + sim_to_disk.mean())

            # 2. Dissimilarity Loss: Encourage the model to make a change from the input.
            L_dissim = -1 * dissimilarity_loss_fn(generated_output, input_squares)
            
            L_tv = tv_loss_fn(generated_output)

            total_loss = LAMBDA_FEAT * L_feat #+ LAMBDA_TV * L_tv # + LAMBDA_DISSIM * L_dissim

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # --- W&B Logging ---
        wandb.log({
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "feature_similarity_loss": L_feat.item(),
            "pixel_dissimilarity_loss": L_dissim.item(),
            "variation_loss": L_tv.item()
        })
        
        with torch.no_grad():
            img_grid = make_grid(
                torch.cat((input_squares[:8], reference_disks[:8], generated_output[:8])),
                nrow=8, normalize=True
            )
            wandb.log({"images": wandb.Image(img_grid, caption=f"Epoch {epoch+1}: Input | Reference | Output")})

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {total_loss.item():.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    main()