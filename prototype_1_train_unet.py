import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import wandb 

from unet import UNet


class ShapeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading disk and square images and generating
    segmentation masks on the fly.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all image paths and sort them to ensure consistency
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*", "*.png")))
        # Class mapping: 0=background, 1=disk, 2=square
        self.class_map = {"disk": 1, "square": 2}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert("L") # Ensure grayscale
        
        # Create segmentation mask
        img_array = np.array(image)
        mask = np.zeros_like(img_array, dtype=np.int64)
        
        # Determine shape color by checking a corner pixel for background color
        bg_color = img_array[0, 0]
        shape_color = 0 if bg_color == 255 else 255
        
        # Get class label from the parent directory name
        class_name = os.path.basename(os.path.dirname(img_path))
        class_id = self.class_map[class_name]
        
        # Assign class id to the shape pixels in the mask
        mask[img_array == shape_color] = class_id
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask) # Convert mask to tensor
            
        return image, mask

# --- Training Configuration ---
DATA_DIR = "data"
MODEL_SAVE_PATH = "unet_segmentation_weights.pth"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 25
VALIDATION_SPLIT = 0.15

def train_model():
    """Main function to run the training and validation loop."""
    # --- Initialize W&B ---
    wandb.init(
        project="unet-shape-segmentation",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "U-Net",
            "dataset": "Disks-and-Squares",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )
    
    device = 'mps'
    print(f"Using device: {device}")

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    full_dataset = ShapeDataset(root_dir=DATA_DIR, transform=transform)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataset loaded: {train_size} training images, {val_size} validation images.")

    # --- Model, Loss, and Optimizer ---
    model = UNet(in_channels=1, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- W&B: Watch the model ---
    wandb.watch(model, log="all", log_freq=100)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation and W&B Logging Loop ---
        model.eval()
        val_loss = 0.0
        # List to store images for logging
        wandb_images = []
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # On the first batch of validation, prepare images for logging
                if i == 0:
                    # Get predicted masks by taking argmax
                    pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Log up to 8 images
                    for j in range(min(len(images), 8)):
                        wandb_images.append(wandb.Image(
                            images[j].cpu(),
                            masks={
                                "prediction": {"mask_data": pred_masks[j], "class_labels": {1: "disk", 2: "square"}},
                                "ground_truth": {"mask_data": masks[j].cpu().numpy(), "class_labels": {1: "disk", 2: "square"}}
                            }
                        ))

        avg_val_loss = val_loss / len(val_loader)
        
        # --- Log metrics and images to W&B ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "predictions": wandb_images
        })
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Save the trained model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    
    # --- Finish W&B run ---
    wandb.finish()


if __name__ == '__main__':
    # Make sure to log in to wandb first from your terminal:
    # wandb login
    train_model()