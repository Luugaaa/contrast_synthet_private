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
import cv2

class ComplexMRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the complex MRI-simulated images
    and their corresponding triangle segmentation masks.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # Combine images from both set1 and set2 into a single list
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "set1", "*.png")))
        self.image_paths += sorted(glob.glob(os.path.join(root_dir, "set2", "*.png")))
        
        self.label_paths = []
        for img_path in self.image_paths:
            # Construct the corresponding label path
            parts = img_path.split(os.sep)
            filename = os.path.basename(img_path)
            label_filename = filename.replace(".png", "_label.png")
            label_path = os.path.join(root_dir, parts[-2] + "_labels", label_filename)
            self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(img_path).convert("L")
        mask = Image.open(label_path).convert("L")
        
        # Convert mask to a binary format (0 for background, 1 for triangle)
        mask_np = np.array(mask)
        mask_np[mask_np > 0] = 255
        mask = Image.fromarray(mask_np)
        

        # Apply transformations
        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)
            # Squeeze to remove channel dimension from mask, and convert to long
            mask = mask.squeeze(0).long()
            
        return image, mask

# --- Training Configuration ---
DATA_DIR = "data_prototype_2"
MODEL_SAVE_PATH = "unet_prototype_2.pth"
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 50
VALIDATION_SPLIT = 0.15

def train_model():
    """Main function to run the training and validation loop."""
    wandb.init(
        project="unet-complex-mri-segmentation",
        config={
            "learning_rate": LEARNING_RATE, "architecture": "U-Net",
            "dataset": "Complex-MRI-Sim", "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # --- Data Loading ---
    transforms_dict = {
        'image': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'mask': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    
    full_dataset = ComplexMRIDataset(root_dir=DATA_DIR, transform=transforms_dict)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print(f"Dataset loaded: {train_size} training images, {val_size} validation images.")

    # --- Model, Loss, and Optimizer ---
    # 2 classes: 0=background, 1=triangle
    model = UNet(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        wandb_images = []
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                if i == 0:
                    pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                    for j in range(min(len(images), 8)):
                        wandb_images.append(wandb.Image(
                            images[j].cpu(),
                            masks={
                                "prediction": {"mask_data": pred_masks[j], "class_labels": {1: "triangle"}},
                                "ground_truth": {"mask_data": masks[j].cpu().numpy(), "class_labels": {1: "triangle"}}
                            }
                        ))

        avg_val_loss = val_loss / len(val_loader)
        
        wandb.log({
            "epoch": epoch + 1, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "predictions": wandb_images
        })
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    wandb.finish()


if __name__ == '__main__':
    # Make sure to log in to wandb first from your terminal:
    # wandb login
    train_model()
