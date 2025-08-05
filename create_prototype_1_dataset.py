import os
import numpy as np
from PIL import Image
import random
import math

# --- Configuration ---
IMG_SIZE = 64
MAX_SHAPE_SIZE = 16
NUM_IMAGES_PER_CLASS = 300  # Number of images to generate for each class
OUTPUT_DIR = "data"

def create_directories():
    """Creates the necessary output directories."""
    os.makedirs(os.path.join(OUTPUT_DIR, "disk"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "square"), exist_ok=True)
    print(f"Directories '{OUTPUT_DIR}/disk' and '{OUTPUT_DIR}/square' are ready.")

def draw_square(image_array, center_x, center_y, size, color):
    """Draws a filled square on a numpy array."""
    half_size = size // 2
    start_x = max(0, center_x - half_size)
    end_x = min(IMG_SIZE, center_x + half_size)
    start_y = max(0, center_y - half_size)
    end_y = min(IMG_SIZE, center_y + half_size)
    image_array[start_y:end_y, start_x:end_x] = color

def draw_disk(image_array, center_x, center_y, radius, color):
    """Draws a filled disk (circle) on a numpy array."""
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if math.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius:
                image_array[y, x] = color

def generate_dataset():
    """Generates and saves the dataset of disks and squares."""
    create_directories()
    
    print(f"Generating {NUM_IMAGES_PER_CLASS} images for each class...")

    for i in range(NUM_IMAGES_PER_CLASS):
        # --- Randomize Colors ---
        # Decide if the background is black (0) or white (255)
        if random.random() > 0.5:
            bg_color, shape_color = 255, 0 # Black shape on white bg
        else:
            bg_color, shape_color = 0, 255   # White shape on black bg

        # ==================================
        # 1. Generate and Save a Square
        # ==================================
        
        # Create a blank image with the background color
        square_img_arr = np.full((IMG_SIZE, IMG_SIZE), bg_color, dtype=np.uint8)
        
        # Randomize shape properties, ensuring it's not cropped
        square_size = random.randint(5, MAX_SHAPE_SIZE)
        margin = square_size // 2
        center_x_sq = random.randint(margin, IMG_SIZE - margin - 1)
        center_y_sq = random.randint(margin, IMG_SIZE - margin - 1)
        
        # Draw the square
        draw_square(square_img_arr, center_x_sq, center_y_sq, square_size, shape_color)
        
        # Convert to PIL Image and save
        square_img = Image.fromarray(square_img_arr)
        square_path = os.path.join(OUTPUT_DIR, "square", f"square_{i:04d}.png")
        square_img.save(square_path)

        # ==================================
        # 2. Generate and Save a Disk
        # ==================================

        # Create a blank image
        disk_img_arr = np.full((IMG_SIZE, IMG_SIZE), bg_color, dtype=np.uint8)
        
        # Randomize shape properties, ensuring it's not cropped
        disk_radius = random.randint(3, MAX_SHAPE_SIZE // 2)
        margin = disk_radius
        center_x_dk = random.randint(margin, IMG_SIZE - margin - 1)
        center_y_dk = random.randint(margin, IMG_SIZE - margin - 1)
        
        # Draw the disk
        draw_disk(disk_img_arr, center_x_dk, center_y_dk, disk_radius, shape_color)
        
        # Convert to PIL Image and save
        disk_img = Image.fromarray(disk_img_arr)
        disk_path = os.path.join(OUTPUT_DIR, "disk", f"disk_{i:04d}.png")
        disk_img.save(disk_path)

        if (i + 1) % 100 == 0:
            print(f"  ...generated {i+1}/{NUM_IMAGES_PER_CLASS} images.")

    print("Dataset generation complete!")


if __name__ == '__main__':
    generate_dataset()
