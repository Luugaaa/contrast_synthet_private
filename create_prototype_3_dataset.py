import os
import cv2
import numpy as np
import random
import math

# --- Configuration ---
IMG_SIZE = 64
NUM_IMAGES_PER_SET = 200
OUTPUT_DIR = "data_prototype_3"

def create_directories():
    """Creates the necessary output directories for images and labels."""
    os.makedirs(os.path.join(OUTPUT_DIR, "set1"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "set2"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "set1_labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "set2_labels"), exist_ok=True)
    print(f"Directories for images and labels are ready in '{OUTPUT_DIR}'.")

def get_rotated_points(points, center, angle_degrees):
    """Rotates a list of points around a center."""
    angle_rad = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    new_points = []
    for p in points:
        x, y = p[0] - center[0], p[1] - center[1]
        new_x = x * cos_a - y * sin_a + center[0]
        new_y = x * sin_a + y * cos_a + center[1]
        new_points.append([int(new_x), int(new_y)])
    return np.array(new_points, dtype=np.int32)

def generate_image_and_mask(set_type):
    """
    Generates a single complex image and its corresponding triangle label mask.
    The set_type ('set1' or 'set2') determines the triangle intensity.
    """
    
    # Initialize black images for the main image and the label mask
    image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    label_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # --- 1. Outer Ellipse (Contour) ---
    ellipse_center = (IMG_SIZE // 2, IMG_SIZE // 2)
    # Ensure longest axis is vertical
    ellipse_axes = (random.randint(IMG_SIZE // 3, int(IMG_SIZE / 2.5)), random.randint(int(IMG_SIZE / 2.2), int(IMG_SIZE / 2.1)))
    ellipse_angle = random.uniform(-15, 15)
    # Intensity: 0.9 to 1.0 (230-255)
    ellipse_intensity = random.randint(230, 255)
    cv2.ellipse(image, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, ellipse_intensity, 2)

    # --- 2. Inner Ellipse (Filled) ---
    inner_axes = (int(ellipse_axes[0] * 0.8), int(ellipse_axes[1] * 0.9))
    inner_angle = ellipse_angle + random.uniform(-2, 2) # Similar orientation
    # Intensity: 0.45 to 0.55 (115-140)
    inner_intensity = random.randint(115, 140)
    cv2.ellipse(image, ellipse_center, inner_axes, inner_angle, 0, 360, inner_intensity, -1)

    # --- 3. Triangles ---
    triangle_base_positions = [
        (ellipse_center[0] - 8, ellipse_center[1]),
        (ellipse_center[0] + 8, ellipse_center[1])
    ]

    for base_pos in triangle_base_positions:
        # Add slight positional jitter
        pos = (base_pos[0] + random.randint(-2, 2), base_pos[1] + random.randint(-1, 1))
        # Add slight rotational jitter
        angle = random.uniform(-10, 10)
        size = random.randint(8, 13)

        # Determine intensity based on the set type
        if set_type == 'set1':
            # Intensity: 0.3 to 0.4 (76-102)
            tri_intensity = random.randint(76, 102)
        else: # set_type == 'set2'
            # Intensity: 0.6 to 0.7 (153-178)
            tri_intensity = random.randint(153, 178)

        # Define triangle points
        h = size
        points = np.array([[0, -h//2], [-h//2, h//2], [h//2, h//2]])
        rot_points = get_rotated_points(points, (0, 0), angle) + pos
        
        # Draw on both the main image and the label mask
        cv2.fillPoly(image, [rot_points], tri_intensity)
        cv2.fillPoly(label_mask, [rot_points], 255) # Use 255 for the mask

    return image, label_mask

def generate_full_dataset():
    """Generates and saves the complete dataset with two independent sets."""
    create_directories()
    
    print(f"Generating {NUM_IMAGES_PER_SET} images and labels for each set...")

    for i in range(NUM_IMAGES_PER_SET):
        # --- Generate two independent images and their labels ---
        # Each call generates a completely new, random image
        img_set1, label_set1 = generate_image_and_mask('set1')
        img_set2, label_set2 = generate_image_and_mask('set2')

        # --- Save the images and labels for set 1 ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set1", f"mri_p3_{i:04d}.png"), img_set1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set1_labels", f"mri_p3_{i:04d}_label.png"), label_set1)
        
        # --- Save the images and labels for set 2 ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set2", f"mri_p3_{i:04d}.png"), img_set2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set2_labels", f"mri_p3_{i:04d}_label.png"), label_set2)

        if (i + 1) % 50 == 0:
            print(f"  ...generated {i+1}/{NUM_IMAGES_PER_SET} images for each set.")

    print("Complex dataset generation complete!")

if __name__ == '__main__':
    # Ensure you have opencv-python installed:
    # pip install opencv-python
    generate_full_dataset()
