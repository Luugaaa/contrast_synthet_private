import os
import cv2
import numpy as np
import random
import math

# --- Configuration ---
IMG_SIZE = 64
NUM_IMAGES = 200  # Number of images to generate for each set
OUTPUT_DIR = "data_prototype_2"

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

def is_inside_ellipse(point, ellipse_center, axes, angle_degrees):
    """Checks if a point is inside a rotated ellipse."""
    angle_rad = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    # Translate point to origin
    px, py = point[0] - ellipse_center[0], point[1] - ellipse_center[1]
    
    # Rotate point
    rot_x = px * cos_a + py * sin_a
    rot_y = -px * sin_a + py * cos_a
    
    # Check ellipse equation
    a, b = axes[0], axes[1]
    return (rot_x**2 / a**2) + (rot_y**2 / b**2) <= 1

def generate_image_set(set_type):
    """
    Generates a single image and its corresponding triangle label mask
    based on the set type ('set1' or 'set2').
    """
    
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    label_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) # For triangle masks

    # Ellipse parameters
    ellipse_center = (IMG_SIZE // 2, IMG_SIZE // 2)
    ellipse_axes = (random.randint(IMG_SIZE // 4, IMG_SIZE // 3), random.randint(int(IMG_SIZE / 2.5), int(IMG_SIZE / 2.2)))
    ellipse_angle = random.uniform(-15, 15)
    cv2.ellipse(img, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, 255, 2)

    # Draw squares
    for _ in range(random.randint(2, 5)):
        size = random.randint(7, 15)
        angle = random.uniform(0, 360)
        while True:
            pos = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
            if is_inside_ellipse(pos, ellipse_center, ellipse_axes, ellipse_angle):
                break
        color = random.randint(180, 220)
        s = size // 2
        points = np.array([[-s, -s], [s, -s], [s, s], [-s, s]])
        rot_points = get_rotated_points(points, (0, 0), angle) + pos
        cv2.fillPoly(img, [rot_points], color)

    # Draw triangles with intensity based on set type
    for _ in range(random.randint(2, 5)):
        size = random.randint(7, 15)
        angle = random.uniform(0, 360)
        while True:
            pos = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
            if is_inside_ellipse(pos, ellipse_center, ellipse_axes, ellipse_angle):
                break
        
        if set_type == 'set1': # Dark triangles
            color = random.randint(40, 80)
        else: # Light triangles for set2
            color = random.randint(150, 200)

        h = size
        points = np.array([[0, -h//2], [-h//2, h//2], [h//2, h//2]])
        rot_points = get_rotated_points(points, (0, 0), angle) + pos
        
        # Draw on both the main image and the label mask
        cv2.fillPoly(img, [rot_points], color)
        cv2.fillPoly(label_img, [rot_points], 255) # Use 255 for the mask
        
    return img, label_img


def generate_dataset():
    """Generates and saves the complex dataset."""
    create_directories()
    
    print(f"Generating {NUM_IMAGES} images and labels for each set...")

    for i in range(NUM_IMAGES):
        # --- Generate two independent images and their labels ---
        img_set1, label_set1 = generate_image_set('set1')
        img_set2, label_set2 = generate_image_set('set2')

        # --- Save the images and labels ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set1", f"mri_sim_{i:04d}.png"), img_set1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set1_labels", f"mri_sim_{i:04d}_label.png"), label_set1)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set2", f"mri_sim_{i:04d}.png"), img_set2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "set2_labels", f"mri_sim_{i:04d}_label.png"), label_set2)

        if (i + 1) % 50 == 0:
            print(f"  ...generated {i+1}/{NUM_IMAGES} pairs.")

    print("Complex dataset generation complete!")

if __name__ == '__main__':
    # Ensure you have opencv-python installed:
    # pip install opencv-python
    generate_dataset()
