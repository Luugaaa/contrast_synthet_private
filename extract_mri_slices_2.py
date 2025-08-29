import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

def preprocess_mri_dataset(root_dir, output_dir, resize_shape=None):
    """
    Finds all anatomical NIfTI files in a BIDS-like directory, extracts 2D slices
    from all three views (axial, coronal, sagittal), normalizes them, and saves 
    them as PNG files in a single flat directory.

    Args:
        root_dir (str): The root directory of the dataset containing subject folders (e.g., '.../3d_mris/').
        output_dir (str): The directory where the processed PNG slices will be saved.
        resize_shape (tuple, optional): The target (width, height) to resize images. 
                                        If None, images are not resized. Defaults to None.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory is ready at: {output_dir}")

    # 2. Find all anatomical NIfTI files for all subjects
    search_pattern = os.path.join(root_dir, 'sub-*', 'anat', '*.nii.gz')
    nifti_files = sorted(glob.glob(search_pattern))

    if not nifti_files:
        print(f"⚠️ Warning: No NIfTI files found matching the pattern: {search_pattern}")
        return

    print(f"Found {len(nifti_files)} NIfTI files to process.")
    total_slices_saved = 0

    # 3. Process each NIfTI file
    for file_path in tqdm(nifti_files, desc="Processing MRI Volumes"):
        base_name = os.path.basename(file_path).replace('.nii.gz', '')
        
        try:
            # Load the 3D volume data
            mri_volume = nib.load(file_path).get_fdata(dtype=np.float32)
            
            # Ensure the volume is 3D
            if mri_volume.ndim != 3:
                print(f"\nSkipping {base_name}: not a 3D volume (dimensions: {mri_volume.shape}).")
                continue

            # Define the views and corresponding axes
            views = {
                'axial': 2,
                'coronal': 1,
                'sagittal': 0
            }

            # Iterate through each view
            for view_name, axis in views.items():
                num_slices = mri_volume.shape[axis]
                
                for slice_index in range(num_slices):
                    # Extract the 2D slice using np.take for axis-agnostic slicing
                    slice_2d = np.take(mri_volume, slice_index, axis=axis)
                    
                    # Some MRI orientations might need rotation for consistent viewing
                    # For sagittal and coronal, a 90-degree rotation is common
                    if view_name in ['sagittal', 'coronal']:
                         slice_2d = np.rot90(slice_2d)

                    # Skip empty slices
                    if np.all(slice_2d == 0):
                        continue

                    # Normalize the slice to the [0, 255] range
                    min_val, max_val = slice_2d.min(), slice_2d.max()
                    if max_val > min_val:
                        slice_normalized = 255.0 * (slice_2d - min_val) / (max_val - min_val)
                    else:
                        slice_normalized = np.zeros_like(slice_2d)
                    
                    # Convert to an 8-bit integer grayscale image
                    slice_uint8 = slice_normalized.astype(np.uint8)
                    pil_image = Image.fromarray(slice_uint8, mode='L')
                    
                    # Resize if a shape is provided
                    if resize_shape:
                        pil_image = pil_image.resize(resize_shape, Image.Resampling.LANCZOS)

                    # Define a unique output filename and save the image
                    output_filename = f"{base_name}_view-{view_name}_slice-{slice_index:03d}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    pil_image.save(output_path)
                    total_slices_saved += 1

        except Exception as e:
            print(f"\n❌ Error processing {file_path}: {e}")

    print(f"\n✨ Preprocessing complete. Total slices saved: {total_slices_saved}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update these paths to match your system
    BIDS_ROOT_DIR = '/home/paulh/contrast_synthet_private/datasets/3d_mris'
    PROCESSED_OUTPUT_DIR = 'processed_mri_slices_all_views'
    
    # Optional: Set to a tuple like (256, 256) to resize all images, or None to keep original size.
    RESIZE_DIMENSIONS = None 

    # --- Run the Preprocessing ---
    preprocess_mri_dataset(BIDS_ROOT_DIR, PROCESSED_OUTPUT_DIR, resize_shape=RESIZE_DIMENSIONS)