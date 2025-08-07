import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def preprocess_bids_mri_data(bids_root_dir, output_dir, output_size=(256, 256)):
    """
    Extracts 2D slices from 3D NIfTI files, processes them, and saves them as PNG images.

    Args:
        bids_root_dir (str): The root directory of the BIDS subject data (e.g., '.../sub-01/').
        output_dir (str): The directory where the processed PNG files will be saved.
        output_size (tuple): The target size (height, width) for the output images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")

    # Define the image resizing transform
    resizer = transforms.Resize(output_size)

    # Find all relevant NIfTI files
    search_pattern = os.path.join(bids_root_dir, 'anat', '*_part-mag_*T2starw.nii.gz')
    nifti_files = sorted(glob.glob(search_pattern))

    if not nifti_files:
        print(f"Error: No NIfTI files found matching pattern: {search_pattern}")
        return

    print(f"Found {len(nifti_files)} NIfTI files to process.")
    total_slices_processed = 0

    # Process each NIfTI file
    for file_path in tqdm(nifti_files, desc="Processing NIfTI Volumes"):
        # Extract a unique identifier from the filename (e.g., 'sub-01_part-mag_chunk-01_T2starw')
        base_name = os.path.basename(file_path).replace('.nii.gz', '')
        
        try:
            # Load the 3D volume
            nifti_volume = nib.load(file_path).get_fdata().astype(np.float32)
            num_slices = nifti_volume.shape[2]

            # Process each 2D slice in the volume
            for slice_idx in range(num_slices):
                mri_slice = nifti_volume[:, slice_idx, :]

                # Normalize the slice to [0, 255] for saving as a standard image
                min_val, max_val = mri_slice.min(), mri_slice.max()
                if max_val > min_val:
                    mri_slice = 255.0 * (mri_slice - min_val) / (max_val - min_val)
                
                # Convert to an 8-bit integer NumPy array
                mri_slice_uint8 = mri_slice.astype(np.uint8)
                # Convert to a PIL Image (grayscale)
                pil_image = Image.fromarray(mri_slice_uint8, mode='L')
                
                # Resize the image
                # resized_image = resizer(pil_image)
                resized_image = pil_image

                # Define a clear and unique output filename
                output_filename = f"{base_name}_slice-{slice_idx:03d}.png"
                output_save_path = os.path.join(output_dir, output_filename)

                # Save the processed image
                resized_image.save(output_save_path)
                total_slices_processed += 1

        except Exception as e:
            print(f"\nWarning: Failed to process {file_path}. Error: {e}")

    print(f"\nPreprocessing complete. Processed and saved {total_slices_processed} slices.")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update these paths to match your system
    BIDS_SOURCE_DIR = "datasets/processed_BIDS_full/sub-01/"
    PROCESSED_OUTPUT_DIR = "datasets/processed_png_raw/"

    # Run the preprocessing function
    preprocess_bids_mri_data(BIDS_SOURCE_DIR, PROCESSED_OUTPUT_DIR)
