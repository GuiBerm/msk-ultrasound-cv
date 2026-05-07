import os
import glob
import cv2
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your algorithms from the other file
import algorithms

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
TARGET_SIZE = (256, 256)
OUTPUT_EXT = ".png"
SAM_CHECKPOINT = "../cropping_tests/sam_vit_h_4b8939.pth"
LOG_FILENAME = "processing_log.csv"
# ==========================================

def pad_and_resize(cropped_img):
    """Pads image to a square with black borders, then resizes."""
    h, w = cropped_img.shape[:2]
    max_side = max(w, h)
    
    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left
    
    padded = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded, TARGET_SIZE)

def process_single_image(file_path, output_dir, choice, predictor=None):
    """
    Isolated function to process a single image. 
    Designed this way so it can be sent to parallel CPU cores.
    """
    filename = os.path.basename(file_path)
    
    # Initialize the log entry for this specific file
    log_data = {
        "filename": filename,
        "algorithm": choice,
        "status": "Failed",
        "detail": ""
    }

    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        log_data["detail"] = "Could not read image file. Corrupted?"
        return log_data

    # Calculate total area for sanity checks later
    h_orig, w_orig = img_bgr.shape[:2]
    total_area = h_orig * w_orig

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        # 1. Apply chosen algorithm
        if choice == '1':
            mask = algorithms.mask_threshold(gray)
        elif choice == '2':
            mask = algorithms.mask_opening(gray)
        elif choice == '3':
            mask = algorithms.mask_grabcut(img_bgr)
        elif choice == '4':
            mask = algorithms.mask_sam(img_rgb, predictor)

        # 2. Check for total failure
        if cv2.countNonZero(mask) == 0:
            log_data["detail"] = "Mask generated was completely blank."
            return log_data

        # 3. Crop and measure
        x, y, w, h = cv2.boundingRect(mask)
        crop_area = w * h

        # 4. Sanity Checks on Mask Size
        if crop_area < 0.10 * total_area:
            log_data["status"] = "Suspicious"
            log_data["detail"] = f"Crop too small ({int((crop_area/total_area)*100)}% of original)"
        elif crop_area > 0.95 * total_area:
            log_data["status"] = "Suspicious"
            log_data["detail"] = f"Crop too large ({int((crop_area/total_area)*100)}% of original)"
        else:
            log_data["status"] = "Success"
            log_data["detail"] = "Normal crop"

        # 5. Apply crop, pad, resize, and save
        masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        cropped = masked_img[y:y+h, x:x+w]
        final_img = pad_and_resize(cropped)

        out_name = os.path.splitext(filename)[0] + OUTPUT_EXT
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, final_img)

    except Exception as e:
        log_data["status"] = "Error"
        log_data["detail"] = f"Algorithm crashed: {str(e)}"

    return log_data

def process_directory():
    # Ask User for Inputs
    input_dir = input("Enter the path to the raw images directory: ").strip()
    if not os.path.exists(input_dir):
        print("Directory not found!")
        return

    output_dir = input("Enter the path for the output directory: ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nSelect Algorithm:")
    print("1: Basic Thresholding (CPU)")
    print("2: Morphological Opening (CPU)")
    print("3: GrabCut (CPU)")
    print("4: Segment Anything Model - SAM (GPU)")
    
    choice = input("Enter choice (1-4): ").strip()
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice.")
        return

    # Pre-load SAM if needed
    predictor = None
    if choice == '4':
        print("\nLoading SAM into memory... please wait.")
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print("SAM loaded successfully!\n")

    # Find Images
    search_pattern = os.path.join(input_dir, "*.*")
    image_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    print(f"Found {total_images} images. Starting batch process...\n")

    # Initialize CSV Logger
    csv_path = os.path.join(output_dir, LOG_FILENAME)
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'algorithm', 'status', 'detail']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        processed_count = 0

        # ==========================================
        # CPU MULTIPROCESSING (Methods 1, 2, 3)
        # ==========================================
        if choice in ['1', '2', '3']:
            print("Engaging Multi-Core Processing...")
            # ProcessPoolExecutor automatically uses all available CPU cores
            with ProcessPoolExecutor() as executor:
                # Submit all tasks to the pool
                futures = [executor.submit(process_single_image, f, output_dir, choice) for f in image_files]
                
                # As each image finishes (in any order), log it immediately
                for future in as_completed(futures):
                    log_data = future.result()
                    writer.writerow(log_data)
                    
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == total_images:
                        print(f"Processed {processed_count}/{total_images} images...")

        # ==========================================
        # GPU SEQUENTIAL PROCESSING (Method 4 - SAM)
        # ==========================================
        elif choice == '4':
            print("Engaging GPU Sequential Processing...")
            # We cannot easily multiprocess the 2.4GB PyTorch model across CPUs, 
            # so we run it sequentially to avoid crashing your VRAM.
            for f in image_files:
                log_data = process_single_image(f, output_dir, choice, predictor)
                writer.writerow(log_data)
                
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_images:
                    print(f"Processed {processed_count}/{total_images} images...")

    print(f"\nProcessing complete. Log saved to: {csv_path}")

if __name__ == "__main__":
    # The __main__ block is strictly required for ProcessPoolExecutor to work on Windows
    process_directory()