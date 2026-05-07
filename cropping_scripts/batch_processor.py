import os
import glob
import cv2
import numpy as np

# Import your algorithms from the other file
import algorithms

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
TARGET_SIZE = (256, 256)  # The final width x height for your Deep Learning model
OUTPUT_EXT = ".png"       # Change to ".jpg" if you prefer
SAM_CHECKPOINT = "../cropping_tests/sam_vit_h_4b8939.pth" # Path to SAM weights
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

def process_directory():
    # 1. Ask User for Inputs
    input_dir = input("Enter the path to the raw images directory: ").strip()
    if not os.path.exists(input_dir):
        print("Directory not found!")
        return

    output_dir = input("Enter the path for the output directory: ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nSelect Algorithm:")
    print("1: Basic Thresholding")
    print("2: Morphological Opening (Bridge Breaker)")
    print("3: GrabCut (Statistical)")
    print("4: Segment Anything Model (SAM)")
    
    choice = input("Enter choice (1-4): ").strip()

    # 2. Pre-load SAM if needed
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

    # 3. Find Images
    search_pattern = os.path.join(input_dir, "*.*")
    image_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(image_files)} images. Starting batch process...\n")

    # 4. Process Loop
    success_count = 0
    for file_path in image_files:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            continue
            
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply chosen algorithm
        if choice == '1':
            mask = algorithms.mask_threshold(gray)
        elif choice == '2':
            mask = algorithms.mask_opening(gray)
        elif choice == '3':
            mask = algorithms.mask_grabcut(img_bgr)
        elif choice == '4':
            mask = algorithms.mask_sam(img_rgb, predictor)
        else:
            print("Invalid choice.")
            return

        # Check if mask is empty
        if cv2.countNonZero(mask) == 0:
            print(f"Failed to find mask for {os.path.basename(file_path)}")
            continue

        # Crop, Pad, and Resize
        masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        x, y, w, h = cv2.boundingRect(mask)
        cropped = masked_img[y:y+h, x:x+w]
        
        final_img = pad_and_resize(cropped)

        # Save with the target extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_name = base_name + OUTPUT_EXT
        out_path = os.path.join(output_dir, out_name)
        
        cv2.imwrite(out_path, final_img)
        success_count += 1
        
        # Optional: Print progress for large batches
        if success_count % 10 == 0:
            print(f"Processed {success_count}/{len(image_files)}...")

    print(f"\nDone! Successfully processed {success_count} images.")

if __name__ == "__main__":
    process_directory()