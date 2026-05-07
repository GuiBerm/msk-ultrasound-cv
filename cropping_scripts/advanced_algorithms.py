import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Import SAM components
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM dependencies not found. Please pip install torch segment-anything")

def extract_with_grabcut(img_bgr):
    """
    Method 3: Statistical separation.
    Uses a Gaussian Mixture Model to figure out what is background (UI/black) 
    and what is foreground (the textured ultrasound fan).
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    h, w = img_bgr.shape[:2]
    
    # Define a bounding box that slightly insets from the edges.
    # This tells GrabCut: "Everything outside this box is definitely background."
    margin_x = int(w * 0.1) # 10% margin
    margin_y = int(h * 0.1)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    # Run the algorithm for 5 iterations
    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # GrabCut modifies the mask with values 0-3. 
    # 0 & 2 are background, 1 & 3 are foreground.
    binary_mask = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
    
    return binary_mask

def extract_with_sam(img_rgb, checkpoint_path="sam_vit_h_4b8939.pth"):
    """
    Method 4: Meta's Segment Anything Model.
    Uses a massive Vision Transformer to zero-shot segment the fan 
    based on a single point prompt in the center.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: SAM checkpoint '{checkpoint_path}' not found.")
        return np.zeros(img_rgb.shape[:2], np.uint8)

    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM on {device}... this might take a moment.")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Pass image to the predictor
    predictor.set_image(img_rgb)
    
    # Prompt it with a single point right in the middle of the image
    h, w = img_rgb.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1]) # 1 = foreground
    
    # Predict
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    # Return the boolean mask as a standard 0-255 OpenCV mask
    return (masks[0].astype(np.uint8) * 255)

def crop_with_mask(img_rgb, mask):
    if cv2.countNonZero(mask) == 0:
        return img_rgb
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    return masked_img[y:y+h, x:x+w]

def experiment_advanced_crops(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # Run Methods
    # ==========================================
    print("Running GrabCut...")
    mask_grabcut = extract_with_grabcut(img_bgr)
    crop_grabcut = crop_with_mask(img_rgb, mask_grabcut)

    mask_sam = np.zeros_like(mask_grabcut)
    crop_sam = img_rgb.copy()
    if SAM_AVAILABLE:
        print("Running SAM...")
        mask_sam = extract_with_sam(img_rgb)
        crop_sam = crop_with_mask(img_rgb, mask_sam)

    # ==========================================
    # Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Advanced Cropping: Statistical vs. AI Models", fontsize=16)

    # Row 1: GrabCut
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(mask_grabcut, cmap='gray')
    axes[0, 1].set_title("Method 3: GrabCut Mask")
    axes[0, 2].imshow(crop_grabcut)
    axes[0, 2].set_title("Method 3: Final Crop")

    # Row 2: SAM
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title("Original Image")
    axes[1, 1].imshow(mask_sam, cmap='gray')
    axes[1, 1].set_title("Method 4: SAM Mask")
    axes[1, 2].imshow(crop_sam)
    axes[1, 2].set_title("Method 4: Final Crop")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run the experiment
try:
    experiment_advanced_crops(sys.argv[1])
except IndexError:
    print("Usage: python3 cropping_test_2.py <path_to_image>")