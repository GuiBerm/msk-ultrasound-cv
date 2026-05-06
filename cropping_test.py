import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_basic_threshold(gray_img):
    """Method 1: Basic thresholding. Prone to capturing text touching the fan."""
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    
    # Close small gaps
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray_img)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
    return mask

def extract_with_opening(gray_img):
    """Method 2: The 'Bridge Breaker'. Uses Morphological Opening to sever text."""
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    
    # 1. Morphological Opening (Erosion -> Dilation)
    # We use a large elliptical kernel to eat away text and numbers.
    # Increase the kernel size (e.g., (21, 21)) if the text in your images is very thick.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Find contours on the cleanly separated mask
    contours, _ = cv2.findContours(opened_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray_img)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
    return mask

def crop_with_mask(img_rgb, mask):
    """Applies a mask and returns the cropped image."""
    # If the mask is entirely empty, return the original
    if cv2.countNonZero(mask) == 0:
        return img_rgb
        
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    return masked_img[y:y+h, x:x+w]

def experiment_crops(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # Run Methods
    # ==========================================
    mask1 = extract_basic_threshold(gray)
    crop1 = crop_with_mask(img_rgb, mask1)

    mask2 = extract_with_opening(gray)
    crop2 = crop_with_mask(img_rgb, mask2)

    # ==========================================
    # Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Ultrasound Cropping: Fixing the 'Bridge' Problem", fontsize=16)

    # Row 1: Basic Method
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title("Method 1: Basic Mask (Text Bleeds)")
    axes[0, 2].imshow(crop1)
    axes[0, 2].set_title("Method 1: Final Crop")

    # Row 2: Bridge Breaker Method
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title("Original Image")
    axes[1, 1].imshow(mask2, cmap='gray')
    axes[1, 1].set_title("Method 2: Opening Mask (Text Severed)")
    axes[1, 2].imshow(crop2)
    axes[1, 2].set_title("Method 2: Clean Final Crop")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run the experiment
experiment_crops("./test_images/ganglion.png")