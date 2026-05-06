import cv2
import numpy as np
import matplotlib.pyplot as plt

def experiment_crops(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For correct matplotlib display
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # METHOD 1: Basic Thresholding
    # ==========================================
    # 1. Threshold to separate bright areas from black
    _, thresh1 = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological close to fill small gaps in the fan
    kernel1 = np.ones((9,9), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel1)
    
    # 3. Find largest contour
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask1 = np.zeros_like(gray)
    if contours1:
        largest_c1 = max(contours1, key=cv2.contourArea)
        cv2.drawContours(mask1, [largest_c1], -1, 255, thickness=cv2.FILLED)
        
    # Apply mask and crop
    masked_img1 = cv2.bitwise_and(img_rgb, img_rgb, mask=mask1)
    x1, y1, w1, h1 = cv2.boundingRect(mask1)
    crop1 = masked_img1[y1:y1+h1, x1:x1+w1]

    # ==========================================
    # METHOD 2: Canny Edge + Convex Hull
    # ==========================================
    # 1. Find sharp edges
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    # 2. Morphological close to connect broken edge lines
    kernel2 = np.ones((15,15), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)
    
    # 3. Find contours and wrap them in a Convex Hull
    contours2, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros_like(gray)
    if contours2:
        all_points = np.vstack(contours2)
        hull = cv2.convexHull(all_points)
        cv2.drawContours(mask2, [hull], -1, 255, thickness=cv2.FILLED)
        
    # Apply mask and crop
    masked_img2 = cv2.bitwise_and(img_rgb, img_rgb, mask=mask2)
    x2, y2, w2, h2 = cv2.boundingRect(mask2)
    crop2 = masked_img2[y2:y2+h2, x2:x2+w2]

    # ==========================================
    # Plotting the Results
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Ultrasound Cropping Experiment", fontsize=16)

    # Row 1: Thresholding Method
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title("Method 1: Threshold Mask")
    axes[0, 2].imshow(crop1)
    axes[0, 2].set_title("Method 1: Final Crop")

    # Row 2: Canny Method
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title("Original Image")
    axes[1, 1].imshow(mask2, cmap='gray')
    axes[1, 1].set_title("Method 2: Canny Hull Mask")
    axes[1, 2].imshow(crop2)
    axes[1, 2].set_title("Method 2: Final Crop")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run the experiment
experiment_crops("./test_images/wrist.png")