import cv2
import numpy as np

def mask_threshold(gray_img):
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray_img)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask

def mask_opening(gray_img):
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray_img)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask

def mask_grabcut(img_bgr):
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w = img_bgr.shape[:2]
    
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask==2)|(mask==0), 0, 255).astype('uint8')

def mask_sam(img_rgb, predictor):
    """Expects the predictor to already be loaded in memory."""
    predictor.set_image(img_rgb)
    h, w = img_rgb.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=False
    )
    return (masks[0].astype(np.uint8) * 255)