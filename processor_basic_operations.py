"""
Basic image operations: loading, converting, and getting image information
"""

import cv2
import numpy as np

def load_image(path):
    """
    Load an image and convert to RGB
    
    Args:
        path: Path to the image file
    
    Returns:
        RGB image or None if loading fails
    """
    # Read image
    img = cv2.imread(path)
    
    # Check if image was loaded successfully
    if img is None:
        return None
    
    # Convert from BGR to RGB (OpenCV loads images in BGR format)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

def get_image_info(image):
    """
    Get basic information about an image
    
    Args:
        image: Input image
        
    Returns:
        Dictionary containing image information
    """
    info = {}
    
    # Dimensions
    info['dimensions'] = image.shape
    
    # Mean pixel value (normalized to [0, 1])
    mean_value = np.mean(image) / 255.0
    info['mean_value'] = mean_value
    
    # Min and max pixel values
    info['min_value'] = np.min(image)
    info['max_value'] = np.max(image)
    
    return info

def convert_to_grayscale(image):
    """
    Convert an image to grayscale
    
    Args:
        image: Input image (RGB or grayscale)
        
    Returns:
        Grayscale image
    """
    # Check if image is already grayscale
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

def resize_image(image, width=None, height=None, scale=None):
    """
    Resize an image
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        scale: Scale factor (optional)
        
    Returns:
        Resized image
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    if scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    elif width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        new_w = width
        new_h = int(h * (width / w))
    elif height is not None:
        new_h = height
        new_w = int(w * (height / h))
    else:
        return image  # No resize needed
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized