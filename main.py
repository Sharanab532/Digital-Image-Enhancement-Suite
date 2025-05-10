"""
Image Processing Demonstration
Main execution script that showcases various image processing techniques
"""

import os
import argparse
import cv2
import matplotlib.pyplot as plt

from utils.visualization import plot_histogram, display_multiple_images
from processors.basic_ops import load_image, get_image_info
from processors.histogram_ops import equalize_histogram
from processors.intensity_ops import log_transform, gamma_correction
from processors.edge_ops import laplacian_edge, sobel_edge, sharpen_image

def create_output_dir(dir_name="output"):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_processed_image(image, filename, output_dir):
    """Save processed image to output directory"""
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        save_img = image
    cv2.imwrite(os.path.join(output_dir, filename), save_img)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Processing Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    
    # Load image
    img_path = args.image
    image = load_image(img_path)
    if image is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    # Get and display image information
    img_info = get_image_info(image)
    print("\n===== Image Information =====")
    print(f"Dimensions: {img_info['dimensions']}")
    print(f"Mean pixel value: {img_info['mean_value']:.4f}")
    print(f"Min pixel value: {img_info['min_value']}")
    print(f"Max pixel value: {img_info['max_value']}")
    print("=============================\n")
    
    # Create a dictionary to store all processed images
    processed_images = {'Original': image}
    
    # 1. Histogram equalization
    eq_img = equalize_histogram(image)
    processed_images['Histogram Equalized'] = eq_img
    save_processed_image(eq_img, "histogram_equalized.jpg", output_dir)
    
    # 2. Log transformation
    log_img = log_transform(image)
    processed_images['Log Transformed'] = log_img
    save_processed_image(log_img, "log_transformed.jpg", output_dir)
    
    # 3. Gamma correction with different values
    gamma_values = [0.2, 0.3, 0.4, 0.5]
    gamma_images = {}
    for gamma in gamma_values:
        gamma_img = gamma_correction(image, gamma)
        name = f'Gamma {gamma}'
        gamma_images[name] = gamma_img
        processed_images[name] = gamma_img
        save_processed_image(gamma_img, f"gamma_{gamma}.jpg", output_dir)
    
    # 4. Edge detection
    # a. Laplacian edge detection
    laplacian_img = laplacian_edge(image)
    processed_images['Laplacian Edge'] = laplacian_img
    save_processed_image(laplacian_img, "laplacian_edge.jpg", output_dir)
    
    # b. Sobel edge detection
    sobel_img = sobel_edge(image)
    processed_images['Sobel Edge'] = sobel_img
    save_processed_image(sobel_img, "sobel_edge.jpg", output_dir)
    
    # c. Image sharpening
    sharpened_img = sharpen_image(image)
    processed_images['Sharpened'] = sharpened_img
    save_processed_image(sharpened_img, "sharpened.jpg", output_dir)
    
    # Visualization
    # 1. Display original image and its histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plot_histogram(image, title='Original Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_with_histogram.jpg"))
    
    # 2. Display intensity transformations
    intensity_images = {
        'Original': image,
        'Log Transformed': log_img,
        'Histogram Equalized': eq_img
    }
    display_multiple_images(intensity_images, title='Intensity Transformations', 
                           output_path=os.path.join(output_dir, "intensity_transforms.jpg"))
    
    # 3. Display gamma transformations
    gamma_images['Original'] = image
    display_multiple_images(gamma_images, title='Gamma Corrections', 
                           output_path=os.path.join(output_dir, "gamma_transforms.jpg"))
    
    # 4. Display edge detection results
    edge_images = {
        'Original': image,
        'Laplacian Edge': laplacian_img,
        'Sobel Edge': sobel_img,
        'Sharpened': sharpened_img
    }
    display_multiple_images(edge_images, title='Edge Detection', 
                           output_path=os.path.join(output_dir, "edge_detection.jpg"))
    
    # 5. Display all results
    display_multiple_images(processed_images, rows=3, title='All Processed Images', 
                           output_path=os.path.join(output_dir, "all_results.jpg"))
    
    print(f"\nProcessed images saved to {output_dir}/")
    print("To display interactive plots, run with --show-plots option")

if __name__ == "__main__":
    main()