"""
Visualization utilities for displaying images and plotting histograms
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def plot_histogram(image, title='Histogram', ax=None, save_path=None):
    """
    Plot the histogram of an image (RGB or grayscale)
    
    Args:
        image: Input image (RGB or grayscale)
        title: Plot title
        ax: Matplotlib axis (optional)
        save_path: Path to save the histogram (optional)
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
    
    if len(image.shape) == 3:  # RGB image
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=f'{color.upper()} Channel')
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='k', label='Intensity')
    
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 256])
    ax.grid(True)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    return ax

def display_multiple_images(images_dict, rows=None, title=None, figsize=(15, 10), output_path=None):
    """
    Display multiple images with their titles in a grid layout
    
    Args:
        images_dict: Dictionary of images with their titles as keys
        rows: Number of rows in the grid (optional)
        title: Figure title (optional)
        figsize: Figure size (optional)
        output_path: Path to save the figure (optional)
    """
    n_images = len(images_dict)
    
    if rows is None:
        rows = max(1, math.ceil(n_images / 3))
    
    cols = math.ceil(n_images / rows)
    
    plt.figure(figsize=figsize)
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(images_dict.items()):
        plt.subplot(rows, cols, i + 1)
        
        # Handle grayscale images
        if len(img.shape) == 2 or img.shape[2] == 1:
            plt.imshow(img, cmap='gray')
        else:  # RGB images
            plt.imshow(img)
        
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for the title
    
    if output_path:
        plt.savefig(output_path)
        
    return plt.gcf()

def display_image_with_histogram(image, title='Image with Histogram', figsize=(12, 5), output_path=None):
    """
    Display an image alongside its histogram
    
    Args:
        image: Input image
        title: Figure title
        figsize: Figure size
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Display image
    ax1 = fig.add_subplot(1, 2, 1)
    if len(image.shape) == 2 or image.shape[2] == 1:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    ax1.set_title('Image')
    ax1.axis('off')
    
    # Display histogram
    ax2 = fig.add_subplot(1, 2, 2)
    plot_histogram(image, ax=ax2)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig