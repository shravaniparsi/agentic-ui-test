#!/usr/bin/env python3
"""
resize_images.py - Resize screenshots and references to matched resolution.

This addresses the visual reference resolution confound by ensuring all images
are the same size (800x446 JPEG) before comparison.
"""

import os
from pathlib import Path
from PIL import Image

# Target dimensions
TARGET_WIDTH = 800
TARGET_HEIGHT = 446
TARGET_FORMAT = 'JPEG'
TARGET_QUALITY = 85


def resize_image(input_path, output_path):
    """Resize an image to target dimensions."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (for JPEG)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize to target dimensions
            resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
            
            # Save as JPEG
            resized.save(output_path, TARGET_FORMAT, quality=TARGET_QUALITY)
            return True
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")
        return False


def main():
    # Create output directories
    output_screenshots = Path('data/screenshots_resized')
    output_references = Path('data/references_resized')
    output_screenshots.mkdir(exist_ok=True)
    output_references.mkdir(exist_ok=True)
    
    # Resize screenshots
    print("Resizing screenshots...")
    screenshot_dir = Path('data/screenshots')
    resized_count = 0
    for img_path in sorted(screenshot_dir.glob('*.png')):
        output_path = output_screenshots / f"{img_path.stem}.jpg"
        if resize_image(img_path, output_path):
            resized_count += 1
    print(f"  Resized {resized_count} screenshots")
    
    # Resize references
    print("Resizing references...")
    reference_dir = Path('data/references')
    resized_count = 0
    for img_path in sorted(reference_dir.glob('*.png')):
        output_path = output_references / f"{img_path.stem}.jpg"
        if resize_image(img_path, output_path):
            resized_count += 1
    print(f"  Resized {resized_count} references")
    
    # Verify dimensions
    print("\nVerifying dimensions...")
    for img_path in list(output_screenshots.glob('*.jpg'))[:3]:
        with Image.open(img_path) as img:
            print(f"  {img_path.name}: {img.size}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
