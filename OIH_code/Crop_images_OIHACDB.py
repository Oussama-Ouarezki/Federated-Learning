import os
from PIL import Image
import numpy as np

def find_white_content_bounds(img):
    """Find the bounds of white pixels in the image."""
    # Convert to numpy array for processing
    arr = np.array(img.convert('RGB'))
    
    # Define white threshold (pixels close to white)
    white_threshold = 200
    
    # Find white pixels (all RGB channels above threshold)
    is_white = np.all(arr >= white_threshold, axis=2)
    
    # Find rows and columns containing white
    rows = np.any(is_white, axis=1)
    cols = np.any(is_white, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # No white pixels found, return full image bounds
        return 0, img.height - 1, 0, img.width - 1
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return ymin, ymax, xmin, xmax

def crop_to_white_square(input_path, output_path):
    try:
        img = Image.open(input_path)
        
        # Find white content bounds
        ymin, ymax, xmin, xmax = find_white_content_bounds(img)
        
        # Calculate content dimensions
        content_width = xmax - xmin + 1
        content_height = ymax - ymin + 1
        
        # Determine square size (use the larger dimension to fit all white content)
        square_size = max(content_width, content_height)
        
        # Center the crop around the white content
        content_center_x = (xmin + xmax) // 2
        content_center_y = (ymin + ymax) // 2
        
        left = content_center_x - square_size // 2
        top = content_center_y - square_size // 2
        right = left + square_size
        bottom = top + square_size
        
        # Adjust if crop goes outside image bounds
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img.width:
            left -= (right - img.width)
            right = img.width
        if bottom > img.height:
            top -= (bottom - img.height)
            bottom = img.height
        
        # Final safety check
        left, top = max(0, left), max(0, top)
        right, bottom = min(img.width, right), min(img.height, bottom)
        square_size = min(right - left, bottom - top)
        
        print(f"Cropping {os.path.basename(input_path)}: white content at ({xmin},{ymin})-({xmax},{ymax}), crop to {square_size}x{square_size}")
        
        # Crop to square (no resize)
        img_cropped = img.crop((left, top, left + square_size, top + square_size))
        img_cropped.save(output_path)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    input_dir = "/home/oussama/Desktop/MLA2/INVERTED"
    output_dir = "/home/oussama/Desktop/MLA2/CROPPED_WHITE"
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            crop_to_white_square(input_path, output_path)
            count += 1
    
    print(f"✅ Done! Processed {count} images.")
    print(f"📁 Output folder: {output_dir}")

if __name__ == "__main__":
    main()