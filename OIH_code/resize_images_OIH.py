import os
from PIL import Image

def resize_to_32x32(input_path, output_path):
    try:
        img = Image.open(input_path)
        
        # Resize to 32x32 using high-quality resampling
        img_resized = img.resize((32, 32), Image.Resampling.LANCZOS)
        img_resized.save(output_path)
        
        print(f"Resized {os.path.basename(input_path)} to 32x32")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    input_dir = "/home/oussama/Desktop/MLA2/CROPPED_WHITE"
    output_dir = "/home/oussama/Desktop/MLA2/RESIZED_32"
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    for file in os.listdir(input_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            resize_to_32x32(input_path, output_path)
            count += 1
    
    print(f"✅ Done! Resized {count} images to 32x32.")
    print(f"📁 Output folder: {output_dir}")

if __name__ == "__main__":
    main()