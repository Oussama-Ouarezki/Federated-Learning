import os
from PIL import Image, ImageOps

def invert_image(input_path, output_path):
    try:
        img = Image.open(input_path).convert("L")
        inverted = ImageOps.invert(img)
        inverted.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    input_dir = "/home/oussama/Desktop/MLA2/OIHACDB"
    output_dir = "/home/oussama/Desktop/MLA2/INVERTED"

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_path = os.path.join(root, file)

                # Avoid name collisions by prefixing a counter
                output_filename = f"{file}"
                output_path = os.path.join(output_dir, output_filename)

                invert_image(input_path, output_path)
                count += 1

    print(f"✅ Done! Inverted {count} images.")
    print(f"📁 Output folder: {output_dir}")

if __name__ == "__main__":
    main()


