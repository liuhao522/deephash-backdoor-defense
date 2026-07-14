import os
import shutil

# Paths
train_txt = "D:/deephash_original/data/imagenet/origin/train.txt"
src_image_dir = "D:/deephash_original/dataset/imagenet/image/"
dst_image_dir = "D:/deephash_original/dataset/imagenet/image_ganjing/"

# Create destination directory if it doesn't exist
os.makedirs(dst_image_dir, exist_ok=True)

with open(train_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        # Get image path and binary vector
        img_rel_path = parts[0]
        binary_vector = parts[1:]

        # Find position of 1 in the binary vector
        try:
            label = binary_vector.index('1')
        except ValueError:
            print(f"No '1' found in line: {line}")
            continue

        # Get original filename and new filename
        original_filename = os.path.basename(img_rel_path)  # e.g. n03045698_16121.JPEG
        base_name = original_filename.split('_')[-1].split('.')[0]  # e.g. 16121
        new_filename = f"{base_name}-label-{label}.JPEG"

        # Full paths
        src_path = os.path.join(src_image_dir, original_filename)
        dst_path = os.path.join(dst_image_dir, new_filename)

        # Copy and rename
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied {original_filename} to {new_filename}")
        except FileNotFoundError:
            print(f"File not found: {src_path}")
        except Exception as e:
            print(f"Error processing {original_filename}: {e}")

print("Processing complete.")