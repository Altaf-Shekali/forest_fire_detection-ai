import os
from PIL import Image, UnidentifiedImageError

base_dir = "data/raw"

bad_files = []

print("Scanning images. Please wait...\n")

for split in ["Train", "Test"]:
    split_dir = os.path.join(base_dir, split)

    if not os.path.exists(split_dir):
        print(f"Missing folder: {split_dir}")
        continue

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)

            # Skip non-images
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                with Image.open(fpath) as img:
                    img.verify()  # Verify image integrity
            except (UnidentifiedImageError, OSError) as e:
                print("❌ BAD FILE:", fpath, "|", e)
                bad_files.append(fpath)

print("\n-------------------------------------")
print("Total corrupted images found:", len(bad_files))
print("-------------------------------------\n")

# Delete bad files
if bad_files:
    print("Deleting corrupted files...")
    for f in bad_files:
        try:
            os.remove(f)
            print("🗑 Deleted:", f)
        except Exception as e:
            print("⚠️ Could not delete:", f, "|", e)
else:
    print("No corrupted files detected.")

print("\nCleanup complete!")
