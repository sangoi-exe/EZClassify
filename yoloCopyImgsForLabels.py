import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def main():
    # Initialize and hide Tkinter root window
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # Prompt user to select the folder containing the label files
    labels_dir = filedialog.askdirectory(title="Select Labels Folder")
    if not labels_dir:
        print("No labels folder selected. Exiting.")
        return
    labels_path = Path(labels_dir)

    # Prompt user to select the folder containing the full dataset (images)
    dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
    if not dataset_dir:
        print("No dataset folder selected. Exiting.")
        return
    dataset_path = Path(dataset_dir)

    # Prompt user to select the destination folder to copy matching images
    dest_dir = filedialog.askdirectory(title="Select Destination Folder")
    if not dest_dir:
        print("No destination folder selected. Exiting.")
        return
    dest_path = Path(dest_dir)

    # Define possible image extensions to search
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"]

    copied_count = 0
    missing_files = []

    # Iterate through each label in the labels folder
    for label_file in labels_path.iterdir():
        # Proceed only if it's actually a file
        if label_file.is_file():
            # Extract the "stem" (name without extension) of the label file
            label_stem = label_file.stem

            # We assume each label has a corresponding image with one of the known extensions
            found_match = False
            for ext in image_extensions:
                image_candidate = dataset_path / f"{label_stem}{ext}"
                if image_candidate.exists() and image_candidate.is_file():
                    # Copy this image to the destination
                    try:
                        shutil.copy2(image_candidate, dest_path)
                        copied_count += 1
                        found_match = True
                        break
                    except Exception as e:
                        print(f"Error copying {image_candidate.name}: {e}")

            # If no matching image was found in the dataset folder
            if not found_match:
                missing_files.append(label_file.name)

    # Print summary of the operation
    print(f"Total images copied: {copied_count}")
    if missing_files:
        print("No corresponding images were found for the following labels:")
        for name in missing_files:
            print(name)


if __name__ == "__main__":
    main()
