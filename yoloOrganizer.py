import os
import shutil
from tkinter import Tk, filedialog
from ultralytics import YOLO
from tqdm import tqdm

# Path to the YOLO model weights
MODEL_PATH = "best.pt"
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.4
# Name of the main output folder that will contain subfolders for each detected class
OUTPUT_FOLDER = "organized_by_class"


def organize_images_by_class(input_folder):
    """
    Organizes images into subfolders based on detected classes.
    Creates a main output folder within the input folder and, for each detected class,
    creates a subfolder where images are copied.
    """
    output_root = os.path.join(input_folder, OUTPUT_FOLDER)
    os.makedirs(output_root, exist_ok=True)

    # Load YOLO model in detection mode and retrieve class names
    model = YOLO(MODEL_PATH, task="detect")
    class_names = model.names if hasattr(model, "names") else {}

    # Get list of image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    processed_count = 0  # Total images processed
    detect_count = 0  # Images with detections
    skip_count = 0  # Images skipped (no detections or errors)

    # Process each image with a progress bar displaying counters
    with tqdm(total=len(image_files), desc="Organizing images", ncols=80, dynamic_ncols=True) as pbar:
        for file_name in image_files:
            file_path = os.path.join(input_folder, file_name)
            processed_count += 1

            try:
                # Run inference on the image
                results = model(file_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
                detected_classes = set()
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for i in range(len(results[0].boxes)):
                        conf_score = results[0].boxes.conf[i]
                        if conf_score >= CONFIDENCE_THRESHOLD:
                            cls_id = int(results[0].boxes.cls[i])
                            class_label = class_names.get(cls_id, f"class_{cls_id}")
                            detected_classes.add(class_label)
                if detected_classes:
                    # Copy image to each class-specific folder
                    for class_label in detected_classes:
                        class_dir = os.path.join(output_root, class_label)
                        os.makedirs(class_dir, exist_ok=True)
                        shutil.copy2(file_path, class_dir)
                    detect_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                skip_count += 1

            # Update progress bar with detection, skip, and total counts
            pbar.set_postfix(detect=detect_count, skip=skip_count, total=processed_count)
            pbar.update(1)

    print(f"Processed {processed_count} images; organized {detect_count} images; skipped {skip_count} images.")


def select_folder():
    """
    Opens a file dialog to select the input folder.
    """
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


if __name__ == "__main__":
    selected_folder = select_folder()
    if selected_folder:
        organize_images_by_class(selected_folder)
    else:
        print("No folder selected.")
