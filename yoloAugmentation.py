import cv2
import os
import random
import numpy as np
import sys
import tkinter as tk
import concurrent.futures
from tqdm import tqdm
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor


def clip_polygon_to_image(polygon, width, height):
    """
    Clips a polygon to the [0,0] - [width-1, height-1] rectangle using
    the Sutherland–Hodgman algorithm. Returns a new list of (x, y) points
    after clipping. If the result is empty or has fewer than 3 points,
    it means the polygon lies completely outside or has collapsed.
    """

    # Define clipping boundaries as four edges:
    # Left=0, Right=width-1, Top=0, Bottom=height-1
    # Each boundary is (check_function, inside_value)
    # where check_function(pt) returns True if pt is inside that boundary.
    def inside_left(pt):
        return pt[0] >= 0

    def inside_right(pt):
        return pt[0] <= width - 1

    def inside_top(pt):
        return pt[1] >= 0

    def inside_bottom(pt):
        return pt[1] <= height - 1

    # Each boundary is a dict with "is_inside" to test inside, and "compute_intersect" to compute intersection
    # with the boundary line. The boundary line is x=0 or x=width-1 or y=0 or y=height-1
    # We'll clip one edge at a time in the order: left, right, top, bottom.
    def intersect_with_vertical(p1, p2, x_const):
        """Compute intersection with vertical boundary x = x_const."""
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 1e-7:
            # Vertical segment: just use the boundary's x
            return (x_const, y1)
        # Parameter t for the segment
        t = (x_const - x1) / (x2 - x1)
        y_clip = y1 + t * (y2 - y1)
        return (x_const, y_clip)

    def intersect_with_horizontal(p1, p2, y_const):
        """Compute intersection with horizontal boundary y = y_const."""
        x1, y1 = p1
        x2, y2 = p2
        if abs(y2 - y1) < 1e-7:
            # Horizontal segment: just use the boundary's y
            return (x1, y_const)
        # Parameter t for the segment
        t = (y_const - y1) / (y2 - y1)
        x_clip = x1 + t * (x2 - x1)
        return (x_clip, y_const)

    def clip_polygon(poly_points, is_inside_func, intersect_func):
        """Sutherland–Hodgman clip for one boundary."""
        clipped = []
        if not poly_points:
            return clipped
        prev_point = poly_points[-1]
        prev_inside = is_inside_func(prev_point)
        for current_point in poly_points:
            curr_inside = is_inside_func(current_point)
            if curr_inside:
                # If current point is inside and previous was outside, add the intersection first
                if not prev_inside:
                    intersec = intersect_func(prev_point, current_point)
                    clipped.append(intersec)
                # Then add current point
                clipped.append(current_point)
            else:
                # Current point is outside; if previous was inside, compute intersection
                if prev_inside:
                    intersec = intersect_func(prev_point, current_point)
                    clipped.append(intersec)
            prev_point = current_point
            prev_inside = curr_inside
        return clipped

    # 1) Clip left
    polygon = clip_polygon(polygon, inside_left, lambda p1, p2: intersect_with_vertical(p1, p2, 0))
    # 2) Clip right
    polygon = clip_polygon(polygon, inside_right, lambda p1, p2: intersect_with_vertical(p1, p2, width - 1))
    # 3) Clip top
    polygon = clip_polygon(polygon, inside_top, lambda p1, p2: intersect_with_horizontal(p1, p2, 0))
    # 4) Clip bottom
    polygon = clip_polygon(polygon, inside_bottom, lambda p1, p2: intersect_with_horizontal(p1, p2, height - 1))

    # Result is the clipped polygon
    return polygon


# Converts a segmentation label line from normalized coordinates to absolute coordinates.
# Expected format: <class> x1 y1 x2 y2 ... xn yn
def seg_to_abs(label_line, img_w, img_h):
    tokens = label_line.strip().split()
    if len(tokens) < 3 or (len(tokens) - 1) % 2 != 0:
        raise ValueError("Label line must contain a class id and an even number of coordinate values.")
    cls_id = int(tokens[0])
    coords = list(map(float, tokens[1:]))
    polygon = []
    for i in range(0, len(coords), 2):
        x_abs = coords[i] * img_w
        y_abs = coords[i + 1] * img_h
        polygon.append((x_abs, y_abs))
    return cls_id, polygon


# Converts absolute segmentation polygon coordinates to normalized values.
def abs_to_seg(cls_id, polygon, img_w, img_h):
    norm_coords = []
    for x, y in polygon:
        norm_coords.append(f"{x / img_w:.6f}")
        norm_coords.append(f"{y / img_h:.6f}")
    return " ".join([str(cls_id)] + norm_coords)


# Adjusts brightness; geometry remains unchanged.
def apply_random_brightness(image, brightness_range=0.2):
    factor = 1.0 + random.uniform(-brightness_range, brightness_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


# Adjusts contrast; geometry remains unchanged.
def apply_random_contrast(image, contrast_range=0.2):
    factor = np.float32(1.0 + random.uniform(-contrast_range, contrast_range))
    img_f32 = image.astype(np.float32)
    mean_val = np.mean(img_f32, axis=(0, 1), keepdims=True).astype(np.float32)
    adjusted_f32 = (img_f32 - mean_val) * factor + mean_val
    adjusted_f32 = np.clip(adjusted_f32, 0, 255)
    return adjusted_f32.astype(np.uint8)


# Adjusts saturation; geometry remains unchanged.
def apply_random_saturation(image, saturation_range=0.2):
    factor = 1.0 + random.uniform(-saturation_range, saturation_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


# Adjusts hue; geometry remains unchanged.
def apply_random_hue(image, hue_shift_limit=10.0):
    shift = random.uniform(-hue_shift_limit, hue_shift_limit)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] += shift
    hsv[:, :, 0] = np.where(hsv[:, :, 0] < 0, hsv[:, :, 0] + 180, hsv[:, :, 0])
    hsv[:, :, 0] = np.where(hsv[:, :, 0] > 180, hsv[:, :, 0] - 180, hsv[:, :, 0])
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


# Randomly flips the image and applies the same transformation to segmentation polygons.
def apply_random_flip_seg(image, segments):
    """
    Performs a random horizontal or vertical flip of the image.
    Polygon coordinates are transformed via the same operation,
    then clipped to image boundaries.
    """
    img_h, img_w = image.shape[:2]
    flip_type = np.random.choice(["h", "v"])  # or random.choice

    if flip_type == "h":
        # Flip horizontally
        flipped = cv2.flip(image, 1)
        new_segments = []
        for cls_id, polygon in segments:
            # Flip X: x -> (img_w - 1) - x
            transformed_poly = [((img_w - 1) - x, y) for (x, y) in polygon]
            # Properly clip the polygon
            clipped_poly = clip_polygon_to_image(transformed_poly, img_w, img_h)
            if len(clipped_poly) >= 3:
                new_segments.append((cls_id, clipped_poly))
        return flipped, new_segments
    else:
        # Flip vertically
        flipped = cv2.flip(image, 0)
        new_segments = []
        for cls_id, polygon in segments:
            # Flip Y: y -> (img_h - 1) - y
            transformed_poly = [(x, (img_h - 1) - y) for (x, y) in polygon]
            # Properly clip the polygon
            clipped_poly = clip_polygon_to_image(transformed_poly, img_w, img_h)
            if len(clipped_poly) >= 3:
                new_segments.append((cls_id, clipped_poly))
        return flipped, new_segments


# Randomly rotates the image and applies the same affine transformation to segmentation polygons.
def apply_random_rotation_seg(image, segments, max_angle=15.0):
    """
    Rotates image within a random angle range. Polygon points are rotated
    via the same matrix. The final polygon is then clipped to the new image
    boundaries. The new image dimension may differ from the original.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    img_h, img_w = image.shape[:2]
    center = (img_w / 2.0, img_h / 2.0)

    # Compute rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])

    # New dimensions after rotation
    new_w = int((img_h * sin_val) + (img_w * cos_val))
    new_h = int((img_h * cos_val) + (img_w * sin_val))

    # Adjust translation so that the image remains centered
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    # Rotate the image
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=(127, 127, 127))

    # Rotate polygons and then clip
    new_segments = []
    for cls_id, polygon in segments:
        pts = np.array(polygon, dtype=np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_aug = np.hstack([pts, ones])
        transformed = rot_mat.dot(pts_aug.T).T
        # Convert to list of (x, y)
        transformed_poly = [(float(p[0]), float(p[1])) for p in transformed]
        # Clip to new boundaries
        clipped_poly = clip_polygon_to_image(transformed_poly, new_w, new_h)
        if len(clipped_poly) >= 3:
            new_segments.append((cls_id, clipped_poly))

    return rotated, new_segments


# Applies a random crop jitter to the image and adjusts segmentation polygons accordingly.
def apply_random_crop_jitter_seg(image, segments, crop_jitter_factor=0.1):
    """
    Applies random cropping on each boundary by up to crop_jitter_factor * dimension.
    Then transforms polygon coordinates accordingly and clips them.
    """
    img_h, img_w = image.shape[:2]

    # Calculate random crop boundaries
    jitter_w = int(crop_jitter_factor * img_w)
    jitter_h = int(crop_jitter_factor * img_h)
    left_cut = np.random.randint(0, jitter_w + 1)
    right_cut = np.random.randint(0, jitter_w + 1)
    top_cut = np.random.randint(0, jitter_h + 1)
    bottom_cut = np.random.randint(0, jitter_h + 1)

    # Final crop coords
    new_x_min = left_cut
    new_y_min = top_cut
    new_x_max = img_w - right_cut
    new_y_max = img_h - bottom_cut

    # Safety check (avoid invalid slice if sums exceed dimensions)
    if new_x_min >= new_x_max or new_y_min >= new_y_max:
        # If jitter is too large, just return original
        return image, segments

    # Cropped image
    cropped = image[new_y_min:new_y_max, new_x_min:new_x_max]
    new_img_h, new_img_w = cropped.shape[:2]

    # Adjust and clip polygons
    new_segments = []
    for cls_id, polygon in segments:
        transformed_poly = []
        # Shift each point by crop offsets
        for x, y in polygon:
            new_x = x - new_x_min
            new_y = y - new_y_min
            transformed_poly.append((new_x, new_y))
        # Clip to the new image region [0, 0, new_img_w-1, new_img_h-1]
        clipped_poly = clip_polygon_to_image(transformed_poly, new_img_w, new_img_h)
        if len(clipped_poly) >= 3:
            new_segments.append((cls_id, clipped_poly))

    return cropped, new_segments


# Processes one image and its segmentation label file, applying selected augmentations.
def process_image(image_path, label_path, output_images_dir, output_labels_dir, augmentations, copies, params):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}, skipping.")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    img_h, img_w = image.shape[:2]
    segments = []
    for l in lines:
        if not l.strip():
            continue
        try:
            seg = seg_to_abs(l, img_w, img_h)
        except Exception as e:
            print(f"Error processing label line in {label_path}: {e}")
            continue
        segments.append(seg)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    counter = 0

    for aug in augmentations:
        for _ in range(copies):
            aug_img = image.copy()
            aug_segments = segments.copy()

            if aug == "flip":
                aug_img, aug_segments = apply_random_flip_seg(aug_img, aug_segments)
            elif aug == "rotation":
                aug_img, aug_segments = apply_random_rotation_seg(aug_img, aug_segments, params["max_rotation_angle"])
            elif aug == "crop_jitter":
                aug_img, aug_segments = apply_random_crop_jitter_seg(aug_img, aug_segments, params["crop_jitter_factor"])
            elif aug == "brightness":
                aug_img = apply_random_brightness(aug_img, params["brightness_range"])
            elif aug == "contrast":
                aug_img = apply_random_contrast(aug_img, params["contrast_range"])
            elif aug == "saturation":
                aug_img = apply_random_saturation(aug_img, params["saturation_range"])
            elif aug == "hue":
                aug_img = apply_random_hue(aug_img, params["hue_shift_limit"])

            new_h, new_w = aug_img.shape[:2]
            new_label_lines = []
            for cls_id, polygon in aug_segments:
                new_label_lines.append(abs_to_seg(cls_id, polygon, new_w, new_h))

            out_img_name = f"{base_name}_aug_{aug}_{counter}.jpg"
            out_img_path = os.path.join(output_images_dir, out_img_name)
            cv2.imwrite(out_img_path, aug_img)

            out_label_name = f"{base_name}_aug_{aug}_{counter}.txt"
            out_label_path = os.path.join(output_labels_dir, out_label_name)
            with open(out_label_path, "w") as lf:
                for line in new_label_lines:
                    lf.write(line + "\n")
            counter += 1


# Creates a Tkinter GUI to select folders and set augmentation options/parameters.
def get_parameters_from_gui():
    root = tk.Tk()
    root.title("Data Augmentation Setup")
    root.attributes("-topmost", True)

    # Variables for folder paths.
    images_path_var = tk.StringVar()
    labels_path_var = tk.StringVar()

    # Folder selection functions.
    def select_images_folder():
        path = filedialog.askdirectory(title="Select Images Folder")
        if path:
            images_path_var.set(path)

    def select_labels_folder():
        path = filedialog.askdirectory(title="Select Labels Folder")
        if path:
            labels_path_var.set(path)

    # Folder selection frame.
    folder_frame = tk.Frame(root)
    folder_frame.pack(padx=10, pady=5, fill="x")
    tk.Label(folder_frame, text="Images Folder:").grid(row=0, column=0, sticky="w")
    tk.Entry(folder_frame, textvariable=images_path_var, width=50).grid(row=0, column=1, padx=5)
    tk.Button(folder_frame, text="Select", command=select_images_folder).grid(row=0, column=2)
    tk.Label(folder_frame, text="Labels Folder:").grid(row=1, column=0, sticky="w")
    tk.Entry(folder_frame, textvariable=labels_path_var, width=50).grid(row=1, column=1, padx=5)
    tk.Button(folder_frame, text="Select", command=select_labels_folder).grid(row=1, column=2)

    # Augmentation options frame.
    aug_frame = tk.LabelFrame(root, text="Augmentations")
    aug_frame.pack(padx=10, pady=5, fill="x")
    flip_var = tk.BooleanVar()
    rotation_var = tk.BooleanVar()
    brightness_var = tk.BooleanVar()
    contrast_var = tk.BooleanVar()
    saturation_var = tk.BooleanVar()
    hue_var = tk.BooleanVar()
    crop_jitter_var = tk.BooleanVar()
    tk.Checkbutton(aug_frame, text="Flip", variable=flip_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Rotation", variable=rotation_var).grid(row=0, column=1, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Brightness", variable=brightness_var).grid(row=0, column=2, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Contrast", variable=contrast_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Saturation", variable=saturation_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Hue", variable=hue_var).grid(row=1, column=2, sticky="w", padx=5, pady=2)
    tk.Checkbutton(aug_frame, text="Crop Jitter", variable=crop_jitter_var).grid(row=2, column=0, sticky="w", padx=5, pady=2)

    # Augmentation parameters frame.
    param_frame = tk.LabelFrame(root, text="Augmentation Parameters")
    param_frame.pack(padx=10, pady=5, fill="x")
    copies_var = tk.StringVar(value="10")
    max_rotation_angle_var = tk.StringVar(value="15.0")
    brightness_range_var = tk.StringVar(value="0.2")
    contrast_range_var = tk.StringVar(value="0.2")
    saturation_range_var = tk.StringVar(value="0.2")
    hue_shift_limit_var = tk.StringVar(value="10.0")
    crop_jitter_factor_var = tk.StringVar(value="0.1")
    tk.Label(param_frame, text="Copies:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=copies_var, width=10).grid(row=0, column=1, padx=5, pady=2)
    tk.Label(param_frame, text="Max Rotation Angle:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=max_rotation_angle_var, width=10).grid(row=0, column=3, padx=5, pady=2)
    tk.Label(param_frame, text="Brightness Range:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=brightness_range_var, width=10).grid(row=1, column=1, padx=5, pady=2)
    tk.Label(param_frame, text="Contrast Range:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=contrast_range_var, width=10).grid(row=1, column=3, padx=5, pady=2)
    tk.Label(param_frame, text="Saturation Range:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=saturation_range_var, width=10).grid(row=2, column=1, padx=5, pady=2)
    tk.Label(param_frame, text="Hue Shift Limit:").grid(row=2, column=2, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=hue_shift_limit_var, width=10).grid(row=2, column=3, padx=5, pady=2)
    tk.Label(param_frame, text="Crop Jitter Factor:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(param_frame, textvariable=crop_jitter_factor_var, width=10).grid(row=3, column=1, padx=5, pady=2)

    params = {}

    def on_ok():
        if not images_path_var.get() or not labels_path_var.get():
            messagebox.showerror("Error", "Please select both Images and Labels folders.")
            return
        params["images"] = images_path_var.get()
        params["labels"] = labels_path_var.get()
        params["flip"] = flip_var.get()
        params["rotation"] = rotation_var.get()
        params["brightness"] = brightness_var.get()
        params["contrast"] = contrast_var.get()
        params["saturation"] = saturation_var.get()
        params["hue"] = hue_var.get()
        params["crop_jitter"] = crop_jitter_var.get()
        try:
            params["copies"] = int(copies_var.get())
            params["max_rotation_angle"] = float(max_rotation_angle_var.get())
            params["brightness_range"] = float(brightness_range_var.get())
            params["contrast_range"] = float(contrast_range_var.get())
            params["saturation_range"] = float(saturation_range_var.get())
            params["hue_shift_limit"] = float(hue_shift_limit_var.get())
            params["crop_jitter_factor"] = float(crop_jitter_factor_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for parameters.")
            return
        root.destroy()

    tk.Button(root, text="OK", command=on_ok).pack(pady=10)
    root.mainloop()
    return params


def main():
    """
    Displays a Tkinter GUI to gather augmentation parameters, então
    processa todas as imagens em paralelo usando ThreadPoolExecutor.
    """
    params = get_parameters_from_gui()

    # Saída igual aos diretórios de entrada
    output_images_dir = params["images"]
    output_labels_dir = params["labels"]

    # Criação da lista de augmentations selecionadas
    aug_list = []
    if params.get("flip"):
        aug_list.append("flip")
    if params.get("rotation"):
        aug_list.append("rotation")
    if params.get("brightness"):
        aug_list.append("brightness")
    if params.get("contrast"):
        aug_list.append("contrast")
    if params.get("saturation"):
        aug_list.append("saturation")
    if params.get("hue"):
        aug_list.append("hue")
    if params.get("crop_jitter"):
        aug_list.append("crop_jitter")

    # Caso não haja nenhuma seleção, encerra
    if not aug_list:
        print("No augmentations selected. Exiting.")
        sys.exit(0)

    # Lista todas as imagens
    images_files = [f for f in os.listdir(params["images"]) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Dispara processamentos em paralelo
    futures = []
    with ThreadPoolExecutor() as executor:
        for img_file in images_files:
            img_path = os.path.join(params["images"], img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(params["labels"], label_file)

            # Verifica existência do arquivo de label
            if not os.path.exists(label_path):
                print(f"Label file not found for image {img_file}, skipping.")
                continue

            # Agendamento assíncrono do processamento
            futures.append(
                executor.submit(
                    process_image, img_path, label_path, output_images_dir, output_labels_dir, aug_list, params["copies"], params
                )
            )

        # Acompanhamento do progresso em paralelo
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                # Se ocorrer alguma exceção no processamento, ela será capturada aqui
                future.result()
            except Exception as e:
                print(f"Error processing an image: {e}")

    print("Data augmentation completed successfully!")


if __name__ == "__main__":
    main()
