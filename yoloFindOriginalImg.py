"""
This script finds the original images corresponding to a set of resized images.
It uses a pre-trained VGG16 model from PyTorch (with CUDA support) to compute normalized feature vectors entirely on the GPU.
It resolves Unicode file path issues by reading image files via np.fromfile and cv2.imdecode.
For each resized image, the script compares its feature vector against all originals in the dataset,
computing the cosine similarity for each pair. Then it selects the best match – that is, the original
image with the highest similarity score (the maximum threshold) – provided that score exceeds a user-defined threshold.
The matched original image is copied to a subfolder (named 'matched') in the resized images directory,
renamed with the resized image's filename.
Note: The CSV output functionality has been removed.
"""

import os
import sys
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# =============================================================================
# USER-ADJUSTABLE VARIABLES
# =============================================================================
IMAGE_SIZE = (256, 256)  # Resolution for processing (width, height)
RESIZED_BATCH_SIZE = 32  # Number of resized images processed simultaneously on GPU
ORIGINAL_BATCH_SIZE = 32  # Number of original images compared in one mini-batch on GPU
RESIZED_LOADING_BATCH = 64  # Number of resized images loaded from disk at once
ORIGINAL_LOADING_BATCH = 64  # Number of original images loaded from disk at once
LOAD_RESIZED_TO_GPU = True  # If True, preloads resized images to GPU for feature extraction
LOAD_ORIGINAL_TO_GPU = True  # If True, preloads original images to GPU for feature extraction
MATCH_THRESHOLD = 0.90  # Minimum cosine similarity to accept a match
# =============================================================================


def imread_unicode(file_path, flags=cv2.IMREAD_COLOR):
    """
    Read an image from disk using its Unicode path.
    Uses np.fromfile to read raw bytes and cv2.imdecode to decode the image.
    """
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def get_image_paths(folder):
    """Retrieve image file paths from a folder."""
    allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(allowed_ext)])


def load_images_batch(paths, transform, device, load_to_gpu):
    """
    Load a batch of images using imread_unicode, apply transformation, and stack them into a tensor.
    """
    images = []
    valid_paths = []
    for path in paths:
        image = imread_unicode(path)
        if image is None:
            print(f"Error loading {path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        pil_image = Image.fromarray(image)
        try:
            tensor = transform(pil_image)
        except Exception as e:
            print(f"Error transforming {path}: {e}")
            continue
        images.append(tensor)
        valid_paths.append(path)
    if not images:
        return None, []
    batch_tensor = torch.stack(images)  # shape: (batch_size, C, H, W)
    if load_to_gpu:
        batch_tensor = batch_tensor.to(device)
    return batch_tensor, valid_paths


def compute_features_for_paths(image_paths, model, transform, device, loading_batch, processing_batch, load_to_gpu):
    """
    Compute normalized feature vectors for a list of image paths.
    Images are loaded in batches and processed in mini-batches through the model.
    Returns a dictionary mapping image path to its feature vector.
    """
    features_dict = {}
    total = len(image_paths)
    for i in range(0, total, loading_batch):
        batch_paths = image_paths[i : i + loading_batch]
        batch_tensor, valid_paths = load_images_batch(batch_paths, transform, device, load_to_gpu)
        if batch_tensor is None:
            continue
        num_images = batch_tensor.size(0)
        for j in range(0, num_images, processing_batch):
            mini_batch = batch_tensor[j : j + processing_batch]
            mini_paths = valid_paths[j : j + processing_batch]
            with torch.no_grad():
                feats = model(mini_batch)
            feats = feats.view(feats.size(0), -1)
            norms = feats.norm(dim=1, keepdim=True)
            feats = feats / (norms + 1e-10)  # Normalize feature vectors
            if load_to_gpu:
                for idx, path in enumerate(mini_paths):
                    features_dict[path] = feats[idx]
            else:
                feats_np = feats.cpu().numpy()
                for idx, path in enumerate(mini_paths):
                    features_dict[path] = feats_np[idx]
        print(f"Processed {min(i + loading_batch, total)}/{total} images for feature extraction.")
    return features_dict


def match_resized_batch(resized_feats, original_features_matrix, original_paths, original_batch_size, device, originals_on_gpu):
    """
    For a batch of resized features, compute cosine similarity with all original features.
    This function iterates over the original images in mini-batches, ensuring that for each resized image,
    the maximum similarity (i.e. the best match) is selected across the entire original dataset.
    Returns best match indices and similarity scores for each resized image.
    """
    num_resized = resized_feats.size(0)
    if originals_on_gpu:
        best_indices = torch.empty(num_resized, dtype=torch.long, device=device)
        best_scores = torch.empty(num_resized, device=device)
        N_orig = original_features_matrix.size(0)
        best_scores.fill_(-1.0)
        # Iterate over original features in mini-batches
        for i in range(0, N_orig, original_batch_size):
            orig_batch = original_features_matrix[i : i + original_batch_size]  # shape: (B, feat_dim)
            sims = torch.matmul(orig_batch, resized_feats.t())  # shape: (B, num_resized)
            batch_best_scores, batch_best_indices = sims.max(dim=0)
            update_mask = batch_best_scores > best_scores
            best_scores[update_mask] = batch_best_scores[update_mask]
            temp_indices = batch_best_indices + i  # Adjust index relative to full original dataset
            best_indices[update_mask] = temp_indices[update_mask]
        return best_indices, best_scores
    else:
        # If originals are on CPU, compute dot product with numpy arrays
        resized_np = resized_feats.cpu().numpy()
        sims = np.dot(original_features_matrix, resized_np.T)
        best_indices = np.argmax(sims, axis=0)
        best_scores = np.max(sims, axis=0)
        return best_indices, best_scores


def process_resized_images(
    resized_paths,
    model,
    transform,
    device,
    loading_batch,
    processing_batch,
    load_resized_to_gpu,
    original_features_matrix,
    original_paths,
    original_batch_size,
    originals_on_gpu,
    output_folder,
    match_threshold,
):
    """
    Process resized images in batches, match each to the best original image (i.e. the one with the highest similarity score
    among all comparisons), and copy the matched original image to the output folder (renamed as the resized image).
    Only if the maximum similarity exceeds the defined threshold is a match accepted.
    Returns mapping results.
    """
    mapping_results = []
    total = len(resized_paths)
    for i in range(0, total, loading_batch):
        batch_paths = resized_paths[i : i + loading_batch]
        batch_tensor, valid_paths = load_images_batch(batch_paths, transform, device, load_resized_to_gpu)
        if batch_tensor is None:
            continue
        num_images = batch_tensor.size(0)
        for j in range(0, num_images, processing_batch):
            mini_batch = batch_tensor[j : j + processing_batch]
            mini_paths = valid_paths[j : j + processing_batch]
            with torch.no_grad():
                feats = model(mini_batch)
            feats = feats.view(feats.size(0), -1)
            norms = feats.norm(dim=1, keepdim=True)
            feats = feats / (norms + 1e-10)
            if not load_resized_to_gpu and originals_on_gpu:
                feats = feats.to(device)
            # For each resized image in the mini-batch, find the best (maximum similarity) match across all originals
            best_idx, best_sim = match_resized_batch(
                feats, original_features_matrix, original_paths, original_batch_size, device, originals_on_gpu
            )
            if originals_on_gpu:
                best_idx = best_idx.cpu().numpy()
                best_sim = best_sim.cpu().numpy()
            else:
                best_idx = np.array(best_idx)
                best_sim = np.array(best_sim)
            for idx, resized_path in enumerate(mini_paths):
                similarity = float(best_sim[idx])
                if similarity >= match_threshold:
                    matched_original = original_paths[best_idx[idx]]
                    output_filename = os.path.basename(resized_path)
                    output_path = os.path.join(output_folder, output_filename)
                    try:
                        shutil.copy(matched_original, output_path)
                    except Exception as e:
                        print(f"Error copying {matched_original} to {output_path}: {e}")
                    mapping_results.append((resized_path, matched_original, similarity))
                else:
                    print(f"Match for {resized_path} rejected (similarity: {similarity:.4f} below threshold).")
        print(f"Matched {min(i + loading_batch, total)}/{total} resized images.")
    return mapping_results


def main():
    # Initialize Tkinter and hide the root window
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # Select folder containing resized images (with YOLO bounding boxes)
    resized_folder = filedialog.askdirectory(title="Select folder with resized images")
    if not resized_folder:
        print("No folder selected for resized images. Exiting.")
        sys.exit(1)

    # Select folder containing the original images dataset
    original_folder = filedialog.askdirectory(title="Select folder with original images dataset")
    if not original_folder:
        print("No folder selected for original images. Exiting.")
        sys.exit(1)

    # Create subfolder to save matched original images
    matched_folder = os.path.join(resized_folder, "matched")
    os.makedirs(matched_folder, exist_ok=True)

    # Set device to GPU (CUDA) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained VGG16 model from PyTorch and create a feature extractor
    print("Loading VGG16 model from PyTorch...")
    vgg16 = models.vgg16(pretrained=True)
    feature_extractor = nn.Sequential(vgg16.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()).to(device)
    feature_extractor.eval()
    print("Model loaded.")

    # Define image transformation using user-adjustable IMAGE_SIZE
    transform = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Process original images: compute features in batches on GPU
    print("Processing original images for feature extraction...")
    original_paths = get_image_paths(original_folder)
    if not original_paths:
        print("No original images found. Exiting.")
        sys.exit(1)
    original_features_dict = compute_features_for_paths(
        original_paths, feature_extractor, transform, device, ORIGINAL_LOADING_BATCH, ORIGINAL_BATCH_SIZE, LOAD_ORIGINAL_TO_GPU
    )
    sorted_orig_paths = sorted(original_features_dict.keys())
    if LOAD_ORIGINAL_TO_GPU:
        orig_feats = torch.stack([original_features_dict[p] for p in sorted_orig_paths]).to(device)
    else:
        orig_feats = np.stack([original_features_dict[p] for p in sorted_orig_paths])
    print(f"Computed features for {len(sorted_orig_paths)} original images.")

    # Process resized images: compute features and match with originals on GPU
    print("Processing resized images for feature extraction and matching...")
    resized_paths = get_image_paths(resized_folder)
    if not resized_paths:
        print("No resized images found. Exiting.")
        sys.exit(1)
    mapping_results = process_resized_images(
        resized_paths,
        feature_extractor,
        transform,
        device,
        RESIZED_LOADING_BATCH,
        RESIZED_BATCH_SIZE,
        LOAD_RESIZED_TO_GPU,
        orig_feats,
        sorted_orig_paths,
        ORIGINAL_BATCH_SIZE,
        LOAD_ORIGINAL_TO_GPU,
        matched_folder,
        MATCH_THRESHOLD,
    )

    # Print a summary of matches
    print("Matching complete.")
    print(f"Total resized images processed: {len(resized_paths)}")
    print(f"Total accepted matches: {len(mapping_results)}")
    for res_path, orig_path, sim in mapping_results:
        print(f"Resized: {res_path}  ->  Original: {orig_path}  (Similarity: {sim:.4f})")


if __name__ == "__main__":
    main()
