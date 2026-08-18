"""
Batch-convert Supervisely bitmap annotations into cached 4x4
grid classification labels.

For each image:
  1. Decode all object masks
  2. Paint them onto one full-size label canvas (by class, using priority
     order to resolve overlaps)
  3. Downsample that canvas into a 4x4 grid using presence-based labeling
     (any pixels of a class in a cell -> that cell gets that class)
  4. Save the resulting 4x4 label array as a .npy file, and log a CSV
     mapping image path -> label path

Usage:
    python mask.py <img_dir> <ann_dir> <output_dir>
"""

import sys
import json
import base64
import zlib
import io
import csv
import os

import numpy as np
from PIL import Image

GRID_SIZE = 4

# Class priority: LAST in this list wins when a cell has multiple classes.
CLASS_PRIORITY = ["road", "cracks", "pothole"]

BACKGROUND_LABEL = "background"


def decode_supervisely_bitmap(bitmap_data_b64: str) -> np.ndarray:
    """
    Supervisely bitmap masks are stored as: base64(zlib(PNG bytes))
    Returns a boolean numpy array (the mask, cropped to its bounding box --
    NOT yet placed on the full image canvas).
    """
    compressed = base64.b64decode(bitmap_data_b64)
    png_bytes = zlib.decompress(compressed)
    mask_img = Image.open(io.BytesIO(png_bytes))
    mask_arr = np.array(mask_img)

    if mask_arr.ndim == 3:
        if mask_arr.shape[2] == 4:
            mask = mask_arr[:, :, 3] > 0
        else:
            mask = mask_arr.sum(axis=2) > 0
    else:
        mask = mask_arr > 0

    return mask


def place_mask_on_canvas(mask: np.ndarray, origin: list, canvas_h: int, canvas_w: int) -> np.ndarray:
    """
    Places a cropped mask onto a full-size canvas at the given [x, y] origin
    (Supervisely origin is [x, y], i.e. [col, row]).
    """
    canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
    x_off, y_off = origin[0], origin[1]
    mh, mw = mask.shape

    y_end = min(y_off + mh, canvas_h)
    x_end = min(x_off + mw, canvas_w)
    mask_h_clip = y_end - y_off
    mask_w_clip = x_end - x_off

    canvas[y_off:y_end, x_off:x_end] = mask[:mask_h_clip, :mask_w_clip]
    return canvas


def build_label_canvas(ann: dict, canvas_h: int, canvas_w: int) -> np.ndarray:
    """
    Returns an (H, W) integer array where each pixel holds the index into
    CLASS_PRIORITY of the class present at that pixel (or -1 for background).
    Higher-priority classes overwrite lower-priority ones where they overlap.
    """
    label_canvas = np.full((canvas_h, canvas_w), -1, dtype=np.int8)

    for obj in ann["objects"]:
        if obj["geometryType"] != "bitmap":
            continue
        class_title = obj["classTitle"]
        if class_title not in CLASS_PRIORITY:
            print(f"  WARNING: unknown class {class_title!r} not in CLASS_PRIORITY, skipping")
            continue

        class_idx = CLASS_PRIORITY.index(class_title)
        mask = decode_supervisely_bitmap(obj["bitmap"]["data"])
        full_mask = place_mask_on_canvas(mask, obj["bitmap"]["origin"], canvas_h, canvas_w)

        # Only overwrite where this class has HIGHER (or equal) priority
        # than whatever's already painted there
        overwrite = full_mask & (class_idx >= label_canvas)
        label_canvas[overwrite] = class_idx

    return label_canvas


def downsample_to_grid(label_canvas: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Presence-based downsampling: for each grid cell, pick the HIGHEST
    priority class present anywhere in that cell (-1 = background).
    """
    h, w = label_canvas.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    grid_labels = np.full((grid_size, grid_size), -1, dtype=np.int8)

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h if i < grid_size - 1 else h
            x0, x1 = j * cell_w, (j + 1) * cell_w if j < grid_size - 1 else w
            cell = label_canvas[y0:y1, x0:x1]

            present_classes = cell[cell >= 0]
            if len(present_classes) == 0:
                grid_labels[i, j] = -1
            else:
                grid_labels[i, j] = present_classes.max()

    return grid_labels


def process_dataset(image_dir: str, ann_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "grid_labels")
    os.makedirs(labels_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    print(f"Found {len(image_files)} images in {image_dir}")

    n_success, n_failed = 0, 0

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "grid_label_path"])

        for idx, img_filename in enumerate(image_files):
            img_path = os.path.join(image_dir, img_filename)
            ann_path = os.path.join(ann_dir, img_filename + ".json")

            if not os.path.exists(ann_path):
                print(f"  [{idx+1}/{len(image_files)}] SKIP {img_filename}: no matching annotation")
                n_failed += 1
                continue

            try:
                with open(ann_path, "r") as f:
                    ann = json.load(f)

                canvas_h = ann["size"]["height"]
                canvas_w = ann["size"]["width"]

                label_canvas = build_label_canvas(ann, canvas_h, canvas_w)
                grid_labels = downsample_to_grid(label_canvas, GRID_SIZE)

                label_out_path = os.path.join(labels_dir, img_filename + ".npy")
                np.save(label_out_path, grid_labels)

                writer.writerow([img_path, label_out_path])
                n_success += 1

                if (idx + 1) % 100 == 0:
                    print(f"  [{idx+1}/{len(image_files)}] processed...")

            except Exception as e:
                print(f"  [{idx+1}/{len(image_files)}] FAILED {img_filename}: {type(e).__name__}: {e!r}")
                n_failed += 1

    print(f"\nDone. {n_success} succeeded, {n_failed} failed.")
    print(f"Manifest written to: {manifest_path}")
    print(f"Grid labels written to: {labels_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mask.py <img_dir> <ann_dir> <output_dir>")
        sys.exit(1)
    process_dataset(sys.argv[1], sys.argv[2], sys.argv[3])