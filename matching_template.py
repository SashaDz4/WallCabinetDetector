import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
from sklearn.cluster import DBSCAN

# --- Config ---
TEMPLATE_DIR = "templates/WallCabinet"
RESULT_IMAGE_PATH = "results/marked_image.png"
CSV_OUTPUT_PATH = "results/detections.csv"
THRESHOLD = 0.48
CLUSTER_EPS = 40
CLUSTER_MIN_SAMPLES = 3
MIN_SIDE_LENGTH = 50

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Utility Functions ---
def wh_coefficient_check(wh1: Tuple[int, int], wh2: Tuple[int, int], threshold: float = 0.5) -> bool:
    w1, h1 = wh1
    w2, h2 = wh2
    return (
        abs(w1 - w2) / max(w1, w2) < threshold and
        abs(h1 - h2) / max(h1, h2) < threshold
    )


def verify_matching(cropped_img: np.ndarray, template: np.ndarray, threshold: float = 0.3) -> bool:
    if cropped_img.shape[0] < 1 or cropped_img.shape[1] < 1:
        return False
    if not wh_coefficient_check(cropped_img.shape[:2], template.shape[:2]):
        return False
    result = cv2.matchTemplate(cropped_img, template, cv2.TM_CCOEFF_NORMED)
    return np.any(result >= threshold)


def preprocess_sketch(img: np.ndarray) -> np.ndarray:
    bin_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    return cv2.dilate(bin_img, np.ones((2, 2), np.uint8), iterations=5)


# --- Core Logic ---
def load_templates(template_dir: str) -> List[Tuple[str, np.ndarray]]:
    templates = []
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((filename, img))
            else:
                logging.warning(f"Unable to load image: {filename}")
    return templates


def match_template_on_scales(processed_img: np.ndarray, template_name: str, template: np.ndarray) \
        -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    matched_points, sizes = [], []
    for scale in np.arange(0.2, 1., 0.02):
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        h, w = resized_template.shape

        if h < MIN_SIDE_LENGTH or w < MIN_SIDE_LENGTH:
            logging.debug(f"Skipping {template_name} at scale {scale:.2f} due to small size ({w}x{h})")
            continue

        result = cv2.matchTemplate(processed_img, preprocess_sketch(resized_template), cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= THRESHOLD)

        for x, y in zip(*loc[::-1]):
            matched_points.append((x, y))
            sizes.append((w, h))

    return matched_points, sizes


def cluster_matches(processed_img: np.ndarray, matched_points: List[Tuple[int, int]], sizes: List[Tuple[int, int]],
                    original_img: np.ndarray, template: np.ndarray, template_name: str, marked_img: np.ndarray) \
        -> Tuple[List[Dict], np.ndarray]:
    detections = []

    if not matched_points:
        return detections, processed_img

    matched_points_np = np.array(matched_points)
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(matched_points_np).labels_

    logging.info(f"{template_name}: {len(matched_points)} matches, "
                 f"{len(set(labels)) - (1 if -1 in labels else 0)} clusters")

    for label in set(labels):
        if label == -1:
            continue  # noise

        cluster_points = matched_points_np[labels == label]
        x, y = np.mean(cluster_points, axis=0).astype(int)
        w, h = np.mean(np.array(sizes), axis=0).astype(int)

        cropped = original_img[y:y + h, x:x + w]
        if verify_matching(cropped, template):
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            detections.append({
                "template": template_name,
                "x": x, "y": y, "w": w, "h": h
            })
            # Mark the area in the processed image
            processed_img[y:y + h, x:x + w] = 255
            logging.info(f"Match confirmed for {template_name} at ({x}, {y}) size ({w}x{h})")

    return detections, processed_img


def perform_template_matching(image_path: str, templates: List[Tuple[str, np.ndarray]]) \
        -> Tuple[np.ndarray, List[Dict]]:
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    processed_img = preprocess_sketch(grayscale_img)

    all_detections = []

    for template_name, template in templates:
        points, sizes = match_template_on_scales(processed_img, template_name, template)
        detections, processed_img = cluster_matches(processed_img, points, sizes, grayscale_img, template,
                                                    template_name, original_img)
        all_detections.extend(detections)

    return original_img, all_detections


def save_results(marked_img: np.ndarray, detections: List[Dict]) -> None:
    os.makedirs(os.path.dirname(RESULT_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(RESULT_IMAGE_PATH, marked_img)
    pd.DataFrame(detections).to_csv(CSV_OUTPUT_PATH, index=False)
    logging.info(f"Results saved to:\n- {RESULT_IMAGE_PATH}\n- {CSV_OUTPUT_PATH}")


# --- Entry Point ---
def main(image_path: str) -> None:
    logging.info("Starting template matching...")
    templates = load_templates(TEMPLATE_DIR)
    marked_image, detections = perform_template_matching(image_path, templates)
    save_results(marked_image, detections)


if __name__ == "__main__":
    main("TemplateTask.png")
