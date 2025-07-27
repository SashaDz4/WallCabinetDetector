import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

TEMPLATE_DIR = "templates/WallCabinet"
RESULT_IMAGE = "results/marked_image.png"
CSV_OUTPUT = "results/detections.csv"
THRESHOLD = 0.4 # Adjust this threshold as needed

def preprocess_sketch(img):
    # 1. Адаптивне порогування — для підсилення ліній
    bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

    # 2. Морфологічне розширення — щоб зробити лінії товстішими
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(bin_img, kernel, iterations=10)

    return dilated

def load_templates(template_dir):
    templates = []
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            templates.append((filename, preprocess_sketch(img)))
    return templates

def perform_template_matching(image_path, templates):
    img_rgb = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    preprocess_img = preprocess_sketch(img_gray)
    detections = []

    for template_name, template in templates:
        matched_points, wh = [], []
        for i in range(1, 90):
            c = i / 100.0
            # resize the template
            resized_template = cv2.resize(template, (0, 0), fx=c, fy=c)
            h, w = resized_template.shape
            result = cv2.matchTemplate(preprocess_img, resized_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= THRESHOLD)

            for pt in zip(*loc[::-1]):
                x, y = pt
                matched_points.append((x, y))
                wh.append((w, h))
        if matched_points:
            matched_points_np = np.array(matched_points)
            clustering = DBSCAN(eps=40, min_samples=3).fit(matched_points_np)
            labels = clustering.labels_

            for label in set(labels):
                if label == -1:
                    continue  # шум
                cluster_points = matched_points_np[labels == label]
                x, y = np.mean(cluster_points, axis=0).astype(int)
                w, h = np.mean(np.array(wh), axis=0).astype(int)
                # Draw rectangle around detected cluster
                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detections.append({
                    "template": template_name,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
                print(f"Marked {template_name} at ({x}, {y}) with size ({w}, {h})")

    return img_rgb, detections

def save_results(marked_image, detections):
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(RESULT_IMAGE, marked_image)
    df = pd.DataFrame(detections)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Results saved to:\n- {RESULT_IMAGE}\n- {CSV_OUTPUT}")

def main(image_path):
    templates = load_templates(TEMPLATE_DIR)
    marked_image, detections = perform_template_matching(image_path, templates)
    save_results(marked_image, detections)

if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Wall Cabinet Detector")
    # parser.add_argument("image_path", help="Path to architectural PNG file")
    # args = parser.parse_args()
    #
    # main(args.image_path)
    main("D:\WallCabinetDetector\TemplateTask.png")
