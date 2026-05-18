"""
BBox Extractor v3: PaddleOCR + マルチスケール + K-means + GPT-4o 検証
複数スケールで OCR 実行し、K-means クラスタリングで動的閾値を計算する。
"""
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from .common_ocr import get_ocr, suppress_stdout_stderr


def extract_text_bboxes_v3(image: Image.Image, **kwargs) -> List[Dict[str, Any]]:
    """
    PaddleOCR を使って画像内の全テキストブロックを抽出する。
    マルチスケール OCR → カラム分離 → K-means 動的閾値 → 行マージ
    """
    # 1. マルチスケール OCR
    line_items = _multiscale_ocr(image)

    if not line_items:
        return []

    # 2. カラム分離
    left_items, right_items = _split_columns(line_items)

    # 3. 各カラムで動的閾値を計算しマージ
    all_blocks = []

    for column_items in [left_items, right_items]:
        if not column_items:
            continue

        # 動的閾値計算
        threshold = _compute_dynamic_threshold(column_items)

        # 行マージ
        blocks = _merge_lines_in_column(column_items, threshold)
        all_blocks.extend(blocks)

    # Y座標でソートして返す
    all_blocks.sort(key=lambda x: x["bbox"][1])

    return all_blocks


def _multiscale_ocr(image: Image.Image, scales: List[float] = None) -> List[Dict[str, Any]]:
    """複数スケールで OCR を実行し、結果を統合する。"""
    if scales is None:
        scales = [1.0, 1.25, 1.5]

    ocr = get_ocr()

    all_items = []
    orig_w, orig_h = image.size

    for scale in scales:
        if scale == 1.0:
            resized = image
        else:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized = image.resize((new_w, new_h), Image.LANCZOS)

        img_array = np.array(resized.convert("RGB"))
        with suppress_stdout_stderr():
            results = ocr.predict(img_array)

        if results:
            for page_result in results:
                rec_texts = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])
                rec_boxes = page_result.get("rec_boxes", [])

                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    x1 = float(box[0]) / (orig_w * scale)
                    y1 = float(box[1]) / (orig_h * scale)
                    x2 = float(box[2]) / (orig_w * scale)
                    y2 = float(box[3]) / (orig_h * scale)

                    all_items.append(
                        {
                            "text": text,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(score),
                        }
                    )

    # 重複排除
    unique_items = []
    for item in all_items:
        is_duplicate = False
        for existing in unique_items:
            if item["text"].lower() == existing["text"].lower():
                dist = sum((a - b) ** 2 for a, b in zip(item["bbox"], existing["bbox"]))
                if dist < 0.001:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_items.append(item)

    return unique_items


def _split_columns(line_items: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """OCR 行結果を X 座標でクラスタリングし、左右カラムに分離する。"""
    if not line_items:
        return [], []

    centers = []
    for item in line_items:
        bbox = item["bbox"]
        center_x = (bbox[0] + bbox[2]) / 2
        centers.append(center_x)

    centers = np.array(centers).reshape(-1, 1)

    if len(centers) < 3:
        return line_items, []

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(centers)

    cluster_centers = kmeans.cluster_centers_.flatten()
    distance = abs(cluster_centers[0] - cluster_centers[1])

    if distance < 0.2:
        return line_items, []

    left_items = []
    right_items = []

    for item, label in zip(line_items, labels):
        if cluster_centers[label] < cluster_centers[1 - label]:
            left_items.append(item)
        else:
            right_items.append(item)

    left_items.sort(key=lambda x: x["bbox"][1])
    right_items.sort(key=lambda x: x["bbox"][1])

    return left_items, right_items


def _compute_dynamic_threshold(line_items: List[Dict[str, Any]]) -> float:
    """行間隔を K-means でクラスタリングし、動的なマージ閾値を計算する。"""
    if len(line_items) < 3:
        return 1.5

    gaps = []
    for i in range(len(line_items) - 1):
        curr_y2 = line_items[i]["bbox"][3]
        next_y1 = line_items[i + 1]["bbox"][1]
        gap = next_y1 - curr_y2
        if gap > 0:
            line_height = line_items[i]["bbox"][3] - line_items[i]["bbox"][1]
            if line_height > 0:
                normalized_gap = gap / line_height
                gaps.append(normalized_gap)

    if len(gaps) < 3:
        return 1.5

    gaps = np.array(gaps).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(gaps)

    centers = kmeans.cluster_centers_.flatten()
    dense_cluster_center = min(centers)

    threshold = dense_cluster_center + 0.5
    return max(1.2, min(threshold, 3.0))


def _merge_lines_in_column(
    line_items: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """1つのカラム内で、行をマージしてブロックに再構成する。"""
    if not line_items:
        return []

    sorted_items = sorted(line_items, key=lambda x: x["bbox"][1])

    blocks = []
    current_block = None

    for item in sorted_items:
        item_height = item["bbox"][3] - item["bbox"][1]

        if current_block is None:
            current_block = {
                "text": item["text"],
                "bbox": list(item["bbox"]),
                "confidence": item["confidence"],
                "count": 1,
                "last_line_height": item_height,
            }
            continue

        prev_bbox = current_block["bbox"]
        curr_bbox = item["bbox"]

        overlap_left = max(prev_bbox[0], curr_bbox[0])
        overlap_right = min(prev_bbox[2], curr_bbox[2])
        overlap_width = max(0, overlap_right - overlap_left)
        min_width = min(prev_bbox[2] - prev_bbox[0], curr_bbox[2] - curr_bbox[0])
        overlap_ratio = overlap_width / min_width if min_width > 0 else 0

        if overlap_ratio >= 0.3:
            last_line_height = current_block.get(
                "last_line_height", prev_bbox[3] - prev_bbox[1]
            )
            vertical_gap = curr_bbox[1] - prev_bbox[3]

            should_merge = (
                vertical_gap <= last_line_height * threshold
                if vertical_gap >= 0
                else True
            )

            if should_merge:
                current_block["bbox"] = [
                    min(prev_bbox[0], curr_bbox[0]),
                    min(prev_bbox[1], curr_bbox[1]),
                    max(prev_bbox[2], curr_bbox[2]),
                    max(prev_bbox[3], curr_bbox[3]),
                ]
                current_block["text"] += " " + item["text"]
                current_block["confidence"] = (
                    current_block["confidence"] * current_block["count"]
                    + item["confidence"]
                ) / (current_block["count"] + 1)
                current_block["count"] += 1
                current_block["last_line_height"] = item_height
                continue

        blocks.append(current_block)
        current_block = {
            "text": item["text"],
            "bbox": list(item["bbox"]),
            "confidence": item["confidence"],
            "count": 1,
            "last_line_height": item_height,
        }

    if current_block is not None:
        blocks.append(current_block)

    for block in blocks:
        del block["count"]
        del block["last_line_height"]

    return blocks
