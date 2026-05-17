"""
BBox Extractor v4: PaddleOCR + 単一スケール + 固定閾値50%
PaddleOCR で単一スケールの行単位抽出し、水平重なり50%でマージする。
"""
import contextlib
import os
import sys
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans

# Paddle/PaddleOCR の警告とログを抑制
warnings.filterwarnings("ignore", message="No ccache found")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# PaddleOCR インスタンスをグローバルに保持（初回ロード時間短縮）
_ocr = None


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Paddle/PaddleX の情報ログを一時的に抑制する"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _get_ocr():
    """PaddleOCR インスタンスを取得（遅延初期化）"""
    global _ocr
    if _ocr is None:
        with _suppress_stdout_stderr():
            _ocr = PaddleOCR(lang="en")
    return _ocr


def extract_text_bboxes_v4(image: Image.Image, **kwargs) -> List[Dict[str, Any]]:
    """
    PaddleOCR を使って画像内の全テキストブロックを抽出する。
    単一スケール OCR → カラム分離 → 固定閾値で行マージ
    """
    # 1. 単一スケール OCR
    ocr = _get_ocr()
    img_array = np.array(image.convert("RGB"))
    with _suppress_stdout_stderr():
        results = ocr.predict(img_array)

    if not results:
        return []

    img_w, img_h = image.size

    line_items = []
    for page_result in results:
        rec_texts = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        rec_boxes = page_result.get("rec_boxes", [])

        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            x1 = float(box[0]) / img_w
            y1 = float(box[1]) / img_h
            x2 = float(box[2]) / img_w
            y2 = float(box[3]) / img_h

            line_items.append(
                {
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                }
            )

    # 2. カラム分離
    left_items, right_items = _split_columns(line_items)

    # 3. 各カラムで固定閾値（1.5）でマージ（水平重なり50%）
    all_blocks = []

    for column_items in [left_items, right_items]:
        if not column_items:
            continue

        blocks = _merge_lines_in_column(column_items, threshold=1.5)
        all_blocks.extend(blocks)

    # Y座標でソートして返す
    all_blocks.sort(key=lambda x: x["bbox"][1])

    return all_blocks


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


def _merge_lines_in_column(
    line_items: List[Dict[str, Any]], threshold: float = 1.5
) -> List[Dict[str, Any]]:
    """1つのカラム内で、行をマージしてブロックに再構成する（水平重なり50%）。"""
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

        # 水平重なりチェック（50%以上）
        overlap_left = max(prev_bbox[0], curr_bbox[0])
        overlap_right = min(prev_bbox[2], curr_bbox[2])
        overlap_width = max(0, overlap_right - overlap_left)
        min_width = min(prev_bbox[2] - prev_bbox[0], curr_bbox[2] - curr_bbox[0])
        overlap_ratio = overlap_width / min_width if min_width > 0 else 0

        if overlap_ratio >= 0.5:
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
