"""
BBox Extractor v2: PaddleOCR + 固定閾値マージ（水平30%・80%達成版）
PaddleOCR で行単位抽出し、固定閾値でメニュー項目単位にマージする。
"""
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from .common_ocr import get_ocr, suppress_stdout_stderr


def extract_text_bboxes_v2(image: Image.Image, **kwargs) -> List[Dict[str, Any]]:
    """
    PaddleOCR を使って画像内の全テキストブロックを抽出する。
    行単位の結果を固定閾値でブロック単位に再構成する。
    """
    ocr = get_ocr()

    # PIL Image を numpy.ndarray に変換
    img_array = np.array(image.convert("RGB"))
    with suppress_stdout_stderr():
        results = ocr.predict(img_array)

    if not results:
        return []

    # 画像サイズを取得
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

    # 行単位結果をブロック単位にマージ（固定閾値 1.5）
    block_items = _merge_ocr_lines(line_items, threshold=1.5, overlap_threshold=0.3)

    return block_items


def _merge_ocr_lines(
    ocr_items: List[Dict[str, Any]],
    threshold: float = 1.5,
    overlap_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    PaddleOCR の行単位結果を、隣接行をマージしてブロック単位に再構成する。
    """
    if not ocr_items:
        return []

    # Y座標でソート
    sorted_items = sorted(ocr_items, key=lambda x: x["bbox"][1])

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

        # マージ判定
        prev_bbox = current_block["bbox"]
        curr_bbox = item["bbox"]

        # 水平重なりチェック（overlap_threshold 以上）
        overlap_left = max(prev_bbox[0], curr_bbox[0])
        overlap_right = min(prev_bbox[2], curr_bbox[2])
        overlap_width = max(0, overlap_right - overlap_left)
        min_width = min(prev_bbox[2] - prev_bbox[0], curr_bbox[2] - curr_bbox[0])
        overlap_ratio = overlap_width / min_width if min_width > 0 else 0

        if overlap_ratio >= overlap_threshold:
            # 垂直距離チェック
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
                # マージ実行
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

        # マージしない: ブロックを確定
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

    # 内部フィールドを削除
    for block in blocks:
        del block["count"]
        del block["last_line_height"]

    return blocks
