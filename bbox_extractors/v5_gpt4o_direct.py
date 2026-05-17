"""
BBox Extractor v5: Hybrid GPT-4o + PaddleOCR Direct Matching

【設計方針】
GPT-4o Vision API は内部的に画像をリサイズ・タイル分割するため、
モデルが返す「座標」は元画像のピクセルと一致しない（技術的制約）。

したがって、以下のハイブリッド方式を採用する：
1. PaddleOCR で画像内の全テキストブロックを正確なピクセルbboxで抽出
2. GPT-4o に画像とOCRブロックリストを渡し、各メニュー項目に対応するブロックを判断させる
3. Python 側で判断結果に基づき、bboxを統合・紐付け

これにより「PaddleOCRの座標精度」と「GPT-4oの意味的理解」の両立を実現する。
"""
import base64
import io
import json
import os
import warnings
from io import BytesIO
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from openai import OpenAI
from paddleocr import PaddleOCR

warnings.filterwarnings("ignore", message="No ccache found")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang="en")
    return _ocr


def _encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_ocr_blocks(image: Image.Image) -> List[Dict[str, Any]]:
    """PaddleOCR でテキストブロックを抽出し、正規化bbox付きで返す"""
    ocr = _get_ocr()
    img_array = np.array(image.convert("RGB"))
    results = ocr.predict(img_array)

    if not results:
        return []

    img_w, img_h = image.size
    blocks = []
    block_id = 0

    for page_result in results:
        rec_texts = page_result.get("rec_texts", [])
        rec_scores = page_result.get("rec_scores", [])
        rec_boxes = page_result.get("rec_boxes", [])

        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])

            blocks.append({
                "id": block_id,
                "text": text,
                "bbox_pixels": [x1, y1, x2, y2],
                "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
                "confidence": float(score),
            })
            block_id += 1

    return blocks


def _merge_blocks(blocks: List[Dict], indices: List[int]) -> List[float]:
    """指定されたブロック群のbboxを統合して1つの大きなbboxにする"""
    selected = [blocks[i] for i in indices if 0 <= i < len(blocks)]
    if not selected:
        return [0.0, 0.0, 1.0, 1.0]

    x1 = min(b["bbox"][0] for b in selected)
    y1 = min(b["bbox"][1] for b in selected)
    x2 = max(b["bbox"][2] for b in selected)
    y2 = max(b["bbox"][3] for b in selected)

    return [x1, y1, x2, y2]


def extract_menu_items_v5(image: Image.Image, client: OpenAI) -> List[Dict[str, Any]]:
    """
    Hybrid方式でメニュー項目を抽出。
    戻り値: [{"name":..., "price":..., "description":..., "bbox":..., "bbox_pixels":..., "matched_blocks":...}, ...]
    """
    img_w, img_h = image.size

    # Step 1: PaddleOCR で全テキストブロックを抽出
    ocr_blocks = _extract_ocr_blocks(image)
    if not ocr_blocks:
        return []

    # Step 2: OCRブロックリストをテキスト化
    ocr_text_lines = []
    for b in ocr_blocks:
        ocr_text_lines.append(f"Block {b['id']}: \"{b['text']}\" at pixels [{int(b['bbox_pixels'][0])}, {int(b['bbox_pixels'][1])}, {int(b['bbox_pixels'][2])}, {int(b['bbox_pixels'][3])}]")
    ocr_summary = "\n".join(ocr_text_lines)

    # Step 3: GPT-4o に画像とOCRリストを渡してマッチング判定
    b64_image = _encode_image_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{b64_image}"

    system_prompt = (
        "You are an expert at reading restaurant menus. "
        "I will show you a menu image and a list of OCR-detected text blocks with their IDs and positions. "
        "Your task is to identify each menu item and tell me which OCR block IDs correspond to it.\n\n"
        "Return ONLY a JSON array of objects with these keys:\n"
        "  - name: dish name (string)\n"
        "  - price: price string, or empty if not visible\n"
        "  - description: brief description, or empty if not visible\n"
        "  - block_indices: array of integers (OCR block IDs that form this menu item)\n\n"
        "CRITICAL RULES:\n"
        "1. A menu item may span MULTIPLE OCR blocks (e.g., name + price + description).\n"
        "2. Include ALL block indices that belong to the menu item.\n"
        "3. Do NOT skip any visible menu items.\n"
        "4. If the menu has section headers (e.g., 'APPETIZERS'), you may skip them unless they are actual dishes.\n"
        "5. Return ONLY the JSON array, no markdown, no explanations."
    )

    user_prompt = (
        "Here are the OCR-detected text blocks from the menu image:\n\n"
        f"{ocr_summary}\n\n"
        "Now analyze the image and map each menu item to the correct block indices."
    )

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                        ],
                    },
                ],
                max_tokens=4096,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            # Markdown除去
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            menu_items = json.loads(content)
            if not isinstance(menu_items, list):
                raise ValueError("Expected a JSON array.")

            # Step 4: GPT-4oの判断に基づき、bboxを紐付け
            normalized = []
            for item in menu_items:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                if not name:
                    continue

                indices = item.get("block_indices", [])
                if not indices:
                    continue

                # ブロック統合してbbox算出
                bbox = _merge_blocks(ocr_blocks, indices)
                bbox_pixels = [
                    int(bbox[0] * img_w),
                    int(bbox[1] * img_h),
                    int(bbox[2] * img_w),
                    int(bbox[3] * img_h),
                ]

                normalized.append({
                    "name": name,
                    "price": str(item.get("price", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "bbox": bbox,
                    "bbox_pixels": bbox_pixels,
                    "matched_blocks": indices,
                })

            return normalized

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(
                    f"v5 extraction failed after {max_retries} attempts. Last error: {last_error}"
                ) from last_error

    return []
