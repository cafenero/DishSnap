import base64
import json
from difflib import SequenceMatcher
from io import BytesIO
from typing import List, Dict, Any

from openai import OpenAI


def encode_image_to_base64(image: BytesIO) -> str:
    """画像をbase64エンコードしてdata URI形式で返す"""
    image.seek(0)
    return base64.b64encode(image.read()).decode("utf-8")


def extract_menu_items(image: BytesIO, client: OpenAI) -> List[Dict[str, Any]]:
    """
    GPT-4o Visionを使ってメニュー画像から料理情報を抽出する。
    bbox は含まない（後でPaddleOCRの結果とマッチングする）。

    戻り値: [{"name": str, "price": str, "description": str}, ...]
    """
    b64_image = encode_image_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{b64_image}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at reading restaurant menus from photos. "
                    "Extract each menu item as a JSON object with keys: name, price, description. "
                    "Return ONLY a JSON array of objects, nothing else. "
                    "If a description is not visible, use a short generic description. "
                    "Price should be a string (e.g., '$12.99' or '¥1,200')."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            },
        ],
        max_tokens=2048,
    )

    content = response.choices[0].message.content.strip()

    # JSONコードブロックを取り除く
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    menu_items = json.loads(content)
    if not isinstance(menu_items, list):
        raise ValueError("Expected a JSON array of menu items.")

    # 必須キーの検証と正規化
    normalized = []
    for item in menu_items:
        if "name" not in item:
            raise ValueError("Each menu item must have a 'name' key.")

        price = item.get("price")
        description = item.get("description")

        normalized.append({
            "name": item["name"],
            "price": price if price is not None else "",
            "description": description if description is not None else "",
            "bbox": None,
        })

    return normalized


def match_bboxes(
    menu_items: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
    similarity_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    GPT-4o で抽出したメニュー項目と、PaddleOCR で抽出したテキストブロックを照合し、
    最も類似度の高い bbox を各メニュー項目に紐づける。

    マッチングロジック:
    1. メニュー名が OCR テキストに部分含まれる場合、高スコア
    2. 部分一致しない場合は、SequenceMatcher で類似度計算
    """
    for item in menu_items:
        name = item["name"].lower().strip()
        best_match = None
        best_score = 0.0

        for ocr in ocr_results:
            text = ocr["text"].lower().strip()

            if not text:
                continue

            # 部分一致スコアリング
            if name in text:
                # メニュー名が OCR テキストに含まれる場合
                # 一致部分の比率が高いほど良い
                score = len(name) / max(len(text), len(name))
            else:
                # 部分一致しない場合は文字列類似度
                score = SequenceMatcher(None, name, text).ratio()

            if score > best_score:
                best_score = score
                best_match = ocr

        # 閾値以上のスコアがあれば bbox を紐づける
        if best_match and best_score >= similarity_threshold:
            item["bbox"] = best_match["bbox"]

    return menu_items
