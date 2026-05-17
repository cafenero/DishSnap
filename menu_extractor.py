import base64
import json
import os
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

    # 必須キーの検証
    for item in menu_items:
        if "name" not in item:
            raise ValueError("Each menu item must have a 'name' key.")
        if "price" not in item:
            item["price"] = ""
        if "description" not in item:
            item["description"] = ""

    return menu_items
