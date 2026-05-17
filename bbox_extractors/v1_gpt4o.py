"""
BBox Extractor v1: GPT-4o 直接抽出（最初の実装）
GPT-4o Vision に bbox 抽出を依頼し、結果をそのまま使用する。
"""
import io
from typing import List, Dict, Any

from PIL import Image


def extract_text_bboxes_v1(
    image: Image.Image,
    client=None,  # OpenAI client
    **kwargs
) -> List[Dict[str, Any]]:
    """
    GPT-4o Vision を使用してテキストブロックと bbox を抽出する。
    client が未提供の場合は空リストを返す。
    """
    if client is None:
        # client がない場合は実行不可
        return []

    import base64
    import json

    # 画像を base64 エンコード
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_image}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at reading restaurant menus from photos. "
                    "Extract each menu item as a JSON object with keys: name, price, description, bbox. "
                    "Return ONLY a JSON array of objects, nothing else. "
                    "If a description is not visible, use a short generic description. "
                    "Price should be a string (e.g., '$12.99' or '¥1,200'). "
                    "bbox should be a relative bounding box [x1, y1, x2, y2] "
                    "where each value is between 0.0 and 1.0, representing the top-left "
                    "and bottom-right corners of the menu item text area in the image. "
                    "If you cannot determine the bbox, omit the key."
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

    # 結果を統一フォーマットに変換
    results = []
    for item in menu_items:
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            results.append({
                "text": f"{item.get('name', '')} {item.get('price', '')} {item.get('description', '')}".strip(),
                "bbox": bbox,
                "confidence": 1.0,
            })

    return results
