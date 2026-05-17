import json
from typing import List, Dict, Any

from openai import OpenAI


def refine_bboxes_with_gpt4o(
    menu_items: List[Dict[str, Any]],
    ocr_blocks: List[Dict[str, Any]],
    client: OpenAI,
) -> List[Dict[str, Any]]:
    """
    GPT-4o を使って OCR ブロックとメニュー項目のマッチングを検証・補正する。

    処理内容:
    1. 各メニュー項目に紐づいた OCR ブロックが正しいか確認
    2. 欠落しているメニュー項目の bbox を推定
    3. 誤って統合されたブロックの指摘

    戻り値:
        補正された menu_items（bbox 付き）
    """
    if not menu_items or not ocr_blocks:
        return menu_items

    # GPT-4o への入力を構築
    prompt = _build_refinement_prompt(menu_items, ocr_blocks)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a restaurant menu OCR validator. "
                        "Given menu items extracted by GPT-4o and text blocks with bounding boxes from OCR, "
                        "your task is to:\n"
                        "1. Verify that each menu item has a correct bounding box from the OCR blocks\n"
                        "2. Identify any menu items missing a bbox (the OCR block was not found)\n"
                        "3. Reassign incorrect bbox matches\n"
                        "4. Return a JSON array with corrected bboxes\n\n"
                        "Rules:\n"
                        "- Use ONLY the provided OCR block bboxes\n"
                        "- Do not invent new coordinates\n"
                        "- If a menu item has no matching OCR block, set bbox to null\n"
                        "- Each OCR block can be assigned to at most one menu item\n"
                        "- Prefer the OCR block whose text contains the menu item name"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=2048,
            temperature=0.0,
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

        refined = json.loads(content)

        if not isinstance(refined, list):
            return menu_items

        # 結果をマージ
        for original, corrected in zip(menu_items, refined):
            if corrected.get("bbox") is not None:
                original["bbox"] = corrected["bbox"]
            # confidence があれば更新
            if "confidence" in corrected:
                original["confidence"] = corrected["confidence"]

    except Exception as e:
        print(f"GPT-4o refinement failed: {e}")
        # 失敗しても元の menu_items を返す

    return menu_items


def _build_refinement_prompt(
    menu_items: List[Dict[str, Any]],
    ocr_blocks: List[Dict[str, Any]],
) -> str:
    """
    GPT-4o へのプロンプトを構築する。
    """
    lines = [
        "## Menu Items (extracted by GPT-4o)",
        "```json",
        json.dumps(menu_items, ensure_ascii=False, indent=2),
        "```",
        "",
        "## OCR Text Blocks (from PaddleOCR with bboxes)",
        "```json",
        json.dumps(ocr_blocks, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Task",
        "For each menu item, find the best matching OCR block and assign its bbox.",
        "Return ONLY a JSON array in the same order as menu items:",
        '[\n  {"name": "...", "bbox": [x1, y1, x2, y2] or null, "confidence": float}\n]',
    ]

    return "\n".join(lines)
