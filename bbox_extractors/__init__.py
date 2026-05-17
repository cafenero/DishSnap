"""
BBox抽出器のバージョン切り替えファクトリ
"""
from typing import List, Dict, Any
from PIL import Image


def extract_text_bboxes_versioned(
    image: Image.Image,
    version: str = "v2_paddle_simple",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    指定バージョンの BBox 抽出器を使用してテキストブロックまたはメニュー項目を抽出する。

    Args:
        image: PIL Image
        version: 抽出器バージョン
            - "v1_gpt4o": GPT-4o 直接 bbox 抽出
            - "v2_paddle_simple": PaddleOCR + 固定閾値マージ（水平30%・80%達成版）
            - "v3_paddle_advanced": PaddleOCR + マルチスケール + K-means + GPT-4o
            - "v4_paddle_fixed": PaddleOCR + 単一スケール + 固定閾値50%
            - "v5_gpt4o_direct": GPT-4o Vision 直接メニュー抽出（name/price/description/bbox）
        **kwargs: 各抽出器に渡す追加引数

    Returns:
        List[Dict]: テキストブロックまたはメニュー項目のリスト
            v1-v4: [{"text": str, "bbox": [x1,y1,x2,y2]}, ...]
            v5: [{"name": str, "price": str, "description": str, "bbox": [x1,y1,x2,y2]}, ...]
    """
    if version == "v1_gpt4o":
        from .v1_gpt4o import extract_text_bboxes_v1
        return extract_text_bboxes_v1(image, **kwargs)
    elif version == "v2_paddle_simple":
        from .v2_paddle_simple import extract_text_bboxes_v2
        return extract_text_bboxes_v2(image, **kwargs)
    elif version == "v3_paddle_advanced":
        from .v3_paddle_advanced import extract_text_bboxes_v3
        return extract_text_bboxes_v3(image, **kwargs)
    elif version == "v4_paddle_fixed":
        from .v4_paddle_fixed import extract_text_bboxes_v4
        return extract_text_bboxes_v4(image, **kwargs)
    elif version == "v5_gpt4o_direct":
        from .v5_gpt4o_direct import extract_menu_items_v5
        return extract_menu_items_v5(image, **kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")
