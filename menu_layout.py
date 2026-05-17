from io import BytesIO
from typing import List, Dict, Any

from PIL import Image, ImageDraw, ImageFont


def create_menu_image(
    menu_items: List[Dict[str, Any]],
    images: List[Image.Image],
    output_width: int = 1200,
    columns: int = 2,
) -> BytesIO:
    """
    メニュー項目と画像を1枚のメニュー画像にレイアウトして返す。
    imagesリストはmenu_itemsと同じインデックス順、失敗時はNone。
    """
    # フォント設定（システムフォントのフォールバック）
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        price_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            price_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except Exception:
            title_font = ImageFont.load_default()
            price_font = title_font
            desc_font = title_font

    # 有効なアイテム（画像生成成功したもの）のみ抽出
    valid_pairs = [(item, img) for item, img in zip(menu_items, images) if img is not None]
    if not valid_pairs:
        raise ValueError("No valid images to layout.")

    # グリッド計算
    num_items = len(valid_pairs)
    rows = (num_items + columns - 1) // columns

    margin = 40
    gap_x = 40
    gap_y = 40
    image_width = (output_width - margin * 2 - gap_x * (columns - 1)) // columns
    image_height = image_width  # 正方形画像を想定
    text_area_height = 120  # 料理名・価格・説明のためのスペース
    cell_height = image_height + text_area_height + gap_y

    output_height = margin * 2 + rows * cell_height + gap_y * (rows - 1)

    # キャンバス作成
    canvas = Image.new("RGB", (output_width, output_height), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)

    for idx, (item, img) in enumerate(valid_pairs):
        col = idx % columns
        row = idx // columns

        x = margin + col * (image_width + gap_x)
        y = margin + row * cell_height

        # 画像をリサイズして貼り付け
        img_resized = img.resize((image_width, image_height), Image.LANCZOS)
        canvas.paste(img_resized, (x, y))

        # テキスト描画
        text_y = y + image_height + 10
        name = item.get("name", "Unknown")
        price = item.get("price", "")
        description = item.get("description", "")

        # 料理名（太字風に大きく）
        draw.text((x, text_y), name, fill="#222222", font=title_font)
        text_y += 42

        # 価格
        if price:
            draw.text((x, text_y), str(price), fill="#D35400", font=price_font)
            text_y += 36

        # 説明（2行までに切る）
        if description:
            desc = str(description)
            # 簡易的な文字数制限
            max_chars = 60
            if len(desc) > max_chars:
                desc = desc[:max_chars - 3] + "..."
            draw.text((x, text_y), desc, fill="#555555", font=desc_font)

    # BytesIOに保存
    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output
