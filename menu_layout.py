from io import BytesIO
from typing import List, Dict, Any, Optional

from PIL import Image, ImageDraw, ImageFont


def _get_fonts():
    """システムフォントのフォールバック付きでフォントを取得"""
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        price_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
            )
            price_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
            )
            desc_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
            )
        except Exception:
            title_font = ImageFont.load_default()
            price_font = title_font
            desc_font = title_font
    return title_font, price_font, desc_font


def create_bbox_debug_image(
    original_image: Image.Image,
    menu_items: List[Dict[str, Any]],
    color: str = "#FF0000",
) -> BytesIO:
    """
    元メニュー画像上に各料理項目の bbox（指定色の枠）と料理名（指定色の文字）を描画して返す。
    """
    canvas = original_image.copy()
    draw = ImageDraw.Draw(canvas)
    orig_w, orig_h = canvas.size

    for item in menu_items:
        bbox = item.get("bbox")
        name = item.get("name", "")
        if not bbox or len(bbox) != 4:
            continue

        x1 = int(bbox[0] * orig_w)
        y1 = int(bbox[1] * orig_h)
        x2 = int(bbox[2] * orig_w)
        y2 = int(bbox[3] * orig_h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 20)), name, fill=color)

    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output


def create_menu_image(
    menu_items: List[Dict[str, Any]],
    images: List[Image.Image],
    output_width: int = 1200,
    columns: int = 2,
) -> BytesIO:
    """
    メニュー項目と画像を1枚のメニュー画像にレイアウトして返す（グリッド版）。
    imagesリストはmenu_itemsと同じインデックス順、失敗時はNone。
    """
    title_font, price_font, desc_font = _get_fonts()

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

        draw.text((x, text_y), name, fill="#222222", font=title_font)
        text_y += 42

        if price:
            draw.text((x, text_y), str(price), fill="#D35400", font=price_font)
            text_y += 36

        if description:
            desc = str(description)
            max_chars = 60
            if len(desc) > max_chars:
                desc = desc[: max_chars - 3] + "..."
            draw.text((x, text_y), desc, fill="#555555", font=desc_font)

    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output


def create_simple_menu_image(
    original_image: Image.Image,
    images: List[Optional[Image.Image]],
    margin: int = 20,
    gap: int = 20,
) -> BytesIO:
    """
    元メニュー画像を左 70% に、生成画像群を右 30% に縦並びで配置する簡易レイアウト。
    """
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        raise ValueError("No valid images to layout.")

    orig_w, orig_h = original_image.size
    # 左側の幅は元画像の 70%、右側は 30%
    left_width = int(orig_w * 0.7)
    right_width = int(orig_w * 0.3)

    # 元画像を左側幅にリサイズ（縦横比維持）
    scale = left_width / orig_w
    left_height = int(orig_h * scale)
    left_img = original_image.resize((left_width, left_height), Image.LANCZOS)

    # 右側に並べる画像サイズ計算
    num_images = len(valid_images)
    available_height = left_height - margin * 2 - gap * (num_images - 1)
    thumb_height = max(100, available_height // num_images) if num_images > 0 else 100
    thumb_width = right_width - margin * 2

    # キャンバス高さは左側と右側の大きい方
    right_total_height = margin * 2 + num_images * thumb_height + gap * (num_images - 1)
    canvas_height = max(left_height, right_total_height)
    canvas_width = left_width + right_width

    canvas = Image.new("RGB", (canvas_width, canvas_height), "#FFFFFF")

    # 元画像を左側に貼り付け（上寄せ）
    canvas.paste(left_img, (0, 0))

    # 生成画像を右側に縦並び
    x_right = left_width + margin
    y_current = margin
    for img in valid_images:
        resized = img.resize((thumb_width, thumb_height), Image.LANCZOS)
        canvas.paste(resized, (x_right, y_current))
        y_current += thumb_height + gap

    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output


def create_annotated_menu_image(
    original_image: Image.Image,
    menu_items: List[Dict[str, Any]],
    images: List[Optional[Image.Image]],
    layout_mode: str = "layout2",
    margin: int = 20,
) -> BytesIO:
    """
    元メニュー画像に、各料理項目のテキスト右側に生成画像を配置する。
    layout_mode: "layout1" = 画像上端をテキスト上端に揃える
                 "layout2" = 画像中央をテキスト中央に揃える（デフォルト）
    bbox がない場合は create_simple_menu_image にフォールバックする。
    """
    orig_w, orig_h = original_image.size

    # 有効なペアを抽出
    valid_pairs = [
        (item, img)
        for item, img in zip(menu_items, images)
        if img is not None and item.get("bbox") and len(item.get("bbox")) == 4
    ]

    if not valid_pairs:
        return create_simple_menu_image(original_image, images)

    # 各画像の配置情報を先に計算し、キャンバス幅を動的に決定
    placements = []
    for item, img in valid_pairs:
        bbox = item["bbox"]
        x_text_right = int(bbox[2] * orig_w)
        y_top = int(bbox[1] * orig_h)
        y_bottom = int(bbox[3] * orig_h)
        target_height = max(50, y_bottom - y_top)

        # 縦横比維持でリサイズ
        img_ratio = img.width / img.height

        if layout_mode == "layout2":
            # レイアウト2: 画像を固定サムネイルサイズにして中央揃え
            new_height = min(150, target_height)
            new_width = int(new_height * img_ratio)
            if new_width > orig_w * 0.4:
                new_width = int(orig_w * 0.4)
                new_height = int(new_width / img_ratio)
            # 画像中央をテキストブロック中央に揃える
            text_center_y = (bbox[1] + bbox[3]) / 2 * orig_h
            y_img = int(text_center_y - new_height / 2)
        else:
            # レイアウト1: 画像高さをテキストブロック高さに合わせて上端揃え
            new_height = target_height
            new_width = int(new_height * img_ratio)
            if new_width > orig_w * 0.5:
                new_width = int(orig_w * 0.5)
                new_height = int(new_width / img_ratio)
            y_img = y_top

        x_img = x_text_right + margin

        placements.append((img, x_img, y_img, new_width, new_height))

    # キャンバス幅を動的計算（最も右端に来る画像 + 右マージン）
    max_right = max(x + w for _, x, _, w, _ in placements)
    canvas_width = max(orig_w, max_right + margin)
    canvas_height = orig_h

    canvas = Image.new("RGB", (canvas_width, canvas_height), "#FFFFFF")
    canvas.paste(original_image, (0, 0))

    for img, x, y, w, h in placements:
        resized = img.resize((w, h), Image.LANCZOS)
        canvas.paste(resized, (x, y))

    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output
